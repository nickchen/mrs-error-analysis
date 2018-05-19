#!/usr/bin/env python

import os
import sys
import json
import argparse
import penman as PM
import traceback
import shlex
import re

from collections import defaultdict
from functools import partial

from delphin.codecs import simplemrs
from delphin.mrs import xmrs, eds, penman

class Arg(object):
    def __init__(self, level, arg_tokens):
        self._args = {}
        self._level = level
        self._end = True
        if isinstance(arg_tokens, list):
            arg_tokens = iter(arg_tokens)
        if Arg.parse(self, arg_tokens):
            self._end = False

    @staticmethod
    def parse(obj, arg_tokens):
        has_more_args = False
        try:
            token = next(arg_tokens)
            while True:
                if token.startswith("ARG"):
                    if token != "ARG":
                        assert token == ("ARG%s" % (obj._level)), "first arg should be same level"
                    arg_type = arg_tokens.next()
                    if arg_type not in obj._args:
                        obj._args[arg_type] = Arg(obj._level + 1, arg_tokens)
                        has_more_args = not obj._args[arg_type]._end
                    else:
                        has_more_args = Arg.parse(obj._args[arg_type], arg_tokens)
                token = next(arg_tokens)
        except StopIteration:
            pass
        return has_more_args

    def print_erg(self):
        # print "HERE", self._level, self._end, self._args
        for key, erg in self._args.iteritems():
            print "%s%s" % (self._level * " ", key)
            erg.print_erg()

    def level_args(self, level):
        args = []
        for arg, value  in self._args.iteritems():
            if self._level == level:
                args.append(arg)
            args += value.level_args(level)
        return args

class ErgPredicate(Arg):
    def __init__(self):
        super(ErgPredicate, self).__init__(0, [])

    def parse_args(self, predicate, args):
        self._predicate = predicate
        arg_tokens = iter(re.split("\s|\,|\.", args))
        Arg.parse(self, arg_tokens)

class Erg(object):
    def __init__(self, input):
        self._ergs = defaultdict(ErgPredicate)
        self.parse(input)

    def parse(self, input):
        head = input.next()
        input.next()
        self._counts = defaultdict(int)
        for line in input:
            (predicate, args) = line.strip().split(":")
            predicate = predicate.strip()
            args = args.strip()
            self._ergs[predicate].parse_args(predicate, args)
            self._counts[predicate] += 1

    def __contains__(self, predicate):
        return predicate in self._ergs

    def __getitem__(self, predicate):
        return self._ergs[predicate]

class EdmPredicate(object):
    def __init__(self, start, end, sentence):
        self.start  = int(start)
        self.end = int(end)
        self.len = self.end - self.start
        self.sentence = sentence
        self.span = sentence[self.start:self.end]
        self.predicates = []
        self.carg = []
        self.args = []

    def append(self, predicate):
        self.predicates.append(predicate.rstrip("_rel"))
        self.predicates = sorted(self.predicates)

    def carg(self, value):
        self.carg.append(value)

    def arg(self, name, value):
        self.args.append({'name':name, 'index':value})

    def __str__(self):
        return "<%s>:<%s>" % (self.predicates, self.args)

    def parse_arg_level(self, arg_name):
        arg_names = arg_name.split("/")
        if arg_names[0] == "ARG":
            return arg_names[0], 0
        return arg_names[0], int(arg_names[0][3:])

    def is_same_span(self, other):
        return self.start == other.start and self.end == other.end

    def get_types(self, erg_dict):
        ts = []
        for p in self.predicates:
            if "_u_" in p:
                ts.append("u")
            else:
                if p in erg_dict:
                    erg_pred = erg_dict[p]
                    ts += erg_pred.level_args(0)
        return ts

    def external_args(self):
        """find all the outgoing args, where the arg_predicate span is not self"""
        already_yield = []
        for arg in self.args:
            if arg['name'].startswith("ARG"):
                arg_predicate = arg['predicate']
                if not arg_predicate.is_same_span(self):
                    arg_name, arg_level = self.parse_arg_level(arg['name'])
                    if arg_name not in already_yield:
                        yield arg_name, arg_level, arg_predicate
                        already_yield.append(arg_name)

    def match_args(self, predicate_str, erg_dict):
        errors = defaultdict(int)
        erg_pred = erg_dict[predicate_str]
        for arg_name, arg_level, arg_predicate in self.external_args():
            erg_level_types = erg_pred.level_args(arg_level)
            if len(erg_level_types) > 0:
                # only do arg comparsion if erg have args
                # otherwise it's extra arg
                arg_predicate_types = arg_predicate.get_types(erg_dict)
                # XXX assume "x" is wildcard
                if len(set(erg_level_types) & set(arg_predicate_types)) == 0 and not "x" in arg_predicate_types:
                    errors["%s incorrect - %s" % (arg_name, predicate_str)] += 1
                    # print arg_name, arg_level, arg_predicate
            else:
                errors["%s extra - %s" % (arg_name, predicate_str)] += 1
        # print errors
        return errors

class EdmContainer(object):
    def __init__(self, sentence):
        self._sentence = sentence
        self._entries = {}

    def parse(self, line):
        for item in line.split(";"):
            if len(item.strip()) == 0:
                continue
            (index, typename, typevalue) = item.strip().split(" ")
            if index not in self._entries:
                (start, end) = index.split(":")
                self._entries[index] = EdmPredicate(start, end, self._sentence)
            if typename == "NAME":
                self._entries[index].append(typevalue)
            elif typename == "CARG":
                self._entries[index].carg.append(typevalue)
            else:
                self._entries[index].arg(typename, typevalue)

    def align_with(self, other):
        """Using self as the model, make the other align with self"""
        for index, self_predicate in self._entries.iteritems():
            if index not in other._entries and self_predicate.span.endswith((".")):
                # candidate for alignment fix
                other_key = "%d:%d" % (self_predicate.start, self_predicate.end - 1)
                if other_key in other._entries:
                    other_predicate = other._entries[other_key]
                    del other._entries[other_key]
                    if index not in other._entries:
                        other._entries[index] = other_predicate
                        other_predicate.start = self_predicate.start
                        other_predicate.end = self_predicate.end
                        other_predicate.span = self_predicate.span
                        other_predicate.len = self_predicate.len
                    else:
                        other._entries[index].merge(other_predicate)

    def edm_link_args(self):
        for index, predicate in self._entries.iteritems():
            for d in predicate.args:
                assert d['index'] in self._entries
                d['predicate'] = self._entries[d['index']]
                del d['index']

    def match_args(self, index, predicate_str, erg):
        predicate = self._entries[index]
        return predicate.match_args(predicate_str, erg)

class Processor(object):
    def __init__(self, argparse_ns):
        self.__package_amr_loads = partial(penman.loads, model=xmrs.Dmrs)
        self.mrs = []
        self.gold = []
        self._gold_edm = []
        self.system = []
        self._system_edm = []
        self._edm_compare_result = []
        self.out_dir = None
        self._files = {}
        self._limit = argparse_ns.limit
        self.erg = None
        self.amr_loads = self._local_amr_loads
        if hasattr(argparse_ns, "out_dir"):
            self.out_dir = argparse_ns.out_dir
        for f in ("ace", "system", "gold", "erg", "system_edm", "gold_edm", "abstract"):
            if hasattr(argparse_ns, f):
                self._files[f] = getattr(argparse_ns, f)

    def _package_amr_loads(self, s):
        xs = self.__package_amr_loads(s)
        return xs[0]

    def _local_amr_loads(self, s):
        graphs = PM.loads(s, cls=penman.XMRSCodec)
        assert len(graphs) == 1, "only one graph"
        triples = graphs[0].triples()
        # triples = sorted(triples, key=lambda a: a.source)
        try:
            return xmrs.Dmrs.from_triples(triples)
        except Exception as e:
            raise

    def parse_edm_file(self, input):
        assert len(self.mrs) > 0, "MRS parsed already"
        ret = []
        i = 0
        for line in input:
            e = EdmContainer(self.mrs[i].get('sentence'))
            e.parse(line)
            ret.append(e)
            i += 1
            if self._limit > 0 and len(ret) >= self._limit:
                break
        return ret

    def edm_post_process(self):
        self.edm_link_args()
        self.edm_align()
        self.edm_analyze()

    def edm_link_args(self):
        for i in range(len(self._system_edm)):
            self._system_edm[i].edm_link_args()

    def edm_align(self):
        assert len(self._gold_edm) == len(self._system_edm), "same length"
        for i in range(len(self._gold_edm)):
            self._gold_edm[i].align_with(self._system_edm[i])

    def load_edm(self):
        self.load_erg()
        if len(self.mrs) == 0:
            self.parse_mrs(self._files["ace"])
        self.parse_edm_gold(self._files["gold_edm"])
        self.parse_edm_system(self._files["system_edm"])
        self.edm_post_process()

    def load_json(self):
        self.parse_mrs(self._files["ace"])
        self.load_edm()
        self.parse_amr_system(self._files["system"])
        self.parse_amr_gold(self._files["gold"])

    def load_erg(self):
        if self.erg is None:
            self.erg = Erg(self._files["erg"])
            self.erg.parse(self._files["abstract"])

    def to_json(self, indent=2):
        assert(len(self.mrs) == len(self.system))
        assert(len(self.mrs) == len(self.gold))
        for i in range(len(self.mrs)):
            readings = 0
            result = {
                "input" : self.mrs[i]['sentence'],
                "results" : [],
            }
            if 'mrs' in self.mrs[i]:
                result["results"].append(
                    {"result-id": "ACE MRS",
                     "mrs": xmrs.Mrs.to_dict(self.mrs[i]['mrs'], properties=True)})
                readings += 1
            if 'dmrs' in self.gold[i]:
                result["results"].append(
                    {"result-id": "Gold DMRS (convert from penman)",
                     "dmrs": xmrs.Dmrs.to_dict(self.gold[i]['dmrs'], properties=True)})
                readings += 1
            if 'dmrs' in self.system[i]:
                result["results"].append(
                    {"result-id": "System DMRS (convert from penman)",
                     "dmrs": xmrs.Dmrs.to_dict(self.system[i]['dmrs'], properties=True)})
                readings += 1
            if len(self._gold_edm) != 0 and len(self._gold_edm) == len(self._system_edm):
                result["results"].append({"result-id": "EDM", "edm": self._edm_compare_result[i]})
                readings += 1
            result["readings"] = readings
            file_outpath = os.path.join(self.out_dir, "n%s.json" % i)
            with open(file_outpath, "w") as f:
                f.write(json.dumps(result, indent=indent))

    def convert_mrs(self, mrs, properties=True, indent=None):
        if len(mrs) == 1:
            return {'sentence': mrs[0][len("SKIP: "):]}
        else:
            # CLS = partial(penman.dumps, model=xmrs.Dmrs)
            sent = mrs[0]
            xs = simplemrs.loads_one(" ".join(mrs[1:]))
            if isinstance(xs, xmrs.Mrs):
                x = xs
            else:
                x = xmrs.Mrs.from_xmrs(xs)
            return {'sentence': sent[len("SENT: "):], 'mrs': x}

    def _mrs_file_to_lines(self, input):
        mrs = []
        for line in input:
            line = line.strip()
            if len(line) == 0:
                if len(mrs) == 0: continue
                yield mrs
                mrs = []
                continue
            mrs.append(line)
        if len(mrs) > 0:
            yield mrs


    def parse_mrs(self, input):
        for mrs in self._mrs_file_to_lines(input):
            self.mrs.append(self.convert_mrs(mrs))
            if self._limit > 0 and len(self.mrs) >= self._limit:
                break

    def convert_amr(self, lines):
        amr_string = "\n".join(lines)
        amr_string = amr_string.replace("|", "-")
        try:
            xmrs_obj = self.amr_loads(amr_string.strip())
            x = xmrs.Dmrs.from_xmrs(xmrs_obj)
            return {'amr': amr_string, 'dmrs':x}
        except Exception as e:
            traceback.print_exc()

        return {'amr': amr_string}

    def _parse_amr_file_to_lines(self, input):
        lines = []
        for line in input:
            line = line.rstrip()
            if len(line) == 0:
                if len(lines) > 0:
                    yield lines
                lines = []
                continue
            lines.append(line)
        if len(lines) > 0:
            yield lines

    def parse_amr_file(self, input):
        out_list = []
        for lines in self._parse_amr_file_to_lines(input):
            amr = self.convert_amr(lines)
            if amr is not None:
                out_list.append(amr)
            if self._limit > 0 and len(out_list) >= self._limit:
                break
        return out_list

    def parse_edm_gold(self, input):
        self._gold_edm = self.parse_edm_file(input)

    def parse_edm_system(self, input):
        self._system_edm = self.parse_edm_file(input)

    def parse_amr_gold(self, input):
        self.gold = self.parse_amr_file(input)
        self.gold_error = len([s for s in self.system if s.get('dmrs') is None])

    def parse_amr_system(self, input):
        self.system = self.parse_amr_file(input)
        self.system_error = len([s for s in self.system if s.get('dmrs') is None])

    def edm_analyze(self):
        self.edm_system_gold_compare()

    def validate_args(self, system, index, predicate):
        # print predicate
        assert self.erg is not None, "erg defined"
        assert predicate in self.erg, "predicate in erg"
        return system.match_args(index, predicate, self.erg)

    def edm_system_gold_compare(self):
        assert len(self._gold_edm) == len(self._system_edm), "same length"
        for i in range(len(self._gold_edm)):
            d = self._edm_dict(i, self._gold_edm[i], self._system_edm[i])
            self._edm_compare_result.append(d)

    def _edm_dict(self, i, gold, system):
        gold_set = set([k for k in gold._entries.iterkeys()])
        system_set = set([k for k in system._entries.iterkeys()])
        predicates = {}
        stats = defaultdict(int)
        for index in list(gold_set | system_set):
            pred = gold._entries[index] if index in gold._entries else system._entries[index]
            predicates[index] = {
                "start": pred.start,
                "end": pred.end,
                "len": (pred.end - pred.start),
                "span": pred.span,
                "predicate" : {}
            }
            gold_index_predicates = set(gold._entries[index].predicates) if index in gold_set else set()
            system_index_predicates = set(system._entries[index].predicates) if index in system_set else set()
            stats['total'] += len(system_index_predicates | gold_index_predicates)
            stats['gold'] += len(gold_index_predicates - system_index_predicates)
            stats['system'] += len(system_index_predicates - gold_index_predicates)
            stats['common'] += len(system_index_predicates & gold_index_predicates)
            if len(gold_index_predicates) > 0:
                predicates[index]["predicate"]["gold"] = list(gold_index_predicates)
            if len(system_index_predicates) > 0:
                predicates[index]["predicate"]["system"] = list(system_index_predicates)
            def stats_update(prefix, predicates):
                for predicate in predicates:
                    if predicate.endswith("unknown"):
                        stats["%s_%s"% (prefix, 'unknown')] += 1
                    elif predicate in ['compound', 'udef_q', 'proper_q', 'yofc', 'named', 'subord', 'card']:
                        stats["%s_%s"% (prefix, predicate)] += 1
                    elif prefix == "system":
                        if predicate not in self.erg:
                            stats['predicate'] += 1
                        else:
                            pred_errors = self.validate_args(system, index, predicate)
                            if len(pred_errors) > 0:
                                stats['predicate_arg'] += 1
                                if "predicate_errors" not in stats:
                                    stats['predicate_errors'] = defaultdict(int)
                                for name, count in pred_errors.iteritems():
                                    stats['predicate_errors'][name] += count
            stats_update("system", system_index_predicates - gold_index_predicates)
            stats_update("gold", gold_index_predicates - system_index_predicates)

        return {'predicates': predicates, 'stats': stats}

def process_main(ns):
    p = Processor(ns)

def process_main_error(ns):
    p = Processor(ns)
    p.edm_analyze()

def process_main_edm(ns):
    p = Processor(ns)
    p.load_edm()

def process_main_erg(ns):
    p = Processor(ns)
    p.load_erg()

def process_main_json(ns):
    p = Processor(ns)
    p.load_json()
    p.to_json()
    print "total: %d gold: {%d/%d} system: {%d/%d}" % (
        len(p.mrs),
        len(p.gold), p.gold_error,
        len(p.system), p.system_error)

def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(prog=sys.argv[0])
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--ace', type=argparse.FileType('r'),
            default="../../error-analysis-data/dev.erg.mrs")
    subparsers = parser.add_subparsers()

    parser_json = subparsers.add_parser("json")
    parser_json.add_argument('--system', type=argparse.FileType('r'),
            default="../../error-analysis-data/dev.system.dmrs.amr")
    parser_json.add_argument('--gold', type=argparse.FileType('r'),
            default="../../error-analysis-data/dev.gold.dmrs.amr")
    parser_json.add_argument('--gold_edm', type=argparse.FileType('r'),
            default="../../error-analysis-data/dev.gold.edm")
    parser_json.add_argument('--system_edm', type=argparse.FileType('r'),
            default="../../error-analysis-data/dev.system.dmrs.edm")
    parser_json.add_argument('--out_dir', type=str, default="../webpage/data/")
    parser_json.add_argument('--erg', type=argparse.FileType('r'),
            default="../../erg1214/etc/surface.smi")
    parser_json.add_argument('--abstract', type=argparse.FileType('r'),
            default="../../erg1214/etc/abstract.smi")
    parser_json.set_defaults(func=process_main_json)

    parser_error = subparsers.add_parser("error")
    parser_error.add_argument('--erg', type=argparse.FileType('r'),
            default="../../erg1214/etc/surface.smi")
    parser_error.set_defaults(func=process_main_error)

    parser_edm = subparsers.add_parser("edm")
    parser_edm.add_argument('--gold_edm', type=argparse.FileType('r'),
            default="../../error-analysis-data/dev.gold.edm")
    parser_edm.add_argument('--system_edm', type=argparse.FileType('r'),
            default="../../error-analysis-data/dev.system.dmrs.edm")
    parser_edm.add_argument('--erg', type=argparse.FileType('r'),
            default="../../erg1214/etc/surface.smi")
    parser_edm.add_argument('--abstract', type=argparse.FileType('r'),
            default="../../erg1214/etc/abstract.smi")
    parser_edm.set_defaults(func=process_main_edm)

    parser_erg = subparsers.add_parser("erg")
    parser_erg.add_argument('--erg', type=argparse.FileType('r'),
            default="../../erg1214/etc/surface.smi")
    parser_erg.add_argument('--abstract', type=argparse.FileType('r'),
            default="../../erg1214/etc/abstract.smi")
    parser_erg.set_defaults(func=process_main_erg)

    parser.set_defaults(func=process_main)
    ns = parser.parse_args(args)
    ns.func(ns)


if __name__ == "__main__":
    main()

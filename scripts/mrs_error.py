#!/usr/bin/env python

import os
import sys
import json
import argparse
import penman as PM
import traceback
import shlex
import re

from collections import defaultdict, OrderedDict
from functools import partial

from delphin.codecs import simplemrs
from delphin.mrs import xmrs, eds, penman

class Arg(object):
    def __init__(self, level):
        self._args = {}
        self._level = level
        self._end = True

    def print_erg(self):
        # print "HERE", self._level, self._end, self._args
        for key, erg in self._args.iteritems():
            print "%s%s" % (self._level * " ", key)
            erg.print_erg()

    def level_args(self, level):
        args = set([])
        for arg, value in self._args.iteritems():
            if self._level == level:
                args.add(arg)
            if not isinstance(value, bool):
                args.update(value.level_args(level))
        return args

class ErgPredicate(Arg):
    def __init__(self):
        super(ErgPredicate, self).__init__(0)

    def parse_args(self, predicate, args):
        self._predicate = predicate
        arg_tokens = iter(re.split("\s|\,|\.", args))

        def get_args(tokens):
            args = None
            try:
                while True:
                    token = next(tokens)
                    if token.startswith("ARG"):
                        if args is not None:
                            yield args
                        args = [token, tokens.next()]
                    elif token in ("RSTR",):
                        args.append(token)
            except StopIteration:
                if args is not None:
                    yield args
        obj = self
        for arg_list in get_args(arg_tokens):
            obj._end = False
            arg = arg_list[0]
            arg_type = arg_list[1]
            if len(arg_list) > 2:
                obj._args[arg_list[-1]] = True
            if arg_type not in obj._args:
                obj._args[arg_type] = Arg(obj._level + 1)
            obj = obj._args[arg_type]

class Erg(object):
    def __init__(self, input):
        self._ergs = defaultdict(ErgPredicate)
        self.parse(input)

    def parse(self, input):
        head = input.next()
        input.next()
        for line in input:
            (predicate, args) = line.strip().split(":")
            predicate = predicate.strip()
            args = args.strip()
            self._ergs[predicate].parse_args(predicate, args)

    def __contains__(self, predicate):
        return predicate in self._ergs

    def __getitem__(self, predicate):
        return self._ergs[predicate]

class EdmPredicate(object):
    def __init__(self, start, end, sentence, sentence_index):
        self.start  = int(start)
        self.end = int(end)
        self.len = self.end - self.start
        self._sentence_index = sentence_index
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

    def get_types(self, erg_dict, arg_level):
        ts = []
        for p in self.predicates:
            if "_u_" in p:
                ts.append("u")
            else:
                if p in erg_dict:
                    erg_pred = erg_dict[p]
                    ts += erg_pred.level_args(arg_level)
        return ts

    def candidate_args_for_validate(self):
        """find candidate arg for validateion:
                o ARGX - none self referenced - external reference args
                o RSTR - self referenced -
        """
        already_yield = []
        for arg in self.args:
            # for all the arguments
            # find all the args starting with ARGX
            # and return the name, level, and arg_predicate
            arg_predicate = arg['predicate']
            if arg['name'].startswith("ARG") and arg['name'] != "ARG":
                if not arg_predicate.is_same_span(self):
                    arg_name, arg_level = self.parse_arg_level(arg['name'])
                    if arg_name not in already_yield:
                        yield arg_name, arg_level, arg_predicate
                        already_yield.append(arg_name)
            elif arg['name'].startswith("RSTR"):
                if arg_predicate.is_same_span(self):
                    if arg['name'] not in already_yield:
                        yield arg['name'], 0, arg['predicate']
                        already_yield.append(arg['name'])

    def erg_pred_types_at_level(self, arg_level, erg_dict):
        erg_level_types = set()
        for pred in self.predicates:
            if pred in erg_dict:
                erg_pred = erg_dict[pred]
                erg_level_types.update(erg_pred.level_args(arg_level))
        return erg_level_types

    def validate_args(self, predicate_str, erg_dict):
        """Given the erg dictionary, validate the predicate designated by predicate_strself.
                We have the predicate_str, and erg_dict, but not the predicate arguments.
                Since we can have multiple predicates for the span, it's not possible to use
                the EDM information to figure out the predicate and ARG relationship. So
                we do a bucket instead.
        """
        errors = StatsKeeper()
        for arg_name, arg_level, arg_predicate in self.candidate_args_for_validate():
            # get the possible types at this level
            # this includes RSTR
            erg_level_types = self.erg_pred_types_at_level(arg_level, erg_dict)
            # if arg_name.startswith("RSTR"):
            #     print erg_level_types
            if len(erg_level_types) > 0:
                # only do arg comparsion if erg have args
                # otherwise it's extra arg
                arg_predicate_types = arg_predicate.get_types(erg_dict, arg_level)
                # XXX assume "x" is wildcard
                predicate_types_union = set(erg_level_types) & set(arg_predicate_types)
                if len(predicate_types_union) == 0 and not "x" in arg_predicate_types:
                    assert predicate_str in self.predicates, "is own predicate"
                    errors.restart(["incorrect"])
                    errors["count"] += 1
                    errors.restart(["incorrect", arg_name])
                    errors[predicate_str] += 1
                    errors["count"] += 1
            else:
                print arg_name, predicate_str
                assert predicate_str in erg_dict, "in erg"
                # extra arg - accounting
                errors.restart(["extra"])
                errors["count"] += 1
                errors.restart(["extra", arg_name])
                errors[predicate_str] += 1
                errors["count"] += 1
        # print errors
        return errors

class Container(object):
    def __init__(self, sentence, index):
        self._sentence = sentence
        self._sentence_index = index

class EdmContainer(object):
    def __init__(self, sentence, index):
        self._sentence = sentence
        self._sentence_index = index
        self._entries = {}

    def parse(self, line):
        for item in line.split(";"):
            if len(item.strip()) == 0:
                continue
            (index, typename, typevalue) = item.strip().split(" ")
            if index not in self._entries:
                (start, end) = index.split(":")
                self._entries[index] = EdmPredicate(start, end, self._sentence, self._sentence_index)
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

    def validate_args(self, index, predicate_str, erg):
        predicate = self._entries[index]
        return predicate.validate_args(predicate_str, erg)

class StatsKeeper(object):
    def __init__(self):
        self._stats = {}
        self._node = self._stats

    def push(self, key):
        if key not in self._node:
            self._node[key] = defaultdict(int)
        self._node = self._node[key]

    def restart(self, keys):
        self._node = self._stats
        for key in keys:
            self.push(key)

    def __getitem__(self, key):
        return self._node[key]

    def __setitem__(self, key, value):
        self._node[key] = value

    def merge(self, other):
        StatsKeeper.merge_dict(self._stats, other._stats)

    def merge_node(self, other):
        StatsKeeper.merge_dict(self._node, other._stats)

    def to_dict(self):
        return self._stats

    def trim(self, keys, limit=50, count=0, debug=False):
        self.restart(keys[0:-1])
        # replacement
        last = keys[-1]
        self._node[last] = self._sort_limit(self._node[last], limit, count, debug=debug)

    def _sort_limit(self, node, limit, count, debug=False):
        r = OrderedDict()
        def sort_func(k):
            if isinstance(node[k], int):
                return node[k]
            else:
                return sys.maxint
        if isinstance(node, dict):
            for k in sorted(node.iterkeys(), key=sort_func, reverse=True):
                r[k] = node[k]
                if isinstance(r[k], int):
                    if limit > 0 and limit > r[k]:
                        break
                    if count > 0 and len(r) > count:
                        break
                if isinstance(r[k], dict):
                    r[k] = self._sort_limit(r[k], limit, count)
            return r
        return node

    def has_error(self):
        return len(self._node) > 0

    @staticmethod
    def merge_dict(dst, src):
        for name, value in src.iteritems():
            if isinstance(value, int):
                dst[name] += value
            else:
                if name not in dst:
                    dst[name] = defaultdict(int)
                StatsKeeper.merge_dict(dst[name], value)

class Processor(object):
    def __init__(self, argparse_ns):
        self.__package_amr_loads = partial(penman.loads, model=xmrs.Dmrs)
        self.mrs = []
        self.gold = []
        self._gold_edm = []
        self.system = []
        self._system_edm = []
        self._edm_compare_result = []
        self._stats = []
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
            e = EdmContainer(self.mrs[i].get('sentence'), i)
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
        edm_summary = StatsKeeper()
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
                stats = self._stats[i]
                result["results"].append({"result-id": "EDM", "edm": self._edm_compare_result[i]})
                readings += 1
                edm_summary.merge(stats)
            result["readings"] = readings
            file_outpath = os.path.join(self.out_dir, "n%s.json" % i)
            with open(file_outpath, "w") as f:
                f.write(json.dumps(result, indent=indent))
        result = {
            "input" : "",
            "results" : [],
        }
        # python -m SimpleHTTPServer 8000
        edm_summary.trim(["gold_stats", "not in other", "predicates"])
        edm_summary.trim(["system_stats", "not in other", "predicates"])
        edm_summary.trim(["system_stats", "not in erg"], limit=0, count=15)
        edm_summary.trim(["system_stats", "predicate errors"], limit=0, count=15, debug=True)
        result["results"].append({"result-id": "Summary", "summary": edm_summary.to_dict()})
        file_outpath = os.path.join(self.out_dir, "n%s.json" % (len(self.mrs)))
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
        return system.validate_args(index, predicate, self.erg)

    def edm_system_gold_compare(self):
        assert len(self._gold_edm) == len(self._system_edm), "same length"
        for i in range(len(self._gold_edm)):
            (predicates, stats) = self.compare(i, self._gold_edm[i], self._system_edm[i])
            self._stats.append(stats)
            self._edm_compare_result.append({'predicates': predicates, 'stats': stats.to_dict()})

    def compare(self, i, gold, system):
        gold_set = set([k for k in gold._entries.iterkeys()])
        system_set = set([k for k in system._entries.iterkeys()])
        predicates = {}
        shared_stat_key = "shared"
        not_in_stat_key = "not in other"
        stats = StatsKeeper()
        for index in list(gold_set | system_set):
            stats.restart(["shared"])
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
                    stats.restart([prefix])
                    if predicate.endswith("unknown"):
                        stats['unknown'] += 1
                    else:
                        stats.restart([prefix, not_in_stat_key])
                        stats["total"] += 1
                        stats["surface" if predicate.startswith("_") else "abstract"] += 1
                        stats.restart([prefix, not_in_stat_key, "predicates"])
                        stats[predicate] += 1
                    if prefix == "system_stats":
                        stats.restart([prefix])
                        if predicate not in self.erg:
                            # only count non-unknown
                            if not predicate.endswith("unknown"):
                                stats.restart([prefix, "not in erg"])
                                stats["count"] += 1
                                stats[predicate] += 1
                        else:
                            pred_errors = self.validate_args(system, index, predicate)
                            if pred_errors.has_error() > 0:
                                stats['predicate with incorrect arg'] += 1
                                stats.push("predicate errors")
                                stats.merge_node(pred_errors)
            stats_update("system_stats", system_index_predicates - gold_index_predicates)
            stats_update("gold_stats", gold_index_predicates - system_index_predicates)
        return predicates, stats

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

#!/usr/bin/env python

import os
import sys
import json
import argparse
import penman as PM
import traceback
import shlex
import re
import networkx as nx

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
        self._rstr = False

    def has_rstr(self):
        return self._rstr

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
                        self._rstr = True
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
    def __init__(self, predicate):
        self.predicate = EdmPredicate.rtrim(predicate, "_rel")
        self._args = defaultdict(list)

    def args(self, src_arg, dst_index, dst_pred):
        self._args[src_arg].append({'src':src_arg, 'index':dst_index, 'predicate':dst_pred})

    def get_predicate_args(self):
        for key, list_value in self._args.iteritems():
            if key.startswith("ARG"):
                for v in list_value:
                    yield v

    @staticmethod
    def rtrim(s, str):
        if s.endswith(str):
            return s[0:(0-len(str))]
        return s

    def get_types(self, erg_dict):
        """Return all possible types for self, or ARG0"""
        ts = []
        if "_u_" in self.predicate:
            ts.append("u")
        elif self.predicate in erg_dict:
            erg_pred = erg_dict[self.predicate]
            ts += erg_pred.level_args(0)
        return set(ts)

    def erg_pred_types_at_level(self, arg_level, erg_dict):
        erg_level_types = set()
        erg_pred = None
        if self.predicate in erg_dict:
            erg_pred = erg_dict[self.predicate]
            erg_level_types.update(erg_pred.level_args(arg_level))
        return erg_pred, erg_level_types

class EdmPredicateContainer(object):
    def __init__(self, start, end, sentence, sentence_index):
        self.start  = int(start)
        self.end = int(end)
        self.len = self.end - self.start
        self._sentence_index = sentence_index
        self.span = sentence[self.start:self.end]
        self.predicates = {}
        self.carg = []

    def append(self, predicate):
        p = EdmPredicate(predicate)
        if p.predicate in self.predicates:
            return
        self.predicates[p.predicate] = p

    def carg(self, value):
        self.carg.append(value)

    def arg(self, src_pred, src_arg, dst_index, dst_pred):
        src_pred = EdmPredicate.rtrim(src_pred, "_rel")
        dst_pred = EdmPredicate.rtrim(dst_pred, "_rel")
        if src_pred not in self.predicates:
            assert 0
        self.predicates[src_pred].args(src_arg, dst_index, dst_pred)

    def __str__(self):
        return "<%s>:<%s>" % (self.predicates, self.args)

    def parse_arg_level(self, arg_name):
        arg_names = arg_name.split("/")
        if arg_names[0] == "ARG":
            return arg_names[0], 0
        return arg_names[0], int(arg_names[0][3:])

    def is_same_span(self, other):
        return self.start == other.start and self.end == other.end

    def candidate_args_for_validate(self, predicate):
        """find candidate arg for validateion:
                o ARGX - none self referenced - external reference args
                o RSTR - self referenced -
        """
        po = self.predicates[predicate]
        # for all the arguments
        # find all the args starting with ARGX
        # and return the name, level, and arg_predicate
        for arg_name, arg_list in po._args.iteritems():
            for arg_dict in arg_list:
                arg_predicate = arg_dict['predicate']
                if arg_name.startswith("ARG") and arg_name != "ARG":
                    arg_name, arg_level = self.parse_arg_level(arg_name)
                    yield po, arg_name, arg_level, arg_predicate
                elif arg_name.startswith("RSTR"):
                    yield po, arg_name, 0, po

    def validate_args(self, predicate_str, erg_dict):
        """Given the erg dictionary, validate the predicate designated by predicate_strself.
                We have the predicate_str, and erg_dict, but not the predicate arguments.
                Since we can have multiple predicates for the span, it's not possible to use
                the EDM information to figure out the predicate and ARG relationship. So
                we do a bucket instead.
        """
        errors = StatsKeeper()
        for po, arg_name, arg_level, arg_predicate in self.candidate_args_for_validate(predicate_str):
            # get the possible types at this level
            # this includes RSTR
            erg_pred, erg_level_types = po.erg_pred_types_at_level(arg_level, erg_dict)
            if arg_name.startswith("RSTR"):
                if erg_pred is not None and not erg_pred.has_rstr():
                    errors.restart(["extra"])
                    errors["count"] += 1
                    errors.restart(["extra", "RSTR"])
                    errors[predicate_str] += 1
                    errors["count"] += 1
            elif len(erg_level_types) > 0:
                # only do arg comparsion if erg have args
                # otherwise it's extra arg
                arg_predicate_types = arg_predicate.get_types(erg_dict)
                # XXX assume "x" is wildcard
                predicate_types_union = set(erg_level_types) & set(arg_predicate_types)
                if len(predicate_types_union) == 0 and not "x" in arg_predicate_types:
                    assert predicate_str in self.predicates, "is own predicate"
                    errors.restart(["incorrect"])
                    errors["count"] += 1
                    errors.restart(["incorrect", arg_name])
                    errors["%s (%s)" % (predicate_str, arg_predicate.predicate)] += 1
                    errors["count"] += 1
            else:
                assert predicate_str in erg_dict, "in erg"
                # extra arg - accounting
                errors.restart(["extra"])
                errors["count"] += 1
                errors.restart(["extra", arg_name])
                errors[predicate_str] += 1
                errors["count"] += 1
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
        self._stats = StatsKeeper()
        self._align = {}
        self._edges = []

    def predicate_dict_set(self):
        d = defaultdict(set)
        for index, predicate in self._entries.iteritems():
            ps = [p for p in predicate.predicates.iterkeys()]
            if len(ps) > 0:
                d[index].update(ps)
        return d

    def graph_verify(self):
        G = nx.Graph()
        G.add_edges_from(self._edges)
        print nx.is_connected(G)
        components = list(nx.connected_components(G))
        print len(components)

    def parse(self, line):
        for item in line.split(";"):
            if len(item.strip()) == 0:
                continue
            tokens = item.strip().split(" ")
            assert len(tokens) == 3 or len(tokens) == 5, "known numbers"
            index = tokens[0]
            if len(tokens) == 3:
                (typename, typevalue) = (tokens[1], tokens[2])
                if index not in self._entries:
                    (start, end) = index.split(":")
                    self._entries[index] = EdmPredicateContainer(start, end, self._sentence, self._sentence_index)
                if typename == "NAME":
                    self._entries[index].append(typevalue)
                elif typename == "CARG":
                    self._entries[index].carg.append(typevalue)
                    # record edges for graph
                if ":" in typevalue:
                    self._edges.append((index, typevalue))
                else:
                    self._edges.append((index, index))
            else:
                # 156:158 poss ARG1/EQ 163:173 _jetliners/nns_u_unknown_rel
                (src_pred, src_arg, dst_index, dst_pred) = (tokens[1], tokens[2], tokens[3], tokens[4])
                src_pred = EdmPredicate.rtrim(src_pred, "_rel")
                dst_pred = EdmPredicate.rtrim(dst_pred, "_rel")

                if src_pred not in self._entries[index].predicates and src_pred not in self._entries[dst_index].predicates:
                    assert 0
                if src_pred in self._entries[index].predicates:
                    self._entries[index].arg(src_pred, src_arg, dst_index, dst_pred)
                else:
                    self._entries[dst_index].arg(src_pred, src_arg, index, dst_pred)
                self._edges.append((index, dst_index))

    def align_with(self, other):
        """Using self as the model, make the other align with self"""
        for index, self_pc in self._entries.iteritems():
            if index not in other._entries and self_pc.span.endswith((".")):
                # candidate for alignment fix
                other_key = "%d:%d" % (self_pc.start, self_pc.end - 1)
                if other_key in other._entries:
                    other_pc = other._entries[other_key]
                    del other._entries[other_key]
                    other._align[other_key] = index
                    if index not in other._entries:
                        other._entries[index] = other_pc
                        other_pc.start = self_pc.start
                        other_pc.end = self_pc.end
                        other_pc.span = self_pc.span
                        other_pc.len = self_pc.len
                    else:
                        other._entries[index].merge(other_pc)

    def edm_link_args(self):
        for index, predicate_container in self._entries.iteritems():
            for predicate in predicate_container.predicates.itervalues():
                for arg in predicate.get_predicate_args():
                    assert arg['index'] in self._entries
                    assert arg['predicate'] in self._entries[arg['index']].predicates
                    arg['container'] = self._entries[arg['index']]
                    arg['predicate'] = self._entries[arg['index']].predicates[arg['predicate']]

    def validate_args(self, index, predicate, erg):
        pc = self._entries[index]
        assert predicate in pc.predicates
        return pc.validate_args(predicate, erg)

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

class Entry(object):
    def __init__(self, index, mrs):
        self.index = index
        self.mrs = mrs
        self.sentence = mrs['sentence']
        self._gold_edm = None
        self._system_edm = None
        self._edm_compare_result = None
        self._system_amr = None
        self._gold_amr = None
        self._stats = StatsKeeper()

    def edm_link_args(self):
        self._system_edm.edm_link_args()

    def edm_align(self):
        assert self._gold_edm is not None and self._system_edm is not None, "edm parsed"
        self._gold_edm.align_with(self._system_edm)

    def dmrs_predicate_dict_set(self, dmrs, align):
        d = {}
        for i, t, val in dmrs.to_triples():
            if i not in d:
                d[i] = {'predicate': [], 'link': None}
            if t == "predicate":
                d[i]['predicate'].append(val)
            elif t == "lnk":
                assert d[i]['link'] is None
                index = val[2:-2]
                if index in align:
                    index = align[index]
                d[i]['link'] = index
        dd = defaultdict(set)
        for i, v in d.iteritems():
            if v['link'] is not None and len(v['predicate']) > 0:
                dd[v['link']].update(v['predicate'])
        return dd

    def compare_predicates(self):
        if "dmrs" in self._system_amr:
            amr = self.dmrs_predicate_dict_set(self._system_amr['dmrs'], self._system_edm._align)
            edm = self._system_edm.predicate_dict_set()
            amr_keys = set([k for k in amr.iterkeys()])
            edm_keys = set([k for k in edm.iterkeys()])
            for k in (amr_keys - edm_keys):
                self._stats.restart(["system_stats", "format disagreement"])
                self._stats["in arm not edm"] += len(amr[k])
            for k in (edm_keys - amr_keys):
                self._stats.restart(["system_stats", "format disagreement"])
                self._stats["in edm not amr"] += len(edm[k])
        else:
            self._stats.restart(["system_stats", "format disagreement"])
            self._stats["no dmrs"] += 1

    def edm_graph_verify(self):
        G = nx.Graph()
        # self._system_edm.graph_verify()
        G.add_edges_from(self._system_edm._edges)
        if not nx.is_connected(G):
            self._stats.restart(["system_stats", "format disagreement"])
            self._stats["not connected"] += 1


    def edm_analyze(self, erg):
        assert self._gold_edm is not None and self._system_edm is not None, "edm parsed"
        predicates = self.compare(self._gold_edm, self._system_edm, erg)
        self._edm_compare_result = {'predicates': predicates, 'stats': self._stats.to_dict()}
        self.compare_predicates()
        self.edm_graph_verify()


    def compare(self, gold, system, erg):
        gold_set = set([k for k in gold._entries.iterkeys()])
        system_set = set([k for k in system._entries.iterkeys()])
        predicates = {}
        shared_stat_key = "shared"
        not_in_stat_key = "not in other"
        for index in list(gold_set | system_set):
            self._stats.restart(["shared"])
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
            self._stats['total'] += len(system_index_predicates | gold_index_predicates)
            self._stats['gold'] += len(gold_index_predicates - system_index_predicates)
            self._stats['system'] += len(system_index_predicates - gold_index_predicates)
            self._stats['common'] += len(system_index_predicates & gold_index_predicates)
            if len(gold_index_predicates) > 0:
                predicates[index]["predicate"]["gold"] = list(gold_index_predicates)
            if len(system_index_predicates) > 0:
                predicates[index]["predicate"]["system"] = list(system_index_predicates)
            def stats_update(prefix, predicates):
                for predicate in predicates:
                    self._stats.restart([prefix])
                    if predicate.endswith("unknown"):
                        self._stats['unknown'] += 1
                    else:
                        self._stats.restart([prefix, not_in_stat_key])
                        self._stats["total"] += 1
                        self._stats["surface" if predicate.startswith("_") else "abstract"] += 1
                        self._stats.restart([prefix, not_in_stat_key, "predicates"])
                        self._stats[predicate] += 1
                    if prefix == "system_stats":
                        # for system predicates not in gold, we want to validate
                        # the argument
                        self._stats.restart([prefix])
                        if predicate not in erg:
                            # only count non-unknown
                            if not predicate.endswith("unknown"):
                                self._stats.restart([prefix, "not in erg"])
                                self._stats["count"] += 1
                                self._stats[predicate] += 1
                        else:
                            pred_errors = self.validate_args(system, index, predicate, erg)
                            if pred_errors.has_error() > 0:
                                self._stats['predicate with incorrect arg'] += 1
                                self._stats.push("predicate errors")
                                self._stats.merge_node(pred_errors)
            stats_update("system_stats", system_index_predicates - gold_index_predicates)
            stats_update("gold_stats", gold_index_predicates - system_index_predicates)
        return predicates

    def to_json(self):
        readings = 0
        result = {
            "input" : self.sentence,
            "results" : [],
        }
        if 'mrs' in self.mrs:
            result["results"].append(
                {"result-id": "ACE MRS",
                 "mrs": xmrs.Mrs.to_dict(self.mrs['mrs'], properties=True)})
            readings += 1
        if 'dmrs' in self._gold_amr:
            result["results"].append(
                {"result-id": "Gold DMRS (convert from penman)",
                 "dmrs": xmrs.Dmrs.to_dict(self._gold_amr['dmrs'], properties=True)})
            readings += 1
        if 'dmrs' in self._system_amr:
            result["results"].append(
                {"result-id": "System DMRS (convert from penman)",
                 "dmrs": xmrs.Dmrs.to_dict(self._system_amr['dmrs'], properties=True)})
            readings += 1
        if self._gold_edm is not None and self._system_edm is not None:
            result["results"].append({"result-id": "EDM", "edm": self._edm_compare_result})
            readings += 1
        result["readings"] = readings
        return (result, self._stats)

    def validate_args(self, system, index, predicate, erg):
        # print predicate
        assert erg is not None, "erg defined"
        assert predicate in erg, "predicate in erg"
        return system.validate_args(index, predicate, erg)


class Processor(object):
    def __init__(self, argparse_ns):
        self.__package_amr_loads = partial(penman.loads, model=xmrs.Dmrs)
        self.entries = []
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

    def parse_edm_file(self, input, attr):
        assert len(self.entries) > 0, "MRS parsed already"
        ret = []
        i = 0
        for line in input:
            e = EdmContainer(self.entries[i].sentence, i)
            e.parse(line)
            setattr(self.entries[i], attr, e)
            i += 1
            if self._limit > 0 and i >= self._limit:
                break
        return ret

    def edm_post_process(self):
        self.entries_edm_link_args()
        self.entries_edm_align()
        self.entries_edm_analyze()

    def entries_edm_link_args(self):
        for i in range(len(self.entries)):
            self.entries[i].edm_link_args()

    def entries_edm_align(self):
        for i in range(len(self.entries)):
            self.entries[i].edm_align()

    def entries_edm_analyze(self):
        for i in range(len(self.entries)):
            self.entries[i].edm_analyze(self.erg)

    def load_edm(self):
        self.load_erg()
        if len(self.entries) == 0:
            self.parse_mrs(self._files["ace"])
        self.parse_edm_gold(self._files["gold_edm"])
        self.parse_edm_system(self._files["system_edm"])
        self.edm_post_process()

    def load_json(self):
        self.parse_mrs(self._files["ace"])
        self.parse_amr_system(self._files["system"])
        self.parse_amr_gold(self._files["gold"])
        self.load_edm()

    def load_erg(self):
        if self.erg is None:
            self.erg = Erg(self._files["erg"])
            self.erg.parse(self._files["abstract"])

    def to_json(self, indent=2):
        edm_summary = StatsKeeper()
        for i in range(len(self.entries)):
            entry = self.entries[i]
            (result, stats) = entry.to_json()
            edm_summary.merge(stats)
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
        file_outpath = os.path.join(self.out_dir, "n%s.json" % (len(self.entries)))
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
            self.entries.append(Entry(len(self.entries), self.convert_mrs(mrs)))
            if self._limit > 0 and len(self.entries) >= self._limit:
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

    def parse_amr_file(self, input, attr):
        i = 0
        for lines in self._parse_amr_file_to_lines(input):
            amr = self.convert_amr(lines)
            if amr is not None:
                setattr(self.entries[i], attr, amr)
                i += 1
                if self._limit > 0 and i >= self._limit:
                    break
        assert i == len(self.entries)

    def parse_edm_gold(self, input):
        self.parse_edm_file(input, "_gold_edm")

    def parse_edm_system(self, input):
        self.parse_edm_file(input, "_system_edm")

    def parse_amr_gold(self, input):
        self.parse_amr_file(input, "_gold_amr")

    def parse_amr_system(self, input):
        self.parse_amr_file(input, "_system_amr")

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
    print "total: %d " % (len(p.entries))

def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(prog=sys.argv[0])
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--ace', type=argparse.FileType('r'),
            default="../../error-analysis-data-2/dev.erg.mrs")
    subparsers = parser.add_subparsers()

    parser_json = subparsers.add_parser("json")
    parser_json.add_argument('--system', type=argparse.FileType('r'),
            default="../../error-analysis-data-2/dev.system.dmrs.amr.lnk")
    parser_json.add_argument('--gold', type=argparse.FileType('r'),
            default="../../error-analysis-data-2/dev.gold.dmrs.amr.lnk")
    parser_json.add_argument('--gold_edm', type=argparse.FileType('r'),
            default="../../error-analysis-data-2/dev.gold.edm.lab")
    parser_json.add_argument('--system_edm', type=argparse.FileType('r'),
            default="../../error-analysis-data-2/dev.system.dmrs.edm.lab")
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
            default="../../error-analysis-data-2/dev.gold.edm.lab")
    parser_edm.add_argument('--system_edm', type=argparse.FileType('r'),
            default="../../error-analysis-data-2/dev.system.dmrs.edm.lab")
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

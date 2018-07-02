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

class ErgArg(object):
    # u := top.
    # i := u
    # p := u
    # h := p
    # e := i
    # x := i & p.
    #       u
    #      / \
    #     i   p
    #    / \ / \
    #   e   x   h
    hieracy = {
        "u": ["u", "i", "p", "e", "x", "h"],
        "i": ["i", "x", "e", "u"],
        "p": ["p", "x", "h", "u"],
        "h": ["h", "p", "u"],
        "e": ["e", "i", "u"],
        "x": ["x", "i", "p", "u"],
    }
    def __init__(self, token, arg, optional=False):
        self._token = token
        self._args = {}
        self.arg = arg
        self.end = False
        self.optional = optional
        self.rstr = False
        self.level = 0

    def copy(self):
        e = ErgArg(self._token, self.arg, self.optional)
        e.level = self.level
        return e

    def sync(self, other):
        assert self.level == other.level, "same level"

    def add(self, arg):
        arg.level = self.level + 1
        s = "ARG%d" % (self.level)
        assert s == self._token
        if arg.optional:
            # if arg being added is optional
            # mark self as a possible end
            self.end = True
        if arg.arg in self._args:
            _arg = self._args[arg.arg]
            _arg.sync(arg)
            return _arg
        self._args[arg.arg] = arg
        return arg

    def get_all_args(self, args=[]):
        d = self._all if hasattr(self, "_all") else self._args
        for k, v in d.iteritems():
            _args = args[:]
            _args.append("ARG%d %s%s" % (v.level, k,
                "*" if v.optional else ""))
            if v.end:
                yield _args
            for arg in v.get_all_args(_args):
                yield arg

    def get_args(self, args=[]):
        d = self._args
        for k, v in d.iteritems():
            _args = args[:]
            _args.append("ARG%d %s%s" % (v.level, k,
                "*" if v.optional else ""))
            if v.end:
                yield _args
            for arg in v.get_all_args(_args):
                yield arg

    def print_args(self):
        print "%s" % self._token
        for arg in self.get_args([]):
            print arg

class ErgPredicate(ErgArg):
    def __init__(self):
        self._args = {}             # including ARG0
        self._all = {}
        self._rstr = False
        self._predicate = None

    def has_rstr(self):
        return self._rstr

    def parse_args(self, predicate, args):
        self._predicate = predicate
        arg_tokens = iter(re.split("\s|\,|\.", args))
        def get_args(tokens):
            arg = None
            try:
                optional = False
                while True:
                    token = next(tokens)
                    if token.startswith("ARG"):
                        if token == "ARG":
                            token = "ARG0"
                        if arg is not None:
                            yield arg
                        arg = ErgArg(token, tokens.next(), optional=optional)
                    elif token == "RSTR":
                        arg._rstr = True
                        self._rstr = True
                    elif token == "[":
                        optional = True
                    elif token == "]":
                        optional = False
            except StopIteration:
                if arg is not None:
                    yield arg
        objs = [obj for obj in get_args(arg_tokens)]
        assert len(objs) >= 1, "at least one"
        obj = objs[0]
        if obj.arg in self._all:
            self._all[obj.arg].sync(obj)
        else:
            self._all[obj.arg] = obj
        obj = self._all[obj.arg]
        arg = None
        for i in range(1, len(objs)):
            _obj = objs[i]
            obj.add(_obj)
            obj = _obj
            _arg = _obj.copy()
            if i == 1:
                arg = _arg
                if arg.arg in self._args:
                    self._args[arg.arg].sync(arg)
                else:
                    self._args[arg.arg] = arg
                arg = self._args[arg.arg]
            else:
                arg.add(_arg)
                arg = _arg
        obj.end = True

    def print_all(self):
        print "%s" % self._predicate
        for arg in self.get_all_args([]):
            print arg

    def print_args(self):
        print "%s" % self._predicate
        for arg in self.get_args([]):
            print arg

    def get_predicate_arg0(self):
        return set([a for a in self._all.iterkeys()])

class Erg(object):
    def __init__(self, input):
        self._ergs = defaultdict(ErgPredicate)
        self.parse(input)

    def process_line(self, line):
        (predicate, args) = line.strip().split(":")
        predicate = predicate.strip()
        args = args.strip()
        self._ergs[predicate].parse_args(predicate, args)

    def parse(self, input):
        head = input.next()
        input.next()
        for line in input:
            self.process_line(line)

    def __contains__(self, predicate):
        return predicate in self._ergs

    def __getitem__(self, predicate):
        return self._ergs[predicate]

    def print_predicate(self, predicate):
        pred_erg = self.get_predicate(predicate)
        if pred_erg is not None:
            pred_erg.print_all()

    def get_predicate(self, predicate):
        if predicate in self:
            return self._ergs[predicate]
        return None

    def print_erg(self):
        for item in sorted(self._ergs.itervalues(), key=lambda i: i._predicate):
            item.print_all()

    def get_predicate_arg0(self, predicate):
        p = self.get_predicate(predicate)
        t = set([])
        if p is not None:
            t.update(p.get_predicate_arg0())
        return t

class Predicate(object):
    def __init__(self, predicate, node=None):
        self.predicate = Predicate.rtrim(predicate, "_rel")
        self._args = defaultdict(list)
        self._node = node

    def args(self, src_arg, dst_index, dst_pred):
        self._args[src_arg].append({"src":src_arg, "index":dst_index, "predicate":dst_pred})

    def get_predicate_args(self):
        for key, list_value in self._args.iteritems():
            if key.startswith("ARG"):
                for v in list_value:
                    yield v

    def print_all(self):
        print self.predicate
        for k, v in self._args.iteritems():
            print k, v

    @staticmethod
    def rtrim(s, str):
        if s.endswith(str):
            return s[0:(0-len(str))]
        return s

class PredicateContainer(object):
    def __init__(self, start, end, sentence, sentence_index):
        self.start  = int(start)
        self.end = int(end)
        self.len = self.end - self.start
        self._sentence_index = sentence_index
        self.span = sentence[self.start:self.end]
        self.predicates = {}
        self.carg = []

    def append(self, predicate):
        p = Predicate(predicate)
        if p.predicate in self.predicates:
            return
        self.predicates[p.predicate] = p

    def carg(self, value):
        self.carg.append(value)

    def arg(self, src_pred, src_arg, dst_index, dst_pred):
        src_pred = Predicate.rtrim(src_pred, "_rel")
        dst_pred = Predicate.rtrim(dst_pred, "_rel")
        if src_pred not in self.predicates:
            assert 0
        self.predicates[src_pred].args(src_arg, dst_index, dst_pred)

    def __str__(self):
        return "<%s>:<%s>" % (self.predicates, self.args)

    def parse_arg_level(self, arg_name):
        arg_names = arg_name.split("/")
        if arg_names[0] == "ARG":
            return "ARG1", 1
        return arg_names[0], int(arg_names[0][3:])

    def is_same_span(self, other):
        return self.start == other.start and self.end == other.end

    def args_for_validate(self, predicate, erg):
        """find candidate arg for validateion:
                o ARGX - none self referenced - external reference args
                o RSTR - self referenced -
        """
        po = self.predicates[predicate]
        # for all the arguments
        # find all the args starting with ARGX
        # and return the name, level, and arg_predicate
        args = []
        for arg_name, arg_list in po._args.iteritems():
            for arg_dict in arg_list:
                arg_predicate = arg_dict["predicate"]
                if arg_name.startswith("ARG") and arg_name != "ARG":
                    arg_name, arg_level = self.parse_arg_level(arg_name)
                    p = arg_predicate.predicate
                    if p.endswith("unknown"):
                        p = "unknown"
                    arg_types = erg.get_predicate_arg0(p)
                    if len(arg_types) == 0:
                        arg_types = ["u"]
                    arg = (arg_name, arg_level, p, arg_types)
                    if arg not in args:
                        args.append(arg)
                elif arg_name.startswith("RSTR"):
                    arg = ("RSTR", 0, po.predicate)
                    if arg not in args:
                        args.append(arg)
        return sorted(args, key=lambda arg: (arg[1], arg[0] == "RSTR"))

def validate_predicate_erg_args(predicate_str, args, erg, source=None, debug=True):
    """Given the erg dictionary, validate the predicate designated by predicate_strself.
            We have the predicate_str, and erg_dict, but not the predicate arguments.
            Since we can have multiple predicates for the span, it"s not possible to use
            the EDM information to figure out the predicate and ARG relationship. So
            we do a bucket instead.
    """
    errors = StatsKeeper()
    pred_erg = erg.get_predicate(predicate_str)
    if pred_erg is None:
        if predicate_str.endswith("unknown"):
            pred_erg = erg.get_predicate("unknown")
        else:
            errors.restart(["system_stats", "not in erg validate"])
            errors["count"] += 1
            errors[predicate_str] += 1
            return errors

    def get_pred_erg_args_from_prev(prev_args, type_set, level):
        # using previous matched args and types, retrieve next set of args
        _args = []
        _arg_types = []
        for arg in prev_args:
            assert arg.level == level, "right level"
            for k, v in arg._args.iteritems():
                arg_types = [k]
                arg_types += ErgArg.hieracy.get(k, [])
                # print "matching for ARG%s" % (level + 1), type_set, arg_types, predicate_str
                types_u = set(arg_types) & set(type_set)
                # fixme - labels "h" using dmrs.hcon
                if len(types_u) > 0 or k == "h":
                    # print "matched", k
                    _args.append(v)
        return _args

    arg_args = filter(lambda a: a[0].startswith("ARG"), args)
    arg_rstr = filter(lambda a: a[0] == "RSTR", args)
    last = 0
    if len(arg_args) == 0:
        # no args, find non-arg predicate
        has_end = False
        for erg_arg in pred_erg._all.itervalues():
            if erg_arg.end:
                has_end = True
        if not has_end:
            errors.restart(["missing"])
            errors["count"] += 1
            errors.restart(["missing", "ARG%s" % (last + 1)])
            errors[predicate_str] += 1
            errors["count"] += 1
    else:
        erg_args = []
        for erg_arg in pred_erg._all.itervalues():
            if not erg_arg.end:
                erg_args.append(erg_arg)
        if len(arg_rstr) > 0:
            if not pred_erg.has_rstr():
                errors.restart(["extra"])
                errors["count"] += 1
                errors.restart(["extra", "RSTR"])
                errors[predicate_str] += 1
                errors["count"] += 1
        erg_args = [a for a in pred_erg._all.itervalues()]
        prev_args = []
        _erg_args = []
        for arg in arg_args:
            index = last + 1
            if arg[1] != index:
                # not expected index
                if arg[0] in prev_args:
                    errors.restart(["dupping"])
                    errors["count"] += 1
                    errors.restart(["dupping", "%s" % (arg[0])])
                    errors[predicate_str] += 1
                    errors["count"] += 1
                elif arg[1] > index:
                    # we skipped
                    errors.restart(["skipping"])
                    errors["count"] += 1
                    errors.restart(["skipping", "ARG%d" % (index)])
                    errors[predicate_str] += 1
                    errors["count"] += 1
                else:
                    print source, arg[1], index, arg_args
                    assert 0
            else:
                if len(erg_args) == 0:
                    # there shouldn't be any args here
                    errors.restart(["extra"])
                    errors["count"] += 1
                    errors.restart(["extra", arg[0]])
                    errors[arg[2]] += 1
                    errors["count"] += 1
                    break
                # get the types are this level base on previous matched args
                _erg_args = get_pred_erg_args_from_prev(erg_args, arg[3], last)
                if len(_erg_args) == 0:
                    # could find matching arg
                    # incorrect
                    errors.restart(["incorrect"])
                    errors["count"] += 1
                    errors.restart(["incorrect", arg[0]])
                    errors[predicate_str] += 1
                    errors["count"] += 1
                    if debug:
                        print predicate_str, "not match", arg
                    break
                prev_args.append(arg[0])
                erg_args = _erg_args
                last = arg[1]
        if len(_erg_args) != 0:
            has_end = False
            for arg in erg_args:
                if arg.end:
                    has_end = True
            if not has_end:
                errors.restart(["missing"])
                errors["count"] += 1
                errors.restart(["missing", "ARG%s" % (last + 1)])
                errors[predicate_str] += 1
                errors["count"] += 1
    return errors

class AMR(object):
    def __init__(self, index, _dmrs, erg, debug=False):
        self.debug = debug
        self.index = index
        self._nodes = {}
        self._connected = True
        self._well_formed = True
        if _dmrs is not None:
            d = _dmrs.to_dict()
            for node in d.get("nodes", []):
                nodeid = node.get("nodeid")
                assert nodeid not in self._nodes
                self._nodes[nodeid] = Predicate(node.get('predicate'), node=node)
            for link in d.get("links", []):
                arg = link.get('rargname')
                if arg is not None and (arg.startswith("ARG") or arg == "RSTR"):
                    pred = self._nodes[link["from"]]
                    dest = self._nodes[link["to"]]
                    assert pred is not None and dest is not None, "pred dest exists"
                    pred._args[arg].append({"src": arg,
                                            "predicate": dest.predicate,
                                            "types": erg.get_predicate_arg0(dest.predicate)})
            self._connected = _dmrs.is_connected()
            if self._connected:
                try:
                    self.is_well_formed()
                except:
                    self._well_formed = False

    def predicate_index(self):
        """Generate predicate:from:to index"""
        for pred in self._nodes.itervalues():
            yield "%s:%s:%s" % (pred.predicate,
                                pred._node["lnk"]["from"],
                                pred._node["lnk"]["to"])
    def indexes(self):
        """Generate predicate:from:to index"""
        for pred in self._nodes.itervalues():
            yield "%s:%s" % (pred._node["lnk"]["from"],
                             pred._node["lnk"]["to"])

    def predicates_at(self, start, end):
        return [pred.predicate for pred in self._nodes.itervalues() if
            pred._node["lnk"]["from"] == start and pred._node["lnk"]["to"] == end]

    def align(self, align):
        for pred in self._nodes.itervalues():
            index = "%s:%s" % (pred._node["lnk"]["from"], pred._node["lnk"]["to"])
            if index in align:
                (_from, _to) = align[index].split(":")
                pred._node["lnk"]["from"] = int(_from)
                pred._node["lnk"]["to"] = int(_to)

    def validate_predicate_args(self, erg):
        stats = StatsKeeper()
        for pred in self._nodes.itervalues():
            args = []
            for arg_list in pred._args.itervalues():
                for arg in arg_list:
                    arg_level = 0
                    arg_name = arg["src"]
                    if arg_name.startswith("ARG") and arg_name != "ARG":
                        if "-" in arg_name:
                            arg_name = arg_name.split("-")[0]
                        arg_level = int(arg_name[3:])
                    p = arg["predicate"]
                    if arg_name == "ARG":
                        arg_level = 1
                        arg_name = "ARG1"
                    arg_types = arg["types"]
                    if p.endswith("unknown"):
                        p = "unknown"
                        arg_types = erg.get_predicate_arg0(p)
                    if len(arg_types) == 0:
                        arg_types = ["u"]
                    args.append((arg_name, arg_level, p, arg_types))
                args = sorted(args, key=lambda arg: (arg[1], arg[0] == "RSTR"))
            pp = pred.predicate
            if pp.endswith("unknown"):
                pp = "unknown"
            _stats = validate_predicate_erg_args(pp, args, erg,
                                                 source="amr", debug=self.debug)
            if _stats.has_error():
                print "===", self.index, "amr"
                erg.print_predicate(pp)
                print args
                print _stats.to_dict()

                stats.restart(["system_stats"])
                stats["predicate with incorrect arg"] += 1
                stats.restart([])
                stats.merge_node(_stats)
        return stats


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
                    self._entries[index] = PredicateContainer(start, end, self._sentence, self._sentence_index)
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
                src_pred = Predicate.rtrim(src_pred, "_rel")
                dst_pred = Predicate.rtrim(dst_pred, "_rel")

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
                    assert arg["index"] in self._entries
                    assert arg["predicate"] in self._entries[arg["index"]].predicates
                    arg["predicate"] = self._entries[arg["index"]].predicates[arg["predicate"]]

    def validate_args(self, index, predicate, erg):
        pc = self._entries[index]
        assert predicate in pc.predicates
        args = pc.args_for_validate(predicate, erg)
        return validate_predicate_erg_args(predicate, args, erg, source="edm")

class StatsKeeper(object):
    def __init__(self, title=""):
        self._stats = {}
        self._node = self._stats
        self.title = title

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
        self._node = self._stats
        for key in keys[0:-1]:
            if key in self._node:
                self.push(key)
            else:
                return
        # replacement
        last = keys[-1]
        if last in self._node:
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

    def has(self, key):
        return key in self._node

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
    def __init__(self, index, mrs, debug=False):
        self.index = index
        self.mrs = mrs
        self.sentence = mrs["sentence"]
        self._gold_edm = None
        self._system_edm = None
        self._edm_compare_result = None
        self._system_amr = None
        self._gold_amr = None
        self._stats = StatsKeeper()
        self._gold_amr_stats = StatsKeeper()
        self._system_amr_stats = StatsKeeper()
        self.debug = debug

    def edm_link_args(self):
        self._system_edm.edm_link_args()

    def edm_align(self):
        assert self._gold_edm is not None and self._system_edm is not None, "edm parsed"
        self._gold_edm.align_with(self._system_edm)

    def dmrs_predicate_dict_set(self, dmrs, align):
        d = {}
        for i, t, val in dmrs.to_triples():
            if i not in d:
                d[i] = {"predicate": [], "link": None}
            if t == "predicate":
                d[i]["predicate"].append(val)
            elif t == "lnk":
                assert d[i]["link"] is None
                index = val[2:-2]
                if index in align:
                    index = align[index]
                d[i]["link"] = index
        dd = defaultdict(set)
        for i, v in d.iteritems():
            if v["link"] is not None and len(v["predicate"]) > 0:
                dd[v["link"]].update(v["predicate"])
        return dd

    def compare_predicates(self):
        if "dmrs" in self._system_amr:
            amr = self.dmrs_predicate_dict_set(self._system_amr["dmrs"], self._system_edm._align)
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
            self._stats.restart(["system_stats"])
            self._stats["not connected"] += 1

    def has_predicate_error(self):
        self._stats.restart(["system_stats"])
        return self._stats.has("predicate errors")

    def has_predicate_error_extra_arg(self):
        self._stats.restart(["system_stats"])
        if self._stats.has("predicate errors"):
            return "extra" in self._stats["predicate errors"]
        return False

    def has_predicate_error_incorrect_arg(self):
        self._stats.restart(["system_stats"])
        if self._stats.has("predicate errors"):
            return "incorrect" in self._stats["predicate errors"]
        return False

    def is_matched(self):
        self._stats.restart(["shared"])
        return self._stats["total"] == self._stats["common"]

    def amr_analyze(self, erg):
        assert self._gold_amr is not None and self._system_amr is not None, "amr parsed"
        self._gold_amr["amr_obj"] = AMR(self.index, self._gold_amr.get("dmrs"), erg, debug=self.debug)
        self._system_amr["amr_obj"] = AMR(self.index, self._system_amr.get("dmrs"), erg, debug=self.debug)
        self._system_amr["amr_obj"].align(self._system_edm._align)
        self.amr_compare(self._gold_amr["amr_obj"], self._system_amr["amr_obj"], erg, is_gold=True)
        self.amr_compare(self._gold_amr["amr_obj"], self._system_amr["amr_obj"], erg, is_gold=False)

    def edm_analyze(self, erg):
        assert self._gold_edm is not None and self._system_edm is not None, "edm parsed"
        predicates = self.compare(self._gold_edm, self._system_edm, erg)
        self._edm_compare_result = {"predicates": predicates, "stats": self._stats.to_dict()}
        self.compare_predicates()
        self.edm_graph_verify()
        self._stats.restart(["summary"])
        self._stats["count"] = 1
        if self.is_matched():
            self._stats.restart(["summary"])
            self._stats["matched"] = 1
        if self.has_predicate_error():
            self._stats.restart(["summary"])
            self._stats["has predicate error"] = 1
        if self.has_predicate_error_incorrect_arg():
            self._stats.restart(["summary"])
            self._stats["has predicate incorrect arg"] = 1
        if self.has_predicate_error_extra_arg():
            self._stats.restart(["summary"])
            self._stats["has predicate error extra arg"] = 1
            if self.has_predicate_error_incorrect_arg():
                self._stats.restart(["summary"])
                self._stats["has predicate error extra arg and incorrect arg"] = 1

    def amr_compare(self, gold, system, erg, is_gold=False):
        stats_main = None
        pred_errors = None
        if is_gold:
            stats_main = self._gold_amr_stats
            pred_errors = gold.validate_predicate_args(erg)
            prefix = "gold_stats"
            stats_main.restart([prefix])
            if pred_errors.has_error():
                stats_main["predicate with incorrect arg"] += 1
                stats_main.push("predicate errors")
                stats_main.merge_node(pred_errors)
            if not gold._connected:
                stats_main["not connected"] += 1
            if not gold._well_formed:
                stats_main["not well formed"] += 1
        else:
            stats_main = self._system_amr_stats
            pred_errors = system.validate_predicate_args(erg)
            prefix = "system_stats"
            stats_main.restart([prefix])
            if pred_errors.has_error():
                stats_main["predicate with incorrect arg"] += 1
                stats_main.push("predicate errors")
                stats_main.merge_node(pred_errors)
            if not system._connected:
                stats_main["not connected"] += 1
            if not system._well_formed:
                stats_main["not well formed"] += 1

        gold_set = set([p for p in gold.indexes()])
        system_set = set([p for p in system.indexes()])
        for index in list(gold_set | system_set):
            stats_main.restart(["shared"])
            (start, end) = index.split(":")
            start = int(start)
            end = int(end)
            gold_index_predicates = set(gold.predicates_at(start, end))
            system_index_predicates = set(system.predicates_at(start, end))
            stats_main["total"] += len(system_index_predicates | gold_index_predicates)
            stats_main["gold"] += len(gold_index_predicates - system_index_predicates)
            stats_main["system"] += len(system_index_predicates - gold_index_predicates)
            stats_main["common"] += len(system_index_predicates & gold_index_predicates)
            def stats_update(prefix, predicates):
                for predicate in predicates:
                    stats_main.restart([prefix])
                    if predicate.endswith("unknown"):
                        stats_main["unknown"] += 1
                    else:
                        stats_main.restart([prefix, "not in other"])
                        stats_main["total"] += 1
                        stats_main["surface" if predicate.startswith("_") else "abstract"] += 1
                        stats_main.restart([prefix, "not in other", "predicates"])
                        stats_main[predicate] += 1

            stats_update("system_stats", system_index_predicates - gold_index_predicates)
            stats_update("gold_stats", gold_index_predicates - system_index_predicates)
            for predicate in system_index_predicates:
                prefix = "system_stats"
                stats_main.restart([prefix])
                if predicate not in erg:
                    # only count non-unknown
                    if not predicate.endswith("unknown"):
                        stats_main.restart([prefix, "not in erg"])
                        stats_main["count"] += 1
                        stats_main[predicate] += 1
            for predicate in gold_index_predicates:
                prefix = "gold_stats"
                stats_main.restart([prefix])
                if predicate not in erg:
                    # only count non-unknown
                    if not predicate.endswith("unknown"):
                        stats_main.restart([prefix, "not in erg"])
                        stats_main["count"] += 1
                        stats_main[predicate] += 1

    def compare(self, gold, system, erg):
        gold_set = set([k for k in gold._entries.iterkeys()])
        system_set = set([k for k in system._entries.iterkeys()])
        predicates = {}
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
            self._stats["total"] += len(system_index_predicates | gold_index_predicates)
            self._stats["gold"] += len(gold_index_predicates - system_index_predicates)
            self._stats["system"] += len(system_index_predicates - gold_index_predicates)
            self._stats["common"] += len(system_index_predicates & gold_index_predicates)
            if len(gold_index_predicates) > 0:
                predicates[index]["predicate"]["gold"] = list(gold_index_predicates)
            if len(system_index_predicates) > 0:
                predicates[index]["predicate"]["system"] = list(system_index_predicates)
            def stats_update(prefix, predicates):
                for predicate in predicates:
                    self._stats.restart([prefix])
                    if predicate.endswith("unknown"):
                        self._stats["unknown"] += 1
                    else:
                        self._stats.restart([prefix, "not in other"])
                        self._stats["total"] += 1
                        self._stats["surface" if predicate.startswith("_") else "abstract"] += 1
                        self._stats.restart([prefix, "not in other", "predicates"])
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
                            if pred_errors.has_error():
                                self._stats.restart([prefix])
                                self._stats["predicate with incorrect arg"] += 1
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
        if "mrs" in self.mrs:
            result["results"].append(
                {"result-id": "ACE MRS",
                 "mrs": xmrs.Mrs.to_dict(self.mrs["mrs"], properties=True)})
            readings += 1
        if "dmrs" in self._gold_amr:
            result["results"].append(
                {"result-id": "Gold DMRS (convert from penman)",
                 "dmrs": xmrs.Dmrs.to_dict(self._gold_amr["dmrs"], properties=True)})
            readings += 1
        if "dmrs" in self._system_amr:
            result["results"].append(
                {"result-id": "System DMRS (convert from penman)",
                 "dmrs": xmrs.Dmrs.to_dict(self._system_amr["dmrs"], properties=True)})
            readings += 1
        if self._gold_edm is not None and self._system_edm is not None:
            result["results"].append({"result-id": "EDM Stats", "edm": self._edm_compare_result})
            readings += 1
        if self._gold_amr_stats.has_error():
            result["results"].append({"result-id": "Gold AMR Stats", "amr": self._gold_amr_stats.to_dict()})
            readings += 1
        if self._system_amr_stats.has_error():
            result["results"].append({"result-id": "System AMR Stats", "amr": self._system_amr_stats.to_dict()})
            readings += 1
        result["readings"] = readings
        return result

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

    def amr_post_process(self):
        for i in range(len(self.entries)):
            self.entries[i].amr_analyze(self.erg)

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

    def load_amr(self):
        self.parse_mrs(self._files["ace"])
        self.parse_amr_system(self._files["system"])
        self.parse_amr_gold(self._files["gold"])

    def load_json(self):
        self.load_amr()
        self.load_edm()
        self.edm_post_process()
        self.amr_post_process()

    def load_erg(self):
        if self.erg is None:
            self.erg = Erg(self._files["erg"])
            self.erg.parse(self._files["abstract"])
            # self.erg.process_line("")

    def to_json(self, indent=2):
        edm_summary = StatsKeeper("EDM Summary")
        gold_amr_summary = StatsKeeper("AMR Gold Summary")
        system_amr_summary = StatsKeeper("AMR System Summary")
        for i in range(len(self.entries)):
            entry = self.entries[i]
            result = entry.to_json()
            edm_summary.merge(entry._stats)
            gold_amr_summary.merge(entry._gold_amr_stats)
            system_amr_summary.merge(entry._system_amr_stats)
            file_outpath = os.path.join(self.out_dir, "n%s.json" % i)
            with open(file_outpath, "w") as f:
                f.write(json.dumps(result, indent=indent))

        index = len(self.entries)
        for summary in (edm_summary, gold_amr_summary, system_amr_summary):
            result = {
                "input" : "",
                "results" : [],
            }
            # python -m SimpleHTTPServer 8000
            for prefix in ("gold_stats", "system_stats"):
                summary.trim([prefix, "not in other", "predicates"])
                summary.trim([prefix, "not in erg"], limit=0, count=15)
                summary.trim([prefix, "predicate errors"], limit=0, count=15, debug=True)
            result["results"].append({"result-id": summary.title, "summary": summary.to_dict()})
            file_outpath = os.path.join(self.out_dir, "n%s.json" % (index))
            with open(file_outpath, "w") as f:
                f.write(json.dumps(result, indent=indent))
            index += 1

    def convert_mrs(self, mrs, properties=True, indent=None):
        if len(mrs) == 1:
            return {"sentence": mrs[0][len("SKIP: "):]}
        else:
            # CLS = partial(penman.dumps, model=xmrs.Dmrs)
            sent = mrs[0]
            xs = simplemrs.loads_one(" ".join(mrs[1:]))
            if isinstance(xs, xmrs.Mrs):
                x = xs
            else:
                x = xmrs.Mrs.from_xmrs(xs)
            return {"sentence": sent[len("SENT: "):], "mrs": x}

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
            return {"amr": amr_string, "dmrs":x}
        except Exception as e:
            traceback.print_exc()

        return {"amr": amr_string}

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
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ace", type=argparse.FileType("r"),
            default="../../error-analysis-data-2/dev.erg.mrs")
    subparsers = parser.add_subparsers()

    parser_json = subparsers.add_parser("json")
    parser_json.add_argument("--system", type=argparse.FileType("r"),
            default="../../error-analysis-data-2/dev.system.dmrs.amr.lnk")
    parser_json.add_argument("--gold", type=argparse.FileType("r"),
            default="../../error-analysis-data-2/dev.gold.dmrs.amr.lnk")
    parser_json.add_argument("--gold_edm", type=argparse.FileType("r"),
            default="../../error-analysis-data-2/dev.gold.edm.lab")
    parser_json.add_argument("--system_edm", type=argparse.FileType("r"),
            default="../../error-analysis-data-2/dev.system.dmrs.edm.lab")
    parser_json.add_argument("--out_dir", type=str, default="../webpage/data/")
    parser_json.add_argument("--erg", type=argparse.FileType("r"),
            default="../../erg1214/etc/surface.smi")
    parser_json.add_argument("--abstract", type=argparse.FileType("r"),
            default="../../erg1214/etc/abstract.smi")
    parser_json.set_defaults(func=process_main_json)

    parser_error = subparsers.add_parser("error")
    parser_error.add_argument("--erg", type=argparse.FileType("r"),
            default="../../erg1214/etc/surface.smi")
    parser_error.set_defaults(func=process_main_error)

    parser_edm = subparsers.add_parser("edm")
    parser_edm.add_argument("--gold_edm", type=argparse.FileType("r"),
            default="../../error-analysis-data-2/dev.gold.edm.lab")
    parser_edm.add_argument("--system_edm", type=argparse.FileType("r"),
            default="../../error-analysis-data-2/dev.system.dmrs.edm.lab")
    parser_edm.add_argument("--erg", type=argparse.FileType("r"),
            default="../../erg1214/etc/surface.smi")
    parser_edm.add_argument("--abstract", type=argparse.FileType("r"),
            default="../../erg1214/etc/abstract.smi")
    parser_edm.set_defaults(func=process_main_edm)

    parser_erg = subparsers.add_parser("erg")
    parser_erg.add_argument("--erg", type=argparse.FileType("r"),
            default="../../erg1214/etc/surface.smi")
    parser_erg.add_argument("--abstract", type=argparse.FileType("r"),
            default="../../erg1214/etc/abstract.smi")
    parser_erg.set_defaults(func=process_main_erg)

    parser.set_defaults(func=process_main)
    ns = parser.parse_args(args)
    ns.func(ns)


if __name__ == "__main__":
    main()

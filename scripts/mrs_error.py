#!/usr/bin/env python

import os
import sys
import json
import argparse
import penman as PM
import traceback

from collections import defaultdict
from functools import partial

from delphin.codecs import simplemrs
from delphin.mrs import xmrs, eds, penman

class ErgArg(object):
    def __init__(self):
        self._predicate = None
        self._args = defaultdict(list)
    def parse_args(self, predicate, args):
        args = [arg.strip() for arg in args.split("ARG")]
        assert args[0] == "", "first arg empty"
        arg = ErgArg.construct(args[1:])
        self._predicate = predicate
        self._args[arg.arg_index].append(arg)

    @staticmethod
    def construct(arg_array):
        if len(arg_array) == 0:
            return None
        first = arg_array[0]
        # print first, arg_array
        id = first.index(" ")
        arg = ErgArg()
        arg.arg_index = first[0:id]
        arg.arg_attribute = first[id:]
        child = ErgArg.construct(arg_array[1:])
        if child is not None:
            arg._args[child.arg_index].append(child)
        return arg



class Erg(object):
    def __init__(self, input):
        self._tree = defaultdict(ErgArg)
        self.parse(input)
    def parse(self, input):
        head = input.next()
        input.next()
        self._counts = defaultdict(int)
        for line in input:
            (predicate, args) = line.strip().split(":")
            predicate = predicate.strip()
            args = args.strip()
            self._tree[predicate].parse_args(predicate, args)
            self._counts[predicate] += 1

class Processor(object):
    def __init__(self, argparse_ns):
        self.__package_amr_loads = partial(penman.loads, model=xmrs.Dmrs)
        self.mrs = []
        self.system = []
        self.gold = []
        self.out_dir = None
        self._files = {}
        self.erg = None
        self.amr_loads = self._local_amr_loads
        if hasattr(argparse_ns, "out_dir"):
            self.out_dir = argparse_ns.out_dir
        for f in ("ace", "system", "gold", "erg"):
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

    def load_json(self):
        self.parse_mrs(self._files["ace"])
        self.parse_system(self._files["system"])
        self.parse_gold(self._files["gold"])

    def parse_erg(self, input):
        self.erg = Erg(input)

    def analyze(self):
        self.parse_erg(self._files["erg"])


    def to_json(self, indent=2):
        self.load_json()
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
            result["readings"] = readings
            file_outpath = os.path.join(self.out_dir, "n%s.json" % i)
            with open(file_outpath, "w") as f:
                f.write(json.dumps(result, indent=indent))


    def convert_mrs(self, mrs, properties=True, indent=None):
        if len(mrs) == 1:
            self.mrs.append({'sentence': mrs[0][len("SKIP: "):]})
        else:
            # CLS = partial(penman.dumps, model=xmrs.Dmrs)
            sent = mrs[0]
            xs = simplemrs.loads_one(" ".join(mrs[1:]))
            if isinstance(xs, xmrs.Mrs):
                x = xs
            else:
                x = xmrs.Mrs.from_xmrs(xs)
            self.mrs.append({'sentence': sent[len("SENT: "):], 'mrs': x})

    def parse_mrs(self, input):
        mrs = []
        for line in input:
            line = line.strip()
            if len(line) == 0:
                if len(mrs) == 0: continue
                self.convert_mrs(mrs)
                mrs = []
                continue
            mrs.append(line)


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

    def parse_amr_file(self, input):
        out_list = []
        lines = []
        for line in input:
            line = line.rstrip()
            if len(line) == 0:
                if len(lines) > 0:
                    amr = self.convert_amr(lines)
                    if amr is not None:
                        out_list.append(amr)
                lines = []
                continue
            lines.append(line)
        if len(lines) > 0:
            amr = self.convert_amr(lines)
            if amr is not None:
                out_list.append(amr)
        return out_list

    def parse_gold(self, input):
        self.gold = self.parse_amr_file(input)
        self.gold_error = len([s for s in self.system if s.get('dmrs') is None])

    def parse_system(self, input):
        self.system = self.parse_amr_file(input)
        self.system_error = len([s for s in self.system if s.get('dmrs') is None])

def process_main(ns):
    p = Processor(ns)

def process_main_error(ns):
    p = Processor(ns)
    p.analyze()

def process_main_json(ns):
    p = Processor(ns)
    p.to_json()
    print "total: %d gold: {%d/%d} system: {%d/%d}" % (
        len(p.mrs),
        len(p.gold), p.gold_error,
        len(p.system), p.system_error)

def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(prog=sys.argv[0])
    parser.add_argument('--system', type=argparse.FileType('r'),
            default="../../error-analysis-data/dev.system.dmrs.amr")
    parser.add_argument('--gold', type=argparse.FileType('r'),
            default="../../error-analysis-data/dev.gold.dmrs.amr")
    parser.add_argument('--ace', type=argparse.FileType('r'),
            default="../../error-analysis-data/dev.erg.mrs")
    subparsers = parser.add_subparsers()

    parser_json = subparsers.add_parser("json")
    parser_json.add_argument('--out_dir', type=str, default="../webpage/data/")
    parser_json.set_defaults(func=process_main_json)

    parser_error = subparsers.add_parser("error")
    parser.add_argument('--erg', type=argparse.FileType('r'),
            default="../../erg1214/etc/surface.smi")
    parser_error.set_defaults(func=process_main_error)

    parser.set_defaults(func=process_main)
    ns = parser.parse_args(args)
    ns.func(ns)


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import os
import sys
import json
import argparse

from functools import partial

from delphin.codecs import simplemrs
from delphin.mrs import xmrs, eds, penman

class Processor(object):
    def __init__(self, argparse_ns):
        self.mrs = []
        self.system = []
        self.gold = []
        self.amr_loads = partial(penman.loads, model=xmrs.Dmrs)
        self.out_dir = argparse_ns.out_dir
        self.parse_mrs(argparse_ns.ace)
        self.parse_system(argparse_ns.system)
        self.parse_gold(argparse_ns.gold)

    def to_json(self):
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
                f.write(json.dumps(result, indent=None))


    def convert_mrs(self, mrs, properties=True, indent=None):
        if len(mrs) == 1:
            self.mrs.append({'sentence': mrs[0][len("SKIP: "):]})
        else:
            CLS = xmrs.Mrs
            # CLS = partial(penman.dumps, model=xmrs.Dmrs)
            sent = mrs[0]
            xs = simplemrs.loads_one(" ".join(mrs[1:]))
            if isinstance(xs, CLS):
                x = xs
            else:
                x = CLS.from_xmrs(xs)
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
        CLS = xmrs.Dmrs
        amr_string = "\n".join(lines)
        amr_string = amr_string.replace("|", "-")
        try:
            xs = self.amr_loads(amr_string.strip())
            x = CLS.from_xmrs(xs[0])
            return {'amr': amr_string, 'dmrs':x}
        except:
            print "FAILED: ", amr_string
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

    def parse_system(self, input):
        self.system = self.parse_amr_file(input)

def process_main(ns):
    p = Processor(ns)

def process_main_error(ns):
    p = Processor(ns)
    p.to_json()

def process_main_json(ns):
    p = Processor(ns)
    p.to_json()

    print len(p.mrs), len(p.gold), len(p.system)

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
    parser_error.add_argument('--out_dir', type=str, default="../webpage/data/")
    parser_error.set_defaults(func=process_main_error)

    parser.set_defaults(func=process_main)
    ns = parser.parse_args(args)
    ns.func(ns)


if __name__ == "__main__":
    main()

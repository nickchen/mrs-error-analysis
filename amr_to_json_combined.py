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
        self.sentence = []
        self.amr_loads = partial(penman.loads, model=xmrs.Dmrs)
        self.parse_mrs(argparse_ns.ace)
        self.parse_system(argparse_ns.system)
        self.parse_gold(argparse_ns.gold)

    def convert_mrs(self, mrs, properties=True, indent=None):
        CLS = xmrs.Mrs
        # CLS = partial(penman.dumps, model=xmrs.Dmrs)
        sent = mrs[0]
        sent_prefix_len = len("SENT: ")
        sent_mrs = mrs[1:]
        mrs_string = " ".join(sent_mrs)
        self.sentence.append(sent[sent_prefix_len:])
        xs = simplemrs.loads_one(mrs_string)
        if isinstance(xs, CLS):
            self.mrs.append(xs)
        else:
            self.mrs.append(CLS.from_xmrs(xs))

    def parse_mrs(self, input):
        mrs = []
        for line in input:
            line = line.strip()
            if len(line) == 0:
                if len(mrs) == 0: continue
                if len(mrs) == 1:
                    mrs = []
                    continue
                self.convert_mrs(mrs)
                mrs = []
                continue
            mrs.append(line)


    def convert_amr(self, lines):
        CLS = xmrs.Mrs
        amr_string = "\n".join(lines)
        amr_string = amr_string.replace("|", "-")
        amr_string = amr_string.replace("_rel ", " ")
        try:
            xs = self.amr_loads(amr_string.strip())
            return CLS.from_xmrs(xs[0])
        except:
            print "FAILED: ", amr_string
        return None

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
    print len(p.mrs), len(p.sentence), len(p.gold), len(p.system)
#
# def process_mrs(mrs, properties=True, indent=None):
#     CLS = xmrs.Mrs
#     # CLS = partial(penman.dumps, model=xmrs.Dmrs)
#     sent = mrs[0]
#     sent_prefix_len = len("SENT: ")
#     sent_mrs = mrs[1:]
#     mrs_string = " ".join(sent_mrs)
#     xs = simplemrs.loads_one(mrs_string)
#
#     mrs_result = {
#         "result-id": 0,
#         "mrs": CLS.to_dict((xs if isinstance(xs, CLS) else CLS.from_xmrs(xs)), properties=properties),
#     }
#     result = {
#         "readings": 1,
#         "input": sent[sent_prefix_len:],
#         "results": [mrs_result]
#     }
#     print mrs_result
#     print json.dumps(result, indent=indent);

def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(prog=sys.argv[0])
    parser.add_argument('--system', type=argparse.FileType('r'),
            default="../error-analysis-data/dev.system.dmrs.amr")
    parser.add_argument('--gold', type=argparse.FileType('r'),
            default="../error-analysis-data/dev.gold.dmrs.amr")
    parser.add_argument('--ace', type=argparse.FileType('r'),
            default="../error-analysis-data/dev.erg.mrs")

    process_main(parser.parse_args(args))


if __name__ == "__main__":
    main()

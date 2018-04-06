#!/usr/bin/env python

import os
import sys
import json
import argparse

from functools import partial

from delphin.codecs import simplemrs
from delphin.mrs import xmrs, eds, penman

def process_mrs(mrs, properties=True, indent=None):
    CLS = xmrs.Mrs
    # CLS = partial(penman.dumps, model=xmrs.Dmrs)
    sent = mrs[0]
    sent_prefix_len = len("SENT: ")
    sent_mrs = mrs[1:]
    mrs_string = " ".join(sent_mrs)
    xs = simplemrs.loads_one(mrs_string)

    mrs_result = {
        "result-id": 0,
        "mrs": CLS.to_dict((xs if isinstance(xs, CLS) else CLS.from_xmrs(xs)), properties=properties),
    }
    result = {
        "readings": 1,
        "input": sent[sent_prefix_len:],
        "results": [mrs_result]
    }
    print mrs_result
    print json.dumps(result, indent=indent);

def process_file(ns):
    mrs = []
    for line in ns.dev_file:
        line = line.strip()
        if len(line) == 0:
            if len(mrs) == 0: continue
            if len(mrs) == 1:
                mrs = []
                continue
            process_mrs(mrs)
            mrs = []
            continue
        mrs.append(line)

def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(prog=sys.argv[0])
    parser.add_argument('dev_file', type=argparse.FileType('r'))
    parser.add_argument('--gold_standard', type=argparse.FileType('r'))
    ns = parser.parse_args(args)
    process_file(ns)

if __name__ == "__main__":
    main()

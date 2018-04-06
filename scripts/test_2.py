#!/usr/bin/env python

import os
import sys
import json
import argparse


from delphin.mrs import xmrs, penman, eds
from delphin.codecs import simplemrs
from delphin.mrs import xmrs, eds
from delphin.mrs.penman import XMRSCodec
from delphin.mrs import path as mp
from functools import partial

import penman as P

from delphin.mrs.simplemrs import loads_one

nearly_all_dogs_bark = loads_one('''
    [ TOP: h0 INDEX: e2 [ e SF: prop TENSE: pres MOOD: indicative PROG: - PERF: - ]
      RELS: < [ "_nearly_x_deg_rel"<0:6> LBL: h4 ARG0: e5 [ e SF: prop TENSE: untensed MOOD: indicative PROG: - PERF: - ] ARG1: u6 ]
              [ _all_q_rel<7:10> LBL: h4 ARG0: x3 [ x PERS: 3 NUM: pl IND: + ] RSTR: h7 BODY: h8 ]
              [ "_dog_n_1_rel"<11:15> LBL: h9 ARG0: x3 ]
              [ "_bark_v_1_rel"<16:21> LBL: h1 ARG0: e2 ARG1: x3 ] >
      HCONS: < h0 qeq h1 h7 qeq h9 > ]''')

def topdown_paths(x):
    SIMPLEFLAGS = mp.PRED | mp.SUBPATHS | mp.OUTAXES
    for p in mp.explore(x, method='top-down'):
        yield mp.format(p, flags=SIMPLEFLAGS)

for path in sorted(topdown_paths(nearly_all_dogs_bark)):
    print(path)

print nearly_all_dogs_bark.to_dict()

a = xmrs.Dmrs.from_xmrs(nearly_all_dogs_bark)

dumps = partial(penman.dumps, model=xmrs.Dmrs)

for t in a.to_triples():
    print t

print dumps([a])

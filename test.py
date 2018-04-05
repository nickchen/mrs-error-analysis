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

string = """
(n6 / _succeed_v_1_rel
    :ARG2-NEQ (n4 / named_rel
        :CARG "Percival"
        :RSTR-H-of (n3 / proper_q_rel)
        :ARG1-EQ-of (n0 / compound_rel
            :ARG2-NEQ (n1 / _mister_n_1_rel
                :RSTR-H-of (n2 / udef_q_rel)))
        :ARG2-NEQ-of (n5 / parg_d_rel
            :ARG1-EQ n6))
    :EQ-U (n14 / appos_rel
        :ARG1-NEQ (n19 / named_rel
            :CARG "Kadonada"
            :RSTR-H-of (n18 / proper_q_rel)
            :ARG1-EQ-of (n15 / compound_rel
                :ARG2-NEQ (n16 / named_rel
                    :CARG "George"
                    :RSTR-H-of (n17 / proper_q_rel)))
            :ARG2-NEQ-of (n13 / _by_p_rel
                :ARG1-EQ n6
                :ARG2-NEQ (n28 / _and_c_rel
                    :L-INDEX-NEQ (n27 / _chairman_n_of_rel
                        :RSTR-H-of (n25 / udef_q_rel)
                        :ARG1-EQ-of (n20 / compound_rel
                            :ARG2-NEQ (n26 / named_rel
                                :CARG "Facilities"
                                :RSTR-H-of (n24 / proper_q_rel)
                                :ARG1-EQ-of (n21 / compound_rel
                                    :ARG2-NEQ (n22 / named_n_rel
                                        :CARG "US"
                                        :RSTR-H-of (n23 / udef_q_rel))))))
                    :R-INDEX-NEQ (n29 / _president_n_of_rel
                        :RSTR-H-of (n30 / udef_q_rel)))))
        :RSTR-H n28)
    :ARG1-EQ-of (n7 / _on_p_rel
        :ARG2-NEQ (n12 / _basis_n_of_rel
            :ARG1-EQ-of (n9 / compound_rel
                :ARG2-NEQ (n10 / _interim_n_1_rel
                    :RSTR-H-of (n11 / udef_q_rel)))
            :RSTR-H-of (n8 / _a_q_rel))))
"""

string = string.replace("_rel", "")

string_2 = """
(10003 / _bark_v_1
       :lnk "<16:21>"
       :ARG1-NEQ (10002 / _dog_n_1
                        :lnk "<11:15>"
                        :RSTR-H-of (10001 / _all_q
                                          :lnk "<7:10>"
                                          :MOD-EQ-of (10000 / _nearly_x_deg
                                                            :lnk "<0:6>"))))"""

loads = partial(penman.loads, model=xmrs.Dmrs)

graphs = loads(string.strip())

print graphs

def topdown_paths(x):
    SIMPLEFLAGS = mp.PRED | mp.SUBPATHS | mp.OUTAXES
    for p in mp.explore(x, method='top-down'):
        yield mp.format(p, flags=SIMPLEFLAGS)

dumps = partial(penman.dumps, model=xmrs.Dmrs)

CLS = xmrs.Mrs

def dump_dmrs(xs, properties=True,
          pretty_print=False, indent=None, **kwargs):
    if pretty_print and indent is None:
        indent = 2
    x = CLS.from_xmrs(xs[0])
    print x
    print json.dumps(
        [CLS.to_dict(
            (x if isinstance(x, CLS) else CLS.from_xmrs(x)),
            properties=properties
         ) for x in xs],
        indent=indent
    )

dump_dmrs(graphs)

graph = graphs[0]


tri = []
for t in graph.to_triples():
    print t

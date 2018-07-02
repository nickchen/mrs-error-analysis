#!/usr/bin/env python

import sys

def main(inputfile):
    print inputfile
    r = open(inputfile, "r")
    current = []
    index = 0
    for line in r:
        l = line.strip()
        if len(l) == 0:
            outfile = "%s.%d" % (inputfile, index)
            o = open(outfile, "w")
            o.write("\n".join(current))
            index += 1
            current = []
            continue
        current.append(l)

if __name__ == "__main__":
    main(sys.argv[1])

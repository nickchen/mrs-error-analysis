# MRS Error analysis

Project to satisfy the graduation requirement for CLMS at University of Washington

## Prerequisites

* [PyDelphin] https://github.com/delph-in/pydelphin

## See Also

* [DELPH-IN] http://moin.delph-in.net
* [ERS Essence] http://moin.delph-in.net/ErgSemantics/Essence
* [MRS] http://lingo.stanford.edu/sag/papers/copestake.pdf
* [Robust Incremental Neural Semantic Graph Parsing] http://aclweb.org/anthology/P/P17/P17-1112.pdf

## Built With

* [Delphin-Viz] https://github.com/delph-in/delphin-viz
* [Bootstrap] http://getbootstrap.com

## Options

### Local Site
```
python -m SimpleHTTPServer 8000
```

### limit option for debug

Limit the parse entries for quick debugging.

```
 ./mrs_error.py --limit 10 json
```

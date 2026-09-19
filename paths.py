"""Where the paper's evaluation records live.

The frozen snapshot under the project is the single source for every table and
figure in the manuscript.  It sits inside the repository rather than /tmp
because a machine restart clears /tmp and would otherwise take the data with
it.  ADM1_RESULTS overrides the location when re-running against fresh output.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get('ADM1_RESULTS', ROOT / 'results_frozen_20260915'))

EVBASE = RESULTS / 'evbase'     # conventional controller configurations
EVRES = RESULTS / 'evres'       # reward-penalty formulation, per run
EVCMDP = RESULTS / 'evcmdp'     # constrained formulation, per run

def glob(which):
    """which: 'evbase' | 'evres' | 'evcmdp' -> list of json paths."""
    d = {'evbase': EVBASE, 'evres': EVRES, 'evcmdp': EVCMDP}[which]
    if not d.is_dir():
        raise FileNotFoundError(f'{d} not found; set ADM1_RESULTS or restore the snapshot')
    return sorted(str(p) for p in d.glob('*.json'))

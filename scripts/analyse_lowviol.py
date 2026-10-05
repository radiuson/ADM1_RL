#!/usr/bin/env python3
"""What the extended penalty weights say about the low-violation region.

The reported sweep stops at a weight of 5, which leaves the 0--5 % violation
band holding thirteen A2C runs and almost nothing else on the on-policy side.
Weights of 10 and 20 push both classes into that band; this reports what the
class comparison looks like there once they are included.

    python3 scripts/analyse_lowviol.py [extra_evres_dir]

Reads the archive plus, if given, a directory of additional evaluation
records in the same format.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import random
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_tables import DATA, envelope

ON = ('ppo', 'a2c', 'trpo', 'recurrentppo')
OFF = ('sac', 'tqc', 'ddpg', 'td3', 'crossq')
BANDS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 35)]
STEPS = 150000


def load(extra=None):
    """Every reward-penalty run, keyed by family, with its (viol, delta)."""
    _, E = envelope()
    ex = [p[0] for p in E]
    ey = [p[1] for p in E]
    files = glob.glob(f'{DATA}/evres/*.json')
    if extra:
        files += glob.glob(os.path.join(extra, '*.json'))

    runs = collections.defaultdict(list)
    seen = set()
    for f in files:
        d = json.load(open(f))
        if d.get('steps') != STEPS or '_h5_' in d.get('model', ''):
            continue
        a = (d.get('algo') or 'sac').lower()
        if a not in ON + OFF:
            continue
        key = (a, d.get('w'), d.get('seed'))
        if key in seen:                 # one record per cell
            continue
        seen.add(key)
        if not (ex[0] <= d['viol'] <= ex[-1]):
            continue
        runs[a].append((d['viol'],
                        d['ch4'] - float(np.interp(d['viol'], ex, ey)),
                        d.get('w')))
    return runs


def band_table(runs, title):
    print(f'\n{title}')
    print('%-9s %24s %24s %8s' % ('band', 'on-policy', 'off-policy', 'gap'))
    for lo, hi in BANDS:
        o = [d for a in ON for v, d, _ in runs.get(a, []) if lo <= v < hi]
        f = [d for a in OFF for v, d, _ in runs.get(a, []) if lo <= v < hi]
        if not o or not f:
            print('%2d-%2d %% %15s on n=%-3d  off n=%-3d  one side empty'
                  % (lo, hi, '', len(o), len(f)))
            continue
        print('%2d-%2d %% %12s %+6.0f (n=%3d) %12s %+6.0f (n=%3d) %8s'
              % (lo, hi, '', st.median(o), len(o),
                 '', st.median(f), len(f),
                 '%+.0f' % (st.median(o) - st.median(f))))


def families_in_band(runs, lo, hi):
    """Which families actually populate a band, and at which weights."""
    out = {}
    for cls, keys in (('on', ON), ('off', OFF)):
        c = collections.Counter()
        for a in keys:
            for v, _, w in runs.get(a, []):
                if lo <= v < hi:
                    c[f'{a}/{w}'] += 1
        out[cls] = dict(c)
    return out


def main():
    extra = sys.argv[1] if len(sys.argv) > 1 else None
    runs = load(extra)
    n = sum(len(v) for v in runs.values())
    print(f'{n} runs scored against the envelope'
          + (f' (including {extra})' if extra else ' (archive only)'))

    band_table(runs, 'Class medians by violation band')

    for lo, hi in ((0, 2), (2, 5)):
        print(f'\nwhat populates the {lo}-{hi} % band:')
        for cls, d in families_in_band(runs, lo, hi).items():
            print(f'   {cls:<4} {d if d else "none"}')


if __name__ == '__main__':
    main()

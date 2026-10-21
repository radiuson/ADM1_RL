#!/usr/bin/env python3
"""What an unspent excursion allowance costs, measured on the constrained runs.

Section 4.3 states the relationship between the realised violation rate and
methane flow as a slope, so that a family operating further inside its limit
than specified can be read against what production that margin would have
bought.  The conventional envelope estimates the same curve, but only for
conventional controllers; this measures it on the constrained families
themselves.

The design is three families -- CPO, PPOLag and TRPOPID -- swept over the
four cost limits of the main experiment at ten seeds each, algorithm and all
other settings fixed, so the only thing that varies is the rate the policy
is asked to hold.  Methane flow is regressed on the realised rate over three
nested bands, each closed at both ends, so the bands differ only in width.

The interval is a percentile bootstrap resampling (algorithm, seed) clusters
rather than individual runs, so a seed's four runs across the limits stay
together.  The regressor is the realised rate, itself a training outcome, so
the slope is a conditional association and not the causal effect of
relaxing a limit.

    python3 scripts/margin_slope.py
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

DATA = os.environ.get('ADM1_PAPER_DATA',
                      os.path.join(ROOT, 'paper_data_merged'))

FAMS = ('CPO', 'PPOLag', 'TRPOPID')
# Closed at both ends, so the three bands differ in width and nothing else.
BANDS = (('below 2%', 2.0), ('below 5%', 5.0), ('overall', None))
SEED = 20260919
B = 4000


def load():
    rows = []
    for f in glob.glob(f'{DATA}/evcmdp/*.json'):
        d = json.load(open(f))
        if d.get('algo') in FAMS:
            rows.append((d['algo'], d.get('seed'), d['viol'], d['ch4']))
    return rows


def slope(rows):
    x = np.array([r[2] for r in rows], dtype=float)
    y = np.array([r[3] for r in rows], dtype=float)
    if len(set(x.tolist())) < 2:
        return float('nan')
    return float(np.polyfit(x, y, 1)[0])


def main() -> None:
    rows = load()
    print(f'{len(rows)} runs over {len(FAMS)} families '
          f'and {len({r[2] for r in rows})} realised rates')
    for name, hi in BANDS:
        sel = rows if hi is None else [r for r in rows if r[2] <= hi]
        s = slope(sel)

        clusters = {}
        for r in sel:
            clusters.setdefault((r[0], r[1]), []).append(r)
        keys = list(clusters)
        rng = random.Random(SEED)
        draws = []
        for _ in range(B):
            samp = []
            for k in (rng.choice(keys) for _ in keys):
                samp += clusters[k]
            v = slope(samp)
            if v == v:
                draws.append(v)
        draws.sort()
        lo = draws[int(.025 * len(draws))]
        up = draws[int(.975 * len(draws))]
        per = [slope([r for r in sel if r[0] == a]) for a in FAMS]
        print(f'  {name:9s} n={len(sel):3d}  slope {s:5.1f} m3/d per point  '
              f'[{lo:+.0f}, {up:+.0f}]  '
              f'per family {", ".join(f"{a} {v:.1f}" for a, v in zip(FAMS, per))}'
              f'  ({sum(1 for v in per if v > 0)}/{len(per)} positive)')


if __name__ == '__main__':
    main()

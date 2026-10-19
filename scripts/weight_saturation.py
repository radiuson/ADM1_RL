#!/usr/bin/env python3
"""Why raising the penalty weight stops controlling the violation rate.

Section 7.1 of the paper makes two claims that the main design cannot show,
because both need runs outside its four weights and 150k-step budget:

  * raising the weight past the swept range does not restore control -- at
    weights of 10 and 20 the realised rate returns to the levels the low
    weights give;
  * training longer erases the weight's effect -- at 600k steps most PPO and
    RecurrentPPO runs collapse onto one identical point regardless of weight.

The records for both live beside the main set, in results_lowviol/evres and
results_b600k/evres, and this script reads them directly.

    python3 scripts/weight_saturation.py
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIGH_W = os.path.join(ROOT, 'results_lowviol', 'evres')
BUDGET = os.path.join(ROOT, 'results_b600k', 'evres')

# The weights past the swept grid.  results_lowviol/ also holds runs at the
# swept weights from the same round; only 10 and 20 are the point here.
PAST_GRID = ('lw10', 'lw20')
# The nine gradient-based reward-penalty families.  ARS has no gradient step,
# so the weight enters its search differently and it is reported separately.
GRADIENT = ('ppo', 'a2c', 'trpo', 'recurrentppo',
            'sac', 'tqc', 'ddpg', 'td3', 'crossq')
# The weight the swept design ends at, for the comparison the text makes.
TOP_SWEPT = 'lw5'


def read(d):
    for f in sorted(glob.glob(os.path.join(d, '*.json'))):
        r = json.load(open(f))
        r['_algo'] = (r.get('algo') or 'sac').lower()
        yield r


def past_the_grid():
    """Median realised rate per family at weights of 10 and 20."""
    if not os.path.isdir(HIGH_W):
        print(f'  {HIGH_W} not found, skipped')
        return
    runs = list(read(HIGH_W))
    high = defaultdict(list)
    swept = defaultdict(list)
    for r in runs:
        if r['_algo'] not in GRADIENT:
            continue
        if r.get('w') in PAST_GRID:
            high[r['_algo']].append(r['viol'])
        elif r.get('w') == TOP_SWEPT:
            swept[r['_algo']].append(r['viol'])
    n = sum(len(v) for v in high.values())
    print(f'  {len(runs)} runs in the round, {n} at weights of 10 and 20 '
          f'across {len(high)} gradient-based families')
    meds = {a: st.median(v) for a, v in high.items()}
    for a in sorted(meds, key=meds.get):
        print(f'    {a:14s} {meds[a]:5.1f} %  (n={len(high[a]):2d})')
    lo, hi = min(meds.values()), max(meds.values())
    print(f'  median realised rate returns to {lo:.1f}-{hi:.1f} %')
    if swept:
        allswept = [v for vs in swept.values() for v in vs]
        print(f'  against {st.median(allswept):.1f} % at a weight of '
              f'{TOP_SWEPT.replace("lw", "").replace("p", ".")}')


def longer_training():
    """How many 600k-step runs collapse onto a single point."""
    if not os.path.isdir(BUDGET):
        print(f'  {BUDGET} not found, skipped')
        return
    runs = [r for r in read(BUDGET) if r['_algo'] in ('ppo', 'recurrentppo')]
    print(f'  {len(runs)} runs at 600k steps, '
          f'{len({r.get("w") for r in runs})} weights '
          f'x {len({r.get("seed") for r in runs})} seeds '
          f'x {len({r["_algo"] for r in runs})} families')
    pts = Counter((round(r['viol'], 2), round(r['ch4'], 1)) for r in runs)
    (v, c), n = pts.most_common(1)[0]
    print(f'  {n} of the {len(runs)} runs return an identical '
          f'{v:.2f} % at {c:.1f} m3/d')
    same = [r for r in runs
            if round(r['viol'], 2) == v and round(r['ch4'], 1) == c]
    ws = sorted({r.get('w') for r in same})
    print(f'    spanning weights {", ".join(ws)}, so the weight no longer '
          f'separates them')
    # Every one of these terminates early, so the rate is over completed steps.
    ps = same[0].get('per_scenario') or {}
    if ps:
        steps = sum(s['steps'] for s in ps.values())
        viols = sum(s['steps'] * s['viol'] / 100.0 for s in ps.values())
        short = [k for k, s in ps.items() if s['steps'] < 60]
        print(f'    each completes {steps} of 420 steps and violates on '
              f'{viols:.0f} of them')
        print(f'    truncated in {", ".join(short)}')


def main():
    print('--- raising the weight past the swept grid ---')
    past_the_grid()
    print('--- training four times longer ---')
    longer_training()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Stratified bootstrap for the on-policy / off-policy comparison.

Pooling every (weight, seed) run of every family into one sample and resampling
it flat treats runs sharing a weight or a family as independent, which they are
not: a family contributes forty runs that share hyperparameters, and a weight
contributes ten runs that share the reward scale.  A flat resample therefore
understates the spread and can make a difference look better resolved than the
design supports.

Here the resample follows the design.  Families are drawn with replacement
within their update class, then weights (or cost limits) within a family, then
seeds within a weight; the statistic is recomputed on each draw.  The reported
interval is for the difference between the two class medians, which is the
quantity the claim is about -- two separately reported intervals failing to
overlap is a weaker and less direct statement.

    python3 scripts/stratified_stats.py reward
    python3 scripts/stratified_stats.py cmdp
"""
from __future__ import annotations

import glob
import json
import os
import random
import statistics as st
import sys
from collections import defaultdict

import numpy as np

# The per-run evaluation records, shipped in paper_data_merged/ so this runs
# from a clean clone; ADM1_PAPER_DATA overrides it.  See build_tables for what
# distinguishes it from paper_data/.
DATA = os.environ.get(
    'ADM1_PAPER_DATA',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 'paper_data_merged'))

REWARD_ON = ('ppo', 'a2c', 'trpo', 'recurrentppo')
REWARD_OFF = ('sac', 'tqc', 'ddpg', 'td3', 'crossq')
CMDP_ON = ('CUP', 'PCPO', 'TRPOPID', 'FOCOPS', 'CPPOPID',
           'PPOLag', 'CPO', 'TRPOLag', 'OnCRPO', 'P3O')
CMDP_OFF = ('TD3Lag', 'DDPGLag', 'SACPID', 'SACLag')


def envelope():
    # The conventional envelope is defined once, in build_tables, so that the
    # dual-loop family it now includes reaches every script that scores
    # against it rather than only the tables.
    from build_tables import envelope as _env
    B, E = _env()
    xs = [a for a, _, _ in E]
    ys = [b for _, b, _ in E]
    return len(B), len(E), xs, ys


def load(which):
    """-> {family: {stratum: [delta, ...]}}, where stratum is weight or limit."""
    nb, ne, xs, ys = envelope()
    # The analysis set is defined once, in build_tables, so the class medians
    # here and the family medians in the table are taken over the same runs.
    from build_tables import CORRECTED, WEIGHTS

    def delta(d):
        if not (xs[0] <= d['viol'] <= xs[-1]):
            return None
        return d['ch4'] - float(np.interp(d['viol'], xs, ys))

    G = defaultdict(lambda: defaultdict(list))
    if which in ('reward', 'budget'):
        # 'budget' repeats the reward-penalty comparison on the doubled
        # training budget, so the two are computed by the same scheme and the
        # difference between them is the effect of the budget alone.
        steps = 300000 if which == 'budget' else 150000
        # The doubled-budget arm is models_budget alone: three seeds at each
        # of the four weights for every family that was retrained.  The
        # convergence study also holds 300k runs, for SAC only, and pooling
        # them would put a different experiment's ten seeds into this one.
        want_dir = 'models_budget/' if which == 'budget' else None
        # The same analysis set the tables and figures use: the corrected
        # model path where a family has one, and one run per (weight, seed)
        # cell.  Pooling every training of a cell would weight the families
        # that were retrained more heavily than the design intends, so the
        # class medians here and the family medians in the table are taken
        # over the same runs.
        raw = defaultdict(list)
        for f in glob.glob(f'{DATA}/evres/*.json'):
            d = json.load(open(f))
            if d.get('steps') != steps or '_h5_' in d.get('model', ''):
                continue          # ablations are reported separately
            if want_dir and not d.get('model', '').startswith(want_dir):
                continue
            if d.get('w') not in WEIGHTS:
                continue          # the weight-saturation sweep is separate
            a = (d.get('algo') or 'sac').lower()
            if a not in REWARD_ON + REWARD_OFF:
                continue
            want = None if want_dir else CORRECTED.get(a)
            if want and not d.get('model', '').startswith(want):
                continue
            raw[a].append(d)
        for a in raw:
            seen = {}
            for d in sorted(raw[a], key=lambda x: x.get('model', '')):
                seen.setdefault((d.get('w'), d.get('seed')), d)
            for d in seen.values():
                v = delta(d)
                if v is not None:
                    G[a][d.get('w')].append(v)
    else:
        for f in glob.glob(f'{DATA}/evcmdp/*.json'):
            d = json.load(open(f))
            a = d.get('algo')
            if a not in CMDP_ON + CMDP_OFF:
                continue
            v = delta(d)
            if v is not None:
                G[a][str(d.get('cost_limit'))].append(v)
    return G, nb, ne


def class_median_seedblock(G, fams, rng, resample):
    """Median with the families and strata held fixed, resampling seeds only.

    The families and the weights are chosen design levels, not a sample from
    a population of algorithms, so resampling them describes a superpopulation
    the design never drew from.  This variant keeps every family and every
    stratum and resamples the seeds inside each cell, which is the randomness
    the experiment actually repeats.  Seeds are drawn once per family and
    reused across that family's strata, so a seed's runs stay together.
    """
    vals = []
    for a in fams:
        strata = list(G[a])
        if not strata:
            continue
        n = max(len(G[a][w]) for w in strata)
        idx = [rng.randrange(n) for _ in range(n)] if resample else list(range(n))
        for w in strata:
            runs = G[a][w]
            if not runs:
                continue
            vals += [runs[i % len(runs)] for i in idx]
    return st.median(vals) if vals else float('nan')


def class_median(G, fams, rng, resample):
    """Median over a (possibly resampled) draw of families/strata/seeds."""
    fs = [rng.choice(fams) for _ in fams] if resample else list(fams)
    vals = []
    for a in fs:
        strata = list(G[a])
        if not strata:
            continue
        ss = [rng.choice(strata) for _ in strata] if resample else strata
        for w in ss:
            runs = G[a][w]
            if not runs:
                continue
            vals += ([rng.choice(runs) for _ in runs] if resample else runs)
    return st.median(vals) if vals else float('nan')


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'reward'
    ON, OFF = ((REWARD_ON, REWARD_OFF) if which in ('reward', 'budget')
               else (CMDP_ON, CMDP_OFF))
    G, nb, ne = load(which)
    ON = [a for a in ON if G[a]]
    OFF = [a for a in OFF if G[a]]
    rng = random.Random(20260919)
    B = 4000

    on0 = class_median(G, ON, rng, False)
    off0 = class_median(G, OFF, rng, False)
    diffs, ons, offs = [], [], []
    for _ in range(B):
        o = class_median(G, ON, rng, True)
        f = class_median(G, OFF, rng, True)
        ons.append(o)
        offs.append(f)
        diffs.append(o - f)
    diffs.sort(); ons.sort(); offs.sort()
    lo, hi = diffs[int(.025 * B)], diffs[int(.975 * B)]

    # Seed-only resample, families and strata fixed.
    sb = []
    for _ in range(B):
        sb.append(class_median_seedblock(G, ON, rng, True)
                  - class_median_seedblock(G, OFF, rng, True))
    sb.sort()
    sb_lo, sb_hi = sb[int(.025 * B)], sb[int(.975 * B)]
    sb_pos = sum(1 for d in sb if d > 0) / B

    print(f'{which}: envelope {nb} configurations -> {ne} points')
    print(f'  [seed-block, families and strata fixed] '
          f'diff {class_median_seedblock(G, ON, rng, False) - class_median_seedblock(G, OFF, rng, False):+.0f}'
          f'  [{sb_lo:+.0f}, {sb_hi:+.0f}]  P(>0)={100*sb_pos:.1f}%')
    print(f'  on-policy  n={len(ON)} families  median {on0:+.0f}  '
          f'[{ons[int(.025*B)]:+.0f}, {ons[int(.975*B)]:+.0f}]')
    print(f'  off-policy n={len(OFF)} families  median {off0:+.0f}  '
          f'[{offs[int(.025*B)]:+.0f}, {offs[int(.975*B)]:+.0f}]')
    print(f'  difference (on - off)  {on0-off0:+.0f}  [{lo:+.0f}, {hi:+.0f}]  '
          f'{"excludes 0" if lo > 0 or hi < 0 else "includes 0"}')
    print(f'  P(on > off) = {sum(1 for d in diffs if d > 0)/B:.3f}')


if __name__ == '__main__':
    main()

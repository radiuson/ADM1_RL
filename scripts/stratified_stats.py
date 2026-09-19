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

SP = os.environ.get('ADM1_S7', '/tmp/claude-1000/-home-ihpc-code/'
                    'd8e23994-23f1-4be7-bbbc-30cb1d90de5a/scratchpad')
FROZEN = 'results_frozen_20260915'

REWARD_ON = ('ppo', 'a2c', 'trpo', 'recurrentppo')
REWARD_OFF = ('sac', 'tqc', 'ddpg', 'td3', 'crossq')
CMDP_ON = ('CUP', 'PCPO', 'TRPOPID', 'FOCOPS', 'CPPOPID',
           'PPOLag', 'CPO', 'TRPOLag', 'OnCRPO', 'P3O')
CMDP_OFF = ('TD3Lag', 'DDPGLag', 'SACPID', 'SACLag')


def envelope():
    B = [json.load(open(f)) for f in glob.glob(f'{FROZEN}/evbase/*.json')]
    B += [json.load(open(f)) for f in glob.glob(f'{SP}/evbase_ext/*.json')]
    pts = sorted((b['viol'], b['ch4'], b['name']) for b in B)
    E, best = [], -1.0
    for v, c, n in pts:
        if c > best:
            E.append((v, c))
            best = c
    xs = [a for a, _ in E]
    ys = [b for _, b in E]
    return len(B), len(E), xs, ys


def load(which):
    """-> {family: {stratum: [delta, ...]}}, where stratum is weight or limit."""
    nb, ne, xs, ys = envelope()

    def delta(d):
        if not (xs[0] <= d['viol'] <= xs[-1]):
            return None
        return d['ch4'] - float(np.interp(d['viol'], xs, ys))

    G = defaultdict(lambda: defaultdict(list))
    if which == 'reward':
        for f in glob.glob(f'{SP}/s7/evres/*.json'):
            d = json.load(open(f))
            if d.get('steps') != 150000 or '_h5_' in d.get('model', ''):
                continue          # ablations are reported separately
            a = (d.get('algo') or 'sac').lower()
            if a not in REWARD_ON + REWARD_OFF:
                continue
            v = delta(d)
            if v is not None:
                G[a][d.get('w')].append(v)
    else:
        for f in glob.glob(f'{SP}/s7/evcmdp/*.json'):
            d = json.load(open(f))
            a = d.get('algo')
            if a not in CMDP_ON + CMDP_OFF:
                continue
            v = delta(d)
            if v is not None:
                G[a][str(d.get('cost_limit'))].append(v)
    return G, nb, ne


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
    ON, OFF = ((REWARD_ON, REWARD_OFF) if which == 'reward'
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

    print(f'{which}: envelope {nb} configurations -> {ne} points')
    print(f'  on-policy  n={len(ON)} families  median {on0:+.0f}  '
          f'[{ons[int(.025*B)]:+.0f}, {ons[int(.975*B)]:+.0f}]')
    print(f'  off-policy n={len(OFF)} families  median {off0:+.0f}  '
          f'[{offs[int(.025*B)]:+.0f}, {offs[int(.975*B)]:+.0f}]')
    print(f'  difference (on - off)  {on0-off0:+.0f}  [{lo:+.0f}, {hi:+.0f}]  '
          f'{"excludes 0" if lo > 0 or hi < 0 else "includes 0"}')
    print(f'  P(on > off) = {sum(1 for d in diffs if d > 0)/B:.3f}')


if __name__ == '__main__':
    main()

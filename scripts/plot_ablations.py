#!/usr/bin/env python3
"""Loaders for the Section 7 sweep, shared with the table that reports it.

Every claim the text makes about the sweep is a count or a value, so the
results are tabulated by ``build_ablation_table`` rather than drawn; this
module keeps the record loading and the class statistic that table uses.

    python3 scripts/build_ablation_table.py
"""
from __future__ import annotations

import collections
import glob
import json
import random
import statistics as st

import numpy as np

from build_tables import CORRECTED, DATA, WEIGHTS, envelope

ON = ('ppo', 'a2c', 'trpo', 'recurrentppo')
OFF = ('sac', 'tqc', 'ddpg', 'td3', 'crossq')
FAMS = ON + OFF
LABEL = {'ppo': 'PPO', 'a2c': 'A2C', 'trpo': 'TRPO',
         'recurrentppo': 'RecurrentPPO', 'sac': 'SAC', 'tqc': 'TQC',
         'ddpg': 'DDPG', 'td3': 'TD3', 'crossq': 'CrossQ'}

# The swept settings, in the order the sweep varies them.
SETTINGS = [('LR_1e-4', 'lr $10^{-4}$'), ('LR_1e-3', 'lr $10^{-3}$'),
            ('ARCH_64_64', 'net 64,64'), ('ARCH_400_300', 'net 400,300')]


_, E = envelope()
XS = [p[0] for p in E]
YS = [p[1] for p in E]


def delta(d):
    """Methane above the conventional envelope at this run's violation rate."""
    if not (XS[0] <= d['viol'] <= XS[-1]):
        return None
    return d['ch4'] - float(np.interp(d['viol'], XS, YS))


def _keep(d, steps):
    """The main design: four swept weights, corrected settings, one budget."""
    if d.get('steps') != steps or '_h5_' in d.get('model', ''):
        return False
    if d.get('w') not in WEIGHTS:
        return False
    a = (d.get('algo') or 'sac').lower()
    want = CORRECTED.get(a)
    return not (want and not d.get('model', '').startswith(want))


def load_main(steps=150000, stacked=False):
    R = collections.defaultdict(list)
    for f in glob.glob(f'{DATA}/evres/*.json'):
        d = json.load(open(f))
        a = (d.get('algo') or 'sac').lower()
        if stacked:
            if '_h5_' not in d.get('model', ''):
                continue
        elif not _keep(d, steps):
            continue
        v = delta(d)
        if v is not None:
            R[a].append(v)
    return R


def load_sweep():
    """Per-setting, per-family deltas from the hyperparameter sweep."""
    S = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in glob.glob(f'{DATA}/evres_hp/*.json'):
        d = json.load(open(f))
        cell = d['model'].split('models_hp/')[1].split('/')[0]
        for fam in FAMS:
            if cell.startswith(fam + '_'):
                v = delta(d)
                if v is not None:
                    S[cell[len(fam) + 1:]][fam].append(v)
                break
    return S


def class_gap(per_family, rng, B=4000):
    """Difference of class medians, and its interval over resampled families."""
    on = [st.median(per_family[a]) for a in ON if per_family.get(a)]
    off = [st.median(per_family[a]) for a in OFF if per_family.get(a)]
    if not on or not off:
        return None
    point = st.median(on) - st.median(off)
    draws = sorted(st.median([rng.choice(on) for _ in on])
                   - st.median([rng.choice(off) for _ in off])
                   for _ in range(B))
    return point, draws[int(.025 * B)], draws[int(.975 * B)]

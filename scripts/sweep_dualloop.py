#!/usr/bin/env python3
"""Sweep the dual-loop PI family and score it against the existing envelope.

The three conventional families hold the feed-strength multiplier constant for
the episode, so the envelope they define is the best of a static-tuning set,
while a learned policy sets both inputs at every step.  A difference from that
envelope therefore carries the extra dynamic freedom as well as the control
law.  This sweeps a conventional family that has the same freedom -- a PI loop
on the feed rate and a proportional loop with a dead band on the multiplier --
so the two can be compared without it.

    python3 scripts/sweep_dualloop.py [outdir]

Writes one record per configuration, in the same form as the existing
conventional evaluations.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

SC = ['low_load', 'nominal', 'plant_load', 'high_load', 'peak_load',
      'fog_surge', 'elevated_start']
VFA_SOFT = 0.320
VFA_HARD = 1.600
MG = 1000 / 1.0667
Q_LO, Q_HI = 41.0, 159.0

# The feed-rate axes are those of the existing PI sweep; the two new axes are
# the multiplier gain and the dead band it acts outside of.
SETPOINTS = [60, 100, 140, 180, 220, 260, 300, 340, 380]
KCS = [0.05, 0.1, 0.35, 1.0, 2.5]
TAUS = [1.0, 3.0, 8.0, 20.0]
KMS = [0.0005, 0.002, 0.008, 0.03]
BANDS = [15.0, 40.0, 90.0]


def evaluate(args):
    sp, kc, ti, km, band = args
    from env.adm1_gym_env_std import ADM1Env_Std as E
    from training.baselines import DualLoopPI
    V, C = [], []
    for s in SC:
        env = E(s, obs_mode='scada', step_size=1.0)
        env.reset(seed=7)
        ctrl = DualLoopPI(sp, Q_LO, Q_HI, Kc=kc, tau_i=ti, Km=km, band=band)
        ctrl.reset()
        q, m = Q_HI, 1.0
        for _ in range(60):
            st = env.solver.state
            vfa = sum(st[k] for k in ('S_va', 'S_bu', 'S_pro', 'S_ac'))
            q, m = ctrl.act(vfa * MG, st['S_IC'],
                            -np.log10(max(st['S_H_ion'], 1e-14)),
                            env.solver.q_ch4, q, m)
            _, _, te, tr, _ = env.step(np.array([q, m], dtype=np.float32))
            st = env.solver.state
            V.append(sum(st[k] for k in ('S_va', 'S_bu', 'S_pro', 'S_ac')))
            C.append(env.solver.q_ch4)
            if te or tr:
                break
    V, C = np.array(V), np.array(C)
    return {'name': f'dual_sp{sp:g}_Kc{kc:g}_ti{ti:g}_Km{km:g}_b{band:g}',
            'kind': 'dual',
            'viol': float(100 * np.mean(V > VFA_SOFT)),
            'viol_hard': float(100 * np.mean(V > VFA_HARD)),
            'ch4': float(C.mean()),
            'vfa_med': float(np.median(V) * MG),
            'vfa_max': float(V.max() * MG),
            'steps': int(len(V))}


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, 'results_dualloop')
    os.makedirs(out, exist_ok=True)
    grid = list(itertools.product(SETPOINTS, KCS, TAUS, KMS, BANDS))
    print(f'{len(grid)} configurations', flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=20) as ex:
        for rec in ex.map(evaluate, grid, chunksize=4):
            with open(os.path.join(out, rec['name'] + '.json'), 'w') as fh:
                json.dump(rec, fh)
            done += 1
            if done % 200 == 0:
                print(f'  {done}/{len(grid)}', flush=True)
    print(f'wrote {done} records to {out}')


if __name__ == '__main__':
    main()

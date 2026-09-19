#!/usr/bin/env python3
"""Evaluate one conventional-control configuration on the seven-scenario suite.

Same protocol as eval_sb3.py and eval_cmdp.py -- seven scenarios, 200 steps,
seed 7, soft limit 0.320 kg COD/m3 -- so the output is directly comparable with
the learned policies and can be merged into the conventional envelope.

Written to extend that envelope towards zero violation: the existing PI
configurations stop at a setpoint of 150 mg/L, which leaves the envelope
undefined below 0.50 % violation, and runs below that bound cannot be compared
against it.

    python3 evaluation/eval_baseline_ext.py pi 120 0.10 20 out.json
    python3 evaluation/eval_baseline_ext.py const 55 - - out.json
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.adm1_gym_env_std import ADM1Env_Std as E
from training.baselines import PIFeed, ConstantFeed

SC = ['low_load', 'nominal', 'plant_load', 'high_load', 'peak_load',
      'fog_surge', 'elevated_start']
VFA_SOFT = 0.320
VFA_HARD = 1.600
MG = 1000 / 1.0667
Q_LO, Q_HI = 41.0, 159.0


def run(kind, p1, Kc, tau_i, mult=1.3):
    if kind == 'pi':
        C = PIFeed(p1, Q_LO, Q_HI, Kc=Kc, tau_i=tau_i, mult=mult)
    elif kind == 'const':
        C = ConstantFeed(p1, mult=mult)
    else:
        raise SystemExit(f'unknown kind {kind}')

    V, CH = [], []
    for s in SC:
        e = E(s, obs_mode='scada', step_size=1.0)
        e.reset(seed=7)
        C.reset()
        q, mult = Q_HI, 1.3
        for _ in range(200):
            st = e.solver.state
            vfa = sum(st[k] for k in ('S_va', 'S_bu', 'S_pro', 'S_ac'))
            q, mult = C.act(vfa * MG, st['S_IC'],
                            -np.log10(max(st['S_H_ion'], 1e-14)),
                            e.solver.q_ch4, q, mult)
            _, _, te, tr, _ = e.step(np.array([q, mult], dtype=np.float32))
            st = e.solver.state
            V.append(sum(st[k] for k in ('S_va', 'S_bu', 'S_pro', 'S_ac')))
            CH.append(e.solver.q_ch4)
            if te or tr:
                break
    return np.array(V), np.array(CH), C.name


def main():
    kind, p1 = sys.argv[1], float(sys.argv[2])
    Kc = float(sys.argv[3]) if sys.argv[3] != '-' else 0.10
    ti = float(sys.argv[4]) if sys.argv[4] != '-' else 20.0
    out = sys.argv[5]
    mult = float(sys.argv[6]) if len(sys.argv) > 6 else 1.3

    V, CH, name = run(kind, p1, Kc, ti, mult)
    json.dump({'name': name, 'kind': kind,
               'ch4': float(CH.mean()),
               'viol': float(100 * np.mean(V > VFA_SOFT)),
               'viol_hard': float(100 * np.mean(V > VFA_HARD)),
               'vfa_max': float(V.max() * MG),
               'vfa_med': float(np.median(V) * MG),
               'Kc': Kc, 'tau_i': ti, 'mult': mult}, open(out, 'w'))


if __name__ == '__main__':
    main()

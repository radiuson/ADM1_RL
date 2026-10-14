#!/usr/bin/env python3
"""PI tunings that leave the feed strength at the scenario's own value.

The conventional sweep gives every tuning a feed-strength multiplier of at
least 1.1, so even the static families raise the organic load above what the
scenario specifies.  A plant that cannot change its feed composition at all
operates at the nominal strength.  This sweeps the same PI grid at a
multiplier of 1.0, which is that case.

    python3 scripts/sweep_nominal_mult.py [outdir]
"""
from __future__ import annotations

import itertools, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SC = ['low_load', 'nominal', 'plant_load', 'high_load', 'peak_load',
      'fog_surge', 'elevated_start']
VFA_SOFT, VFA_HARD = 0.320, 1.600
MG = 1000 / 1.0667
Q_LO, Q_HI = 41.0, 159.0

SETPOINTS = [20, 45, 70, 100, 130, 160, 190, 220, 250, 280, 310, 340, 370, 400]
KCS = [0.03, 0.1, 0.35, 1.0, 2.5, 5.0]
TAUS = [1.0, 3.0, 8.0, 20.0, 60.0]


def evaluate(args):
    sp, kc, ti = args
    from env.adm1_gym_env_std import ADM1Env_Std as E
    from training.baselines import PIFeed
    V, C = [], []
    for s in SC:
        env = E(s, obs_mode='scada', step_size=1.0)
        env.reset(seed=7)
        ctrl = PIFeed(sp, Q_LO, Q_HI, Kc=kc, tau_i=ti, mult=1.0)
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
    return {'name': f'pinom_sp{sp:g}_Kc{kc:g}_ti{ti:g}_m1', 'kind': 'pi_nominal',
            'viol': float(100 * np.mean(V > VFA_SOFT)),
            'viol_hard': float(100 * np.mean(V > VFA_HARD)),
            'ch4': float(C.mean()),
            'vfa_med': float(np.median(V) * MG),
            'steps': int(len(V))}


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'results_pinominal')
    os.makedirs(out, exist_ok=True)
    grid = list(itertools.product(SETPOINTS, KCS, TAUS))
    print(f'{len(grid)} configurations', flush=True)
    n = 0
    with ProcessPoolExecutor(max_workers=6) as ex:
        for rec in ex.map(evaluate, grid, chunksize=4):
            json.dump(rec, open(os.path.join(out, rec['name'] + '.json'), 'w'))
            n += 1
    print(f'wrote {n} records')


if __name__ == '__main__':
    main()

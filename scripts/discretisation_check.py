#!/usr/bin/env python3
"""Discretisation sensitivity across controllers, not one policy.

Table 1 refines the operator-splitting interval on a single trained policy and
reports the change in mean methane.  Two things are missing from that check.
It covers one controller, so it cannot show that the ordering between families
is stable; and it reports an absolute flow, whereas every comparison in this
paper is a difference from the conventional envelope at matched violation
rate, which moves when both the run and the envelope move.

This refines the interval for a spread of controllers -- conventional PI
tunings and learned policies from both classes, chosen at low, middle and
high violation rates -- and reports the change in that difference.

    python3 scripts/discretisation_check.py [out.json]
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from build_tables import DATA

SC = ['low_load', 'nominal', 'plant_load', 'high_load', 'peak_load',
      'fog_surge', 'elevated_start']
VFA_SOFT = 0.320
MG = 1000 / 1.0667
Q_LO, Q_HI = 41.0, 159.0

FINE = 0.01041667 / 4          # 3.75 min
COARSE = 0.01041667            # 15 min, the setting used throughout


def envelope():
    # One definition of the conventional envelope, in build_tables, so the
    # dual-loop family reaches every script that scores against it.
    from build_tables import envelope as _env
    _, E = _env()
    return [a for a, _, _ in E], [b for _, b, _ in E]


def run_conventional(kind, p1, Kc, ti, mult, dt):
    from env.adm1_gym_env_std import ADM1Env_Std as E
    from training.baselines import ConstantFeed, PIFeed
    V, C = [], []
    for s in SC:
        env = E(s, obs_mode='scada', step_size=1.0)
        env.reset(seed=7)
        env.solver.internal_dt = dt
        ctrl = (PIFeed(p1, Q_LO, Q_HI, Kc=Kc, tau_i=ti, mult=mult)
                if kind == 'pi' else ConstantFeed(p1, mult=mult))
        ctrl.reset()
        q, m = Q_HI, mult
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
    return 100.0 * float(np.mean(V > VFA_SOFT)), float(C.mean())


def run_policy(model_path, algo, dt):
    from evaluation.eval_sb3 import E, NormalizedADM1Env
    import stable_baselines3 as sb3
    import sb3_contrib as sbc
    MAP = {'sac': sb3.SAC, 'ppo': sb3.PPO, 'a2c': sb3.A2C, 'ddpg': sb3.DDPG,
           'td3': sb3.TD3, 'tqc': sbc.TQC, 'trpo': sbc.TRPO,
           'recurrentppo': sbc.RecurrentPPO, 'crossq': sbc.CrossQ,
           'ars': sbc.ARS}
    model = MAP[algo].load(model_path, device='cpu')
    V, C = [], []
    for s in SC:
        env = NormalizedADM1Env(E(s, obs_mode='scada', step_size=1.0))
        obs, _ = env.reset(seed=7)
        env.env.solver.internal_dt = dt
        state = None
        for _ in range(60):
            if algo == 'recurrentppo':
                act, state = model.predict(obs, state=state, deterministic=True)
            else:
                act, _ = model.predict(obs, deterministic=True)
            obs, _, te, tr, _ = env.step(act)
            st = env.env.solver.state
            V.append(sum(st[k] for k in ('S_va', 'S_bu', 'S_pro', 'S_ac')))
            C.append(env.env.solver.q_ch4)
            if te or tr:
                break
    V, C = np.array(V), np.array(C)
    return 100.0 * float(np.mean(V > VFA_SOFT)), float(C.mean())


def pick_policies():
    """One learned policy per class at a low, middle and high violation rate."""
    from plot_figures import load_reward
    R = load_reward()
    ON = ('ppo', 'trpo')
    OFF = ('sac', 'tqc')
    out = []
    for fams, tag in ((ON, 'on'), (OFF, 'off')):
        runs = [(d['viol'], d['ch4'], d['model'], a)
                for a in fams for d in R.get(a, [])]
        runs.sort()
        for q, lab in ((0.1, 'low'), (0.5, 'mid'), (0.9, 'high')):
            v, c, m, a = runs[int(q * (len(runs) - 1))]
            out.append((f'{a} ({tag}, {lab})', a, m, v, c))
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else None
    xs, ys = envelope()

    def delta(v, c):
        if not (xs[0] <= v <= xs[-1]):
            return None
        return c - float(np.interp(v, xs, ys))

    # Conventional tunings at a low, middle and high violation rate.
    conv = [('PI sp=125 (low)', 'pi', 125.0, 0.5, 5.0, 1.3),
            ('PI sp=225 (mid)', 'pi', 225.0, 1.0, 3.0, 1.3),
            ('PI sp=325 (high)', 'pi', 325.0, 0.5, 5.0, 1.3)]

    rows = []
    print('%-26s %18s %18s %9s' % ('controller', '15 min (v%, CH4)',
                                   '3.75 min', 'dD'))
    for name, kind, p1, Kc, ti, mult in conv:
        v0, c0 = run_conventional(kind, p1, Kc, ti, mult, COARSE)
        v1, c1 = run_conventional(kind, p1, Kc, ti, mult, FINE)
        d0, d1 = delta(v0, c0), delta(v1, c1)
        dd = (d1 - d0) if (d0 is not None and d1 is not None) else float('nan')
        rows.append({'name': name, 'coarse': [v0, c0], 'fine': [v1, c1],
                     'D_coarse': d0, 'D_fine': d1, 'dD': dd})
        print('%-26s %8.2f %9.0f %8.2f %9.0f %9.1f'
              % (name, v0, c0, v1, c1, dd))

    for name, algo, model, _, _ in pick_policies():
        v0, c0 = run_policy(model, algo, COARSE)
        v1, c1 = run_policy(model, algo, FINE)
        d0, d1 = delta(v0, c0), delta(v1, c1)
        dd = (d1 - d0) if (d0 is not None and d1 is not None) else float('nan')
        rows.append({'name': name, 'coarse': [v0, c0], 'fine': [v1, c1],
                     'D_coarse': d0, 'D_fine': d1, 'dD': dd})
        print('%-26s %8.2f %9.0f %8.2f %9.0f %9.1f'
              % (name, v0, c0, v1, c1, dd))

    good = [r['dD'] for r in rows if r['dD'] == r['dD']]
    if good:
        print('\n|dD| max %.1f, median %.1f m3/d over %d controllers'
              % (max(abs(x) for x in good),
                 sorted(abs(x) for x in good)[len(good) // 2], len(good)))
    if out_path:
        json.dump(rows, open(out_path, 'w'), indent=1)
        print(f'wrote {out_path}')


if __name__ == '__main__':
    main()

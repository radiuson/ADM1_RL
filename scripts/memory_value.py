#!/usr/bin/env python3
"""How much the observation history adds, measured without training anything.

Section 7 reports that memory widens the class gap rather than closing it,
and leaves the reason open.  The premise behind expecting memory to help is
that the task is partially observed: the policy sees six signals where the
model carries 38 states.  That premise is testable directly.  If the history
of the six observed signals predicts the hidden state, or the next
constrained variable, better than the current observation alone, then memory
has something to work with and the shortfall is about optimisation.  If it
does not, the architecture was given a capacity it could not use.

The test rolls out the conventional controllers -- no learned policy is
involved, so nothing has to be trained -- and fits ridge regressions from
the observation window to two targets:

    next-step VFA       what a safety-aware policy needs to anticipate
    hidden state        the 32 ADM1 states the policy never sees

at window lengths of 1 (memoryless) and 5 (what stacking supplies).  The
quantity reported is the variance explained on held-out trajectories, so a
window that adds nothing shows as no gain.

    python3 scripts/memory_value.py [out.json]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.adm1_gym_env_std import ADM1Env_Std as E
from training.baselines import ConstantFeed, PIFeed

SC = ['low_load', 'nominal', 'plant_load', 'high_load', 'peak_load',
      'fog_surge', 'elevated_start']
Q_LO, Q_HI = 41.0, 159.0
MG = 1000 / 1.0667
WINDOWS = (1, 2, 5)

# A spread of conventional tunings, so the rollouts cover the operating
# region rather than one trajectory.  These are configurations the sweep
# already evaluated.
CONTROLLERS = [('pi', 125.0, 0.5, 5.0), ('pi', 175.0, 0.5, 5.0),
               ('pi', 225.0, 1.0, 3.0), ('pi', 275.0, 0.35, 8.0),
               ('const', 90.0, None, None), ('const', 130.0, None, None)]


def vfa(state):
    return sum(state[k] for k in ('S_va', 'S_bu', 'S_pro', 'S_ac'))


def rollout(kind, p1, Kc, ti, scenario, seed=7):
    """One trajectory: the six observed signals and the full state."""
    env = E(scenario, obs_mode='scada', step_size=1.0)
    env.reset(seed=seed)
    ctrl = (PIFeed(p1, Q_LO, Q_HI, Kc=Kc, tau_i=ti, mult=1.3) if kind == 'pi'
            else ConstantFeed(p1, mult=1.3))
    ctrl.reset()
    q, mult = Q_HI, 1.3
    obs, hidden = [], []
    for _ in range(60):
        s = env.solver.state
        q, mult = ctrl.act(vfa(s) * MG, s['S_IC'],
                           -np.log10(max(s['S_H_ion'], 1e-14)),
                           env.solver.q_ch4, q, mult)
        o, _, te, tr, _ = env.step(np.array([q, mult], dtype=np.float32))
        s = env.solver.state
        obs.append(np.asarray(o, dtype=float))
        hidden.append(np.array([s[k] for k in sorted(s) if
                                isinstance(s[k], (int, float))], dtype=float))
        if te or tr:
            break
    return np.array(obs), np.array(hidden)


def windowed(seq, w):
    """Rows of w consecutive observations, flattened."""
    if len(seq) <= w:
        return np.empty((0, seq.shape[1] * w))
    return np.stack([seq[i - w + 1:i + 1].ravel()
                     for i in range(w - 1, len(seq))])


def ridge_r2(Xtr, Ytr, Xte, Yte, lam=1.0):
    """Variance explained on held-out rows, averaged over targets."""
    mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-9
    my, sy = Ytr.mean(0), Ytr.std(0) + 1e-9
    A = (Xtr - mx) / sx
    Bm = (Ytr - my) / sy
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Bm)
    P = ((Xte - mx) / sx) @ W
    T = (Yte - my) / sy
    ss_res = ((T - P) ** 2).sum(0)
    ss_tot = ((T - T.mean(0)) ** 2).sum(0) + 1e-12
    return float(np.mean(1 - ss_res / ss_tot))


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else None

    # Held out by scenario, so the test measures generalisation across
    # operating conditions rather than interpolation within one trajectory.
    train_sc, test_sc = SC[:5], SC[5:]
    print(f'train scenarios: {train_sc}')
    print(f'test scenarios:  {test_sc}')

    data = {}
    for sc in SC:
        for c in CONTROLLERS:
            o, h = rollout(c[0], c[1], c[2], c[3], sc)
            if len(o) > 6:
                data[(sc, c)] = (o, h)
    print(f'{len(data)} rollouts')

    # Hidden-state columns the observation does not already contain.
    results = {}
    for w in WINDOWS:
        for target in ('next_vfa', 'hidden'):
            Xtr, Ytr, Xte, Yte = [], [], [], []
            for (sc, _), (o, h) in data.items():
                X = windowed(o, w)
                if target == 'next_vfa':
                    Y = o[w:, 0:1]            # VFA is observation 0
                    X = X[:len(Y)]
                else:
                    Y = h[w - 1:]
                    n = min(len(X), len(Y))
                    X, Y = X[:n], Y[:n]
                if len(X) == 0:
                    continue
                (Xtr if sc in train_sc else Xte).append(X)
                (Ytr if sc in train_sc else Yte).append(Y)
            r2 = ridge_r2(np.vstack(Xtr), np.vstack(Ytr),
                          np.vstack(Xte), np.vstack(Yte))
            results[f'{target}_w{w}'] = r2

    print()
    print('held-out variance explained')
    print('%-12s %8s %8s %8s   %s' % ('target', 'w=1', 'w=2', 'w=5',
                                      'gain w1->w5'))
    for target, label in (('next_vfa', 'next VFA'), ('hidden', 'hidden state')):
        r = [results[f'{target}_w{w}'] for w in WINDOWS]
        print('%-12s %8.3f %8.3f %8.3f   %+.3f'
              % (label, r[0], r[1], r[2], r[2] - r[0]))

    if out_path:
        json.dump(results, open(out_path, 'w'), indent=1)
        print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()

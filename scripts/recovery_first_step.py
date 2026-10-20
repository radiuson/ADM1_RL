#!/usr/bin/env python3
"""Why the recovery-from-acidification scenario is excluded.

Section 3.4 drops acidified_recovery because its initial total VFA is already
above the hard limit and no action brings it back below within one control
step, so termination is checked and fires before a controller has held the
digester anywhere.  The scenario would therefore score a controller on its
opening move rather than on keeping a reactor near its limit for 60 days.

This sweeps a grid over both actions and reports the first-step VFA and
methane for every cell.  The grid is dense rather than exhaustive, so the
result is evidence over the sampled actions and not a proof over the
continuous action space.

    python3 scripts/recovery_first_step.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.adm1_gym_env_std import ADM1Env_Std as E

SCENARIO = 'acidified_recovery'
SEED = 7
# kg COD/m3 -> mg HAc-eq/L, the unit Section 3.2 reports VFA in.
MG = 1000 / 1.0667
# The hard band on total VFA that ends an episode.
VFA_HARD = 1.600
# Grid over the two actions: feed rate x feed-strength multiplier.
N_Q, N_M = 9, 7


def total_vfa(env):
    return env._calculate_total_vfa(env.current_state)


def main():
    probe = E(SCENARIO, obs_mode='scada', step_size=1.0)
    lo, hi = probe.action_space.low, probe.action_space.high
    qs = np.linspace(lo[0], hi[0], N_Q)
    ms = np.linspace(lo[1], hi[1], N_M)

    probe.reset(seed=SEED)
    v0 = total_vfa(probe)
    print(f'{SCENARIO}: initial total VFA {v0 * MG:.0f} mg HAc-eq/L '
          f'({v0:.3f} kg COD/m3) against a hard limit of '
          f'{VFA_HARD * MG:.0f} ({VFA_HARD:.3f})')
    print(f'initial VFA is {"above" if v0 > VFA_HARD else "below"} the limit\n')

    rows = []
    for q in qs:
        for m in ms:
            env = E(SCENARIO, obs_mode='scada', step_size=1.0)
            env.reset(seed=SEED)
            _, _, te, tr, _ = env.step(np.array([q, m], dtype=np.float32))
            rows.append((q, m, total_vfa(env) * MG, env.q_ch4, te or tr))

    vfas = [v for *_, v, _, _ in ((r[0], r[1], r[2], r[3], r[4]) for r in rows)]
    vfas = [r[2] for r in rows]
    ch4 = [r[3] for r in rows]
    ended = [r for r in rows if r[4]]
    above = [r for r in rows if r[2] > VFA_HARD * MG]

    print(f'{len(rows)} first-step values over a {N_Q}x{N_M} grid')
    print(f'  all above the hard limit: {len(above) == len(rows)} '
          f'({len(above)}/{len(rows)})')
    lowest = min(rows, key=lambda r: r[2])
    print(f'  lowest first-step VFA {lowest[2]:.0f} mg/L at feed '
          f'{lowest[0]:.0f} m3/d and strength {lowest[1]:.2f}')
    print(f'  episodes terminating on that one step: {len(ended)}/{len(rows)}')
    print(f'  first-step methane spans {min(ch4):.0f} to {max(ch4):.0f} m3/d, '
          f'so the actions do differ over the step')


if __name__ == '__main__':
    main()

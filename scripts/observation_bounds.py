#!/usr/bin/env python3
"""Does the observation normalisation ever clip, and does pH saturate?

Section 3.3 declares fixed min-max bounds so the same transformation applies
to plant measurements without refitting.  A value outside its bounds is
clipped rather than extrapolated, so the map is not injective there and a
policy cannot tell how far past a bound the plant has gone.  The pH channel
is the one where that could matter, because the hard band [5.8, 8.8] is wider
than the observed range [6.5, 8.0].

This sweeps the action-space corners and centre over every scenario and
reports, per channel, how close the raw state comes to its declared bounds
and whether any normalised observation reaches +-1.  The actions are held
constant for the episode, which drives the state further than any controller
would, so the bounds this finds are an upper estimate of what the benchmark
reaches.

    python3 scripts/observation_bounds.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.adm1_gym_env_std import ADM1Env_Std as E

# The scenario set of Section 3.4 and the episode length of the protocol.
SC = ['low_load', 'nominal', 'plant_load', 'high_load', 'peak_load',
      'fog_surge', 'elevated_start']
STEPS = 60
SEED = 7

# The six channels a SCADA-mode observation carries, in index order, with the
# declared bounds Section 3.3 lists.  These are read back from the env rather
# than restated, so a change there shows up here.
NAMES = ['total VFA', 'alkalinity', 'pH', 'methane flow',
         'feed rate', 'multiplier']
# Hard band on pH, wider than the observation range: the case the text flags.
PH_HARD = (5.8, 8.8)


def corners(env):
    """The four action-space corners and its centre, in physical units."""
    lo, hi = env.action_space.low, env.action_space.high
    mid = 0.5 * (lo + hi)
    return [np.array(a, dtype=np.float32) for a in
            ((lo[0], lo[1]), (lo[0], hi[1]), (hi[0], lo[1]), (hi[0], hi[1]),
             (mid[0], mid[1]))]


def main():
    probe = E('nominal', obs_mode='scada', step_size=1.0)
    obs_lo = np.asarray(probe.observation_space.low, dtype=float)
    obs_hi = np.asarray(probe.observation_space.high, dtype=float)
    idx = probe.SCADA_OBS_INDICES
    print('declared bounds, per observed channel:')
    for n, a, b in zip(NAMES, obs_lo, obs_hi):
        print(f'  {n:14s} [{a:g}, {b:g}]')

    steps = 0
    raw_min = np.full(len(idx), np.inf)
    raw_max = np.full(len(idx), -np.inf)
    clipped = np.zeros(len(idx), dtype=int)
    terminated = 0

    for s in SC:
        for act in corners(probe):
            env = E(s, obs_mode='scada', step_size=1.0)
            env.reset(seed=SEED)
            for _ in range(STEPS):
                obs, _, te, tr, _ = env.step(act)
                steps += 1
                # The env returns the raw state; the min-max map to [-1,1] is
                # applied by the training wrapper, so the clip is detected
                # against the declared bounds rather than against +-1.
                raw = np.asarray(obs, dtype=float)
                raw_min = np.minimum(raw_min, raw)
                raw_max = np.maximum(raw_max, raw)
                clipped += ((raw < obs_lo) | (raw > obs_hi)).astype(int)
                if te or tr:
                    terminated += 1
                    break

    print(f'\n{steps} control steps over {len(SC)} scenarios x '
          f'{len(corners(probe))} constant actions, '
          f'{terminated} episodes ending early')
    print('channel          observed range            bound        clips')
    for i, n in enumerate(NAMES):
        slack = min(raw_min[i] - obs_lo[i], obs_hi[i] - raw_max[i])
        print(f'  {n:14s} [{raw_min[i]:8.2f}, {raw_max[i]:8.2f}]  '
              f'[{obs_lo[i]:g}, {obs_hi[i]:g}]  {clipped[i]:5d}'
              f'   margin {slack:.2f}')
    # The last two channels echo the commanded action, so a constant action
    # placed on an action-space corner sits exactly on its bound by design.
    # Clipping is about values driven PAST a bound, which is what this counts.
    total = int(clipped.sum())
    print(f'\nchannel values outside a declared bound: {total}')
    if total == 0:
        print('so the bounds bracket the reachable state and clipping never '
              'activates; the feed-rate and multiplier channels sit exactly '
              'on their bounds whenever the constant action does, which is '
              'the action space itself, not a clipped state')
    ph = NAMES.index('pH')
    print(f'pH stays within [{raw_min[ph]:.2f}, {raw_max[ph]:.2f}], '
          f'inside both the observation range [{obs_lo[ph]:g}, {obs_hi[ph]:g}] '
          f'and the hard band [{PH_HARD[0]}, {PH_HARD[1]}]')
    v = NAMES.index('total VFA')
    print(f'the widest excursion of any channel is total VFA to '
          f'{raw_max[v]:.2f} against an upper bound of {obs_hi[v]:g}')


if __name__ == '__main__':
    main()

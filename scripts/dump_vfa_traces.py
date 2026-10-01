#!/usr/bin/env python3
"""Record the VFA a learned policy and a matched PI loop actually hold.

Section 9 compares a constrained policy against the PI setpoint that reaches
the same violation rate, and finds the learned policy sitting further from the
limit rather than closer to it. The per-run records keep only summaries, so
this re-runs both controllers under the evaluation protocol of
``eval_baseline_ext.py`` and writes every VFA sample, which is what a
distribution figure needs.

    python3 scripts/dump_vfa_traces.py <policy-run-dir> <out.json>
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.adm1_gym_env_std import ADM1Env_Std as E
from training.baselines import PIFeed

SC = ['low_load', 'nominal', 'plant_load', 'high_load', 'peak_load',
      'fog_surge', 'elevated_start']
VFA_SOFT = 0.320
MG = 1000 / 1.0667
Q_LO, Q_HI = 41.0, 159.0
STEPS = 200
SEED = 7

# The pair Section 9 compares: the PI setpoint whose violation rate matches
# the constrained policy's, and that policy at a cost limit of 3.0.
PI_SETPOINT, PI_KC, PI_TAU, PI_MULT = 225.0, 0.10, 20.0, 1.3


def _vfa(state):
    return sum(state[k] for k in ('S_va', 'S_bu', 'S_pro', 'S_ac'))


def trace_pi():
    """VFA under the matched PI loop, exactly as eval_baseline_ext runs it."""
    C = PIFeed(PI_SETPOINT, Q_LO, Q_HI, Kc=PI_KC, tau_i=PI_TAU, mult=PI_MULT)
    V = []
    for s in SC:
        e = E(s, obs_mode='scada', step_size=1.0)
        e.reset(seed=SEED)
        C.reset()
        q, mult = Q_HI, PI_MULT
        for _ in range(STEPS):
            stt = e.solver.state
            q, mult = C.act(_vfa(stt) * MG, stt['S_IC'],
                            -np.log10(max(stt['S_H_ion'], 1e-14)),
                            e.solver.q_ch4, q, mult)
            _, _, te, tr, _ = e.step(np.array([q, mult], dtype=np.float32))
            V.append(_vfa(e.solver.state) * MG)
            if te or tr:
                break
    return V


def trace_policy(run_dir):
    """VFA under the trained constrained policy, as eval_cmdp.py replays it."""
    import glob

    import torch
    from omnisafe.common.normalizer import Normalizer
    from omnisafe.models.actor import ActorBuilder

    from env.normalized_wrapper import NormalizedADM1Env
    from training.omnisafe_env import CMDP_REWARD

    cfg = json.load(open(os.path.join(run_dir, 'config.json')))
    ck = sorted(glob.glob(os.path.join(run_dir, 'torch_save', 'epoch-*.pt')),
                key=lambda p: int(p.split('epoch-')[1].split('.pt')[0]))
    params = torch.load(ck[-1], map_location='cpu')

    probe = NormalizedADM1Env(E('nominal', reward_config=CMDP_REWARD,
                                obs_mode='scada', step_size=1.0))
    mc = cfg['model_cfgs']
    actor = ActorBuilder(
        obs_space=probe.observation_space, act_space=probe.action_space,
        hidden_sizes=mc['actor']['hidden_sizes'],
        activation=mc['actor']['activation'],
        weight_initialization_mode=mc['weight_initialization_mode'],
    ).build_actor(mc['actor_type'])
    actor.load_state_dict(params['pi'])
    actor.eval()
    norm = None
    if 'obs_normalizer' in params:
        norm = Normalizer(shape=probe.observation_space.shape, clip=5)
        norm.load_state_dict(params['obs_normalizer'])

    V = []
    for s in SC:
        env = NormalizedADM1Env(E(s, reward_config=CMDP_REWARD,
                                  obs_mode='scada', step_size=1.0))
        obs, _ = env.reset(seed=SEED)
        for _ in range(STEPS):
            x = torch.as_tensor(obs, dtype=torch.float32)
            if norm is not None:
                x = norm.normalize(x)
            with torch.no_grad():
                act = actor.predict(x, deterministic=True).numpy()
            obs, _, te, tr, _ = env.step(act)
            V.append(_vfa(env.env.solver.state) * MG)
            if te or tr:
                break
    return V


def summarise(name, V):
    lim = VFA_SOFT * MG
    band = [v for v in V if 210 <= v <= lim]
    print(f'{name}: n={len(V)}  median {st.median(V):.1f} mg/L  '
          f'CV {st.pstdev(V) / st.mean(V):.2f}  '
          f'above {lim:.0f}: {100 * sum(1 for v in V if v > lim) / len(V):.2f} %  '
          f'in 210-{lim:.0f} band: {100 * len(band) / len(V):.1f} %')


def main():
    run_dir, out_path = sys.argv[1], sys.argv[2]
    pi = trace_pi()
    summarise('PI ', pi)
    pol = trace_policy(run_dir)
    summarise('RL ', pol)
    json.dump({'pi': pi, 'policy': pol,
               'pi_config': {'setpoint': PI_SETPOINT, 'Kc': PI_KC,
                             'tau_i': PI_TAU, 'mult': PI_MULT},
               'policy_run': run_dir},
              open(out_path, 'w'))
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()

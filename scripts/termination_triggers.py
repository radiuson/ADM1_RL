#!/usr/bin/env python3
"""Which hard predicate ends each early-terminating constrained run?

Three predicates end an episode: total VFA above the hard band, pH outside
[5.8, 8.8], or free ammonia above 0.01 kmol/m3.  Section 5.3 scores a
terminated run on the fixed horizon by counting every step after the breach
as an excursion, which reads as the cost indicator of Eq. (7) only where VFA
is what ended it.  The evaluation records store VFA but not pH or ammonia, so
the trigger is recovered by replaying the scenario from the saved policy.

Needs the trained weights, which the release does not carry; run it against a
local training directory.

    python3 scripts/termination_triggers.py models_cmdp/<FAMILY>/<run>/ <scenario>
"""
from __future__ import annotations

import glob
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from scripts.treatment_metrics import E, NormalizedADM1Env

torch.set_num_threads(1)


def trigger(run_dir: str, scenario: str) -> dict:
    """Replay one scenario and report the state at the terminating step."""
    from omnisafe.common.normalizer import Normalizer
    from omnisafe.models.actor import ActorBuilder

    from training.omnisafe_env import CMDP_REWARD

    run_dir = run_dir.rstrip('/')
    cfg = json.load(open(os.path.join(run_dir, 'config.json')))
    ck = sorted(glob.glob(os.path.join(run_dir, 'torch_save', 'epoch-*.pt')),
                key=lambda q: int(q.split('epoch-')[1].split('.pt')[0]))
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

    env = NormalizedADM1Env(E(scenario, reward_config=CMDP_REWARD,
                              obs_mode='scada', step_size=1.0))
    obs, _ = env.reset(seed=7)
    base = env.env
    for k in range(60):
        x = torch.as_tensor(obs, dtype=torch.float32)
        if norm is not None:
            x = norm.normalize(x)
        with torch.no_grad():
            action = actor.predict(x, deterministic=True).numpy()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            s = base.current_state
            pH = s.get('pH', 7.0)
            nh3 = s.get('S_nh3', 0.0)
            vfa = base._calculate_total_vfa(s)
            return {
                'steps': k + 1,
                'terminated': bool(terminated),
                'pH': pH,
                'pH_fired': pH < base.PH_HARD_LO or pH > base.PH_HARD_HI,
                'S_nh3': nh3,
                'nh3_fired': nh3 > base.NH3_HARD_KMOL_M3,
                'vfa': vfa,
                'vfa_fired': vfa > base.VFA_HARD_KGCOD_M3,
                'vfa_over_soft': vfa > base.VFA_SOFT_KGCOD_M3,
            }
    return {'steps': 60, 'terminated': False}


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        return
    run_dir, scenario = sys.argv[1], sys.argv[2]
    r = trigger(run_dir, scenario)
    print(f'{scenario}: ended after step {r["steps"]} '
          f'(terminated={r.get("terminated")})')
    if r['steps'] < 60 or r.get('terminated'):
        for name, value, fired in (
            ('pH', r['pH'], r['pH_fired']),
            ('S_nh3', r['S_nh3'], r['nh3_fired']),
            ('total VFA', r['vfa'], r['vfa_fired']),
        ):
            print(f'  {name:10s} {value:10.6f}  '
                  f'{"FIRED" if fired else "within band"}')
        if r['nh3_fired'] and not r['vfa_fired']:
            print(f'  total VFA is {"above" if r["vfa_over_soft"] else "below"}'
                  ' the soft limit, so the backfilled steps are a failure'
                  ' charge and not VFA excursions')


if __name__ == '__main__':
    main()

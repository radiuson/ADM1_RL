#!/usr/bin/env python3
"""Evaluate one OmniSafe constrained policy on the scenario set.

Usage: eval_cmdp.py <run_dir> <out.json>    (run in the adm1_safe env)

run_dir is an OmniSafe seed directory containing config.json and torch_save/.
The latest checkpoint is used; a run whose only checkpoint is epoch-0 has not
been trained and should be skipped by the caller.
"""
import os, sys, json, glob
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
import numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch; torch.set_num_threads(1)
from omnisafe.models.actor import ActorBuilder
from omnisafe.common.normalizer import Normalizer
from env.adm1_gym_env_std import ADM1Env_Std as E
from env.normalized_wrapper import NormalizedADM1Env
from training.omnisafe_env import CMDP_REWARD

# Seven scenarios.  'acidified_recovery' is excluded: it starts at a total VFA
# of 2271 mg/L, above the 1600 mg/L hard limit, and the environment treats any
# state above that limit as a catastrophic failure that terminates the episode.
# No action brings the state below the limit within one step -- the best of the
# action-space corners reaches 1654 mg/L -- so every policy terminates on step 1
# with an identical one-step record, and the scenario cannot separate
# controllers.
SC = ['low_load', 'nominal', 'plant_load', 'high_load', 'peak_load',
      'fog_surge', 'elevated_start']
VFA_SOFT = 0.320
VFA_HARD = 1.600   # 1500 mg/L as acetic acid, the inhibition-onset level
MG = 1000 / 1.0667

def main(run_dir, out_path):
    cfg = json.load(open(os.path.join(run_dir, 'config.json')))
    ck = sorted(glob.glob(os.path.join(run_dir, 'torch_save', 'epoch-*.pt')),
                key=lambda p: int(p.split('epoch-')[1].split('.pt')[0]))
    params = torch.load(ck[-1], map_location='cpu')

    probe = NormalizedADM1Env(E('nominal', reward_config=CMDP_REWARD,
                                obs_mode='scada', step_size=1.0))
    mc = cfg['model_cfgs']
    actor = ActorBuilder(obs_space=probe.observation_space,
                         act_space=probe.action_space,
                         hidden_sizes=mc['actor']['hidden_sizes'],
                         activation=mc['actor']['activation'],
                         weight_initialization_mode=mc['weight_initialization_mode']
                         ).build_actor(mc['actor_type'])
    actor.load_state_dict(params['pi']); actor.eval()
    norm = None
    if 'obs_normalizer' in params:
        norm = Normalizer(shape=probe.observation_space.shape, clip=5)
        norm.load_state_dict(params['obs_normalizer'])

    V, C = [], []
    for s in SC:
        env = NormalizedADM1Env(E(s, reward_config=CMDP_REWARD,
                                  obs_mode='scada', step_size=1.0))
        obs, _ = env.reset(seed=7)
        for _ in range(200):
            x = torch.as_tensor(obs, dtype=torch.float32)
            if norm is not None:
                x = norm.normalize(x)
            with torch.no_grad():
                act = actor.predict(x, deterministic=True).numpy()
            obs, r, term, trunc, _ = env.step(act)
            st = env.env.solver.state
            V.append(sum(st[k] for k in ('S_va', 'S_bu', 'S_pro', 'S_ac')))
            C.append(env.env.solver.q_ch4)
            if term or trunc:
                break
    V, C = np.array(V), np.array(C)
    cl = (cfg.get('lagrange_cfgs', {}).get('cost_limit')
          or cfg.get('algo_cfgs', {}).get('cost_limit'))
    json.dump({'algo': cfg['algo'], 'seed': cfg['seed'], 'cost_limit': cl,
               'run_dir': run_dir, 'ch4': float(C.mean()),
               'viol': float(100 * np.mean(V > VFA_SOFT)),
               'viol_hard': float(100 * np.mean(V > VFA_HARD)),
               'vfa_max': float(V.max() * MG),
               'vfa_med': float(np.median(V) * MG)}, open(out_path, 'w'))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

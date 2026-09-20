#!/usr/bin/env python3
"""Evaluate one Stable-Baselines3 / sb3-contrib policy on the scenario set.

Usage: eval_sb3.py <model.zip> <out.json>

Lives in the repository rather than /tmp so it survives a restart; the previous
copy did not.
"""
import os, sys, json
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
import numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch; torch.set_num_threads(1)
from env.adm1_gym_env_std import ADM1Env_Std as E
from env.normalized_wrapper import NormalizedADM1Env
from env.history_wrapper import HistoryWrapper

# Seven scenarios.  'acidified_recovery' is excluded: it starts at a total VFA
# of 2271 mg/L, above the 1600 mg/L hard limit, and the environment treats any
# state above that limit as a catastrophic failure that terminates the episode.
# No action brings the state below the limit within one step -- the best of the
# action-space corners reaches 1654 mg/L -- so every policy terminates on step 1
# with an identical one-step record, and the scenario cannot separate
# controllers.
SC = ['low_load', 'nominal', 'plant_load', 'high_load', 'peak_load',
      'fog_surge', 'elevated_start']
from evaluation.severity import scenario_metrics, summarise

# The soft limit is out of reach in these two even at maximum feed and
# strength, so they contribute structural zeros to any pooled rate.
UNREACHABLE = ('low_load', 'nominal')
REACHABLE = tuple(x for x in SC if x not in UNREACHABLE)

VFA_SOFT = 0.320
VFA_HARD = 1.600   # 1500 mg/L as acetic acid, the inhibition-onset level
MG = 1000 / 1.0667

def main(model_path, out_path):
    meta = json.load(open(os.path.join(os.path.dirname(model_path), 'run_meta.json')))
    algo = meta.get('algo', 'SAC').lower()
    import stable_baselines3 as sb3, sb3_contrib as sbc
    MAP = {'sac': sb3.SAC, 'ppo': sb3.PPO, 'a2c': sb3.A2C, 'ddpg': sb3.DDPG,
           'td3': sb3.TD3, 'tqc': sbc.TQC, 'trpo': sbc.TRPO,
           'recurrentppo': sbc.RecurrentPPO, 'crossq': sbc.CrossQ, 'ars': sbc.ARS}
    model = MAP[algo].load(model_path, device='cpu')
    hist = 5 if '_h5_' in model_path else 1

    V, C = [], []
    per_scen = {}
    for s in SC:
        env = NormalizedADM1Env(E(s, obs_mode='scada', step_size=1.0))
        if hist > 1:
            env = HistoryWrapper(env, hist)
        obs, _ = env.reset(seed=7)
        base = env.env.env if hist > 1 else env.env
        state = None
        Vs = []
        for _ in range(200):
            if algo == 'recurrentppo':
                act, state = model.predict(obs, state=state, deterministic=True)
            else:
                act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(act)
            st = base.solver.state
            vfa = sum(st[k] for k in ('S_va', 'S_bu', 'S_pro', 'S_ac'))
            Vs.append(vfa)
            V.append(vfa)
            C.append(base.solver.q_ch4)
            if term or trunc:
                break
        per_scen[s] = scenario_metrics(Vs)
    V, C = np.array(V), np.array(C)
    sev = summarise(per_scen, REACHABLE)
    json.dump({'model': model_path, 'algo': algo, 'w': meta['reward_config'],
               'seed': meta['seed'], 'steps': meta['total_timesteps'],
               'ch4': float(C.mean()), 'viol': float(100 * np.mean(V > VFA_SOFT)),
               'viol_hard': float(100 * np.mean(V > VFA_HARD)),
               'vfa_max': float(V.max() * MG),
               'vfa_med': float(np.median(V) * MG),
               **{k: v for k, v in sev.items() if k not in ('viol', 'viol_hard', 'vfa_max')}},
              open(out_path, 'w'))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

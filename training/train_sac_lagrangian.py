"""Train SAC under a stated violation budget instead of a penalty weight."""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_num_threads(1)

from stable_baselines3 import SAC                                    # noqa: E402
from env.adm1_gym_env_std import ADM1Env_Std, STD_SCENARIOS          # noqa: E402
from env.normalized_wrapper import NormalizedADM1Env                 # noqa: E402
from training.reward_configs import REWARD_CONFIGS                   # noqa: E402
from training.sac_lagrangian import (CostWrapper, LagrangianCallback,  # noqa: E402
                                     PIDLagrangianCallback)
from training.train_sac_std_cur import (MultiScenarioEnv, PAPER_SCENARIOS,
                                        SAC_HYPERPARAMS)             # noqa: E402


class LagrangianMultiScenario(MultiScenarioEnv):
    """Multi-scenario curriculum whose episodes carry a safety cost."""

    def __init__(self, *args, limit, **kwargs):
        self.limit = limit
        self._cost = None
        super().__init__(*args, **kwargs)

    def _new_env(self):
        super()._new_env()
        self._cost = CostWrapper(self._env, self.limit)
        self._env = self._cost


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--budget', type=float, required=True,
                   help='allowed share of steps above the soft VFA limit')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--timesteps', type=int, default=150_000)
    p.add_argument('--obs-mode', type=str, default='scada',
                   choices=['scada', 'trend', 'full'])
    p.add_argument('--step-size', type=float, default=1.0)
    p.add_argument('--output-dir', type=str, default='models_lag')
    p.add_argument('--lambda-lr', type=float, default=0.5)
    p.add_argument('--dual', type=str, default='pid', choices=['ascent', 'pid'],
                   help='plain dual ascent, or with proportional and derivative action')
    p.add_argument('--verbose', type=int, default=0)
    a = p.parse_args()

    limit = ADM1Env_Std.VFA_SOFT_KGCOD_M3
    # The production term alone drives the policy; the safety term enters only
    # through the multiplier, so the reward configuration carries no VFA weight.
    rc = dict(REWARD_CONFIGS['lw0p1'])
    rc['vfa_penalty_scale'] = 0.0
    rc['vfa_constant_penalty'] = 0.0

    env = LagrangianMultiScenario(
        PAPER_SCENARIOS, reward_config=rc, obs_mode=a.obs_mode,
        seed=a.seed, normalize=True, step_size=a.step_size, limit=limit)
    env.reset(seed=a.seed)

    model = SAC('MlpPolicy', env, seed=a.seed, device='cpu',
                verbose=0, **SAC_HYPERPARAMS)

    if a.dual == 'pid':
        cb = PIDLagrangianCallback(env._cost, budget=a.budget, verbose=a.verbose)
    else:
        cb = LagrangianCallback(env._cost, budget=a.budget, lr=a.lambda_lr,
                                verbose=a.verbose)
    # The wrapper is rebuilt on every episode; keep the callback pointed at the
    # live one and carry the multiplier across.
    _orig = env._new_env

    def _new_env():
        prev = getattr(env, '_cost', None)
        lam = prev.lambda_ if prev is not None else 0.0
        hist = prev.episode_costs if prev is not None else None
        _orig()
        env._cost.lambda_ = lam
        if hist is not None:
            env._cost.episode_costs.extend(hist)
        cb.wrapper = env._cost
    env._new_env = _new_env

    model.learn(total_timesteps=a.timesteps, callback=cb, progress_bar=False)

    tag = f'_{a.obs_mode}' if a.obs_mode != 'scada' else ''
    dtag = '' if a.dual == 'pid' else '_ascent'
    out = (Path(a.output_dir) /
           f'sac_lag_b{a.budget:g}_seed{a.seed}{tag}{dtag}_norm_daily')
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / 'final_model.zip')
    json.dump({'budget': a.budget, 'seed': a.seed, 'obs_mode': a.obs_mode,
               'dual': a.dual,
               'limit_kgcod_m3': limit, 'timesteps': a.timesteps,
               'lambda_final': env._cost.lambda_,
               'lambda_history': cb.history[-40:]},
              open(out / 'run_meta.json', 'w'), indent=2)
    print(f"done budget={a.budget} seed={a.seed} obs={a.obs_mode} "
          f"lambda={env._cost.lambda_:.3f} -> {out}")


if __name__ == '__main__':
    main()

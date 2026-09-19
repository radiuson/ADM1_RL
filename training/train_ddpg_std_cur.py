#!/usr/bin/env python3
"""
SAC Training on Standard ADM1 — Multi-Scenario Curriculum
===========================================================

Trains SAC on ADM1Env_Std with uniform-random scenario sampling over all
six paper scenarios.  This is the standard training recipe for the paper's
SF-SAC, ST-SAC, and Balanced-RL agents.

Scenarios (paper group):
    nominal, high_load, low_load, shock_load, high_load_real, pre_stressed

Usage:
    # SF-SAC  (Safety-First, 3 seeds)
    python training/train_sac_std_cur.py --reward-config safety_first   --seed 42
    python training/train_sac_std_cur.py --reward-config safety_first   --seed 123
    python training/train_sac_std_cur.py --reward-config safety_first   --seed 456

    # ST-SAC  (Safety-Target)
    python training/train_sac_std_cur.py --reward-config safety_target  --seed 42
    ...

    # Balanced-RL (no safety reward, weak quadratic penalty)
    python training/train_sac_std_cur.py --reward-config balanced       --seed 42
    ...

Output layout:
    models_std_paper/
      ddpg_std_cur_<reward>_seed<seed>/
        best_model/best_model.zip
        final_model.zip
        run_meta.json
"""

import argparse
import json
import random
import time
from pathlib import Path

import os
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import numpy as np
import torch
torch.set_num_threads(1)

import gymnasium as gym
from gymnasium import spaces

from env.adm1_gym_env_std import ADM1Env_Std, STD_SCENARIOS
from env.normalized_wrapper import NormalizedADM1Env
from training.reward_configs import REWARD_CONFIGS

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback
)


# ── Paper scenario set ────────────────────────────────────────────────────────

PAPER_SCENARIOS = ['nominal', 'high_load', 'low_load', 'shock_load',
                   'high_load_real', 'pre_stressed', 'lipid_overload']

# ── Hyperparameters ───────────────────────────────────────────────────────────

DDPG_HYPERPARAMS = {
    'learning_rate':   3e-4,
    'buffer_size':     1_000_000,
    'batch_size':      256,
    'tau':             0.005,
    'gamma':           0.99,
    'learning_starts': 10_000,
    'train_freq':      1,
    'gradient_steps':  1,
    'policy_kwargs':   dict(net_arch=[256, 256]),
}

ENT_COEF_MIN      = 0.01
ENT_COEF_MAX      = 0.5    # reduced from 5.0 — allows policy to converge
DEFAULT_TIMESTEPS = 1_000_000


# ── Multi-scenario training wrapper ──────────────────────────────────────────

class MultiScenarioEnv(gym.Env):
    """
    Gymnasium wrapper that samples a new scenario uniformly at random on
    every episode reset.  The action and observation spaces match
    ADM1Env_Std (2-dim action, 12-dim or 5-dim obs).
    """

    def __init__(self, scenarios, reward_config, obs_mode='scada', seed=None, normalize=False, step_size=0.01041667):
        super().__init__()
        self.scenarios     = scenarios
        self.reward_config = reward_config
        self.obs_mode      = obs_mode
        self.normalize     = normalize
        self.step_size     = step_size
        self._rng          = np.random.default_rng(seed)

        # Build a reference env to define spaces
        _ref = ADM1Env_Std(scenarios[0], reward_config=reward_config, obs_mode=obs_mode,
                           step_size=step_size)
        if normalize:
            _ref = NormalizedADM1Env(_ref)
        self.observation_space = _ref.observation_space
        self.action_space      = _ref.action_space
        _ref.close()

        self._env = None

    def _new_env(self):
        if self._env is not None:
            self._env.close()
        scenario = self._rng.choice(self.scenarios)
        self._env = ADM1Env_Std(
            scenario_name=scenario,
            reward_config=self.reward_config,
            obs_mode=self.obs_mode,
            step_size=self.step_size,
        )
        if self.normalize:
            self._env = NormalizedADM1Env(self._env)

    def reset(self, seed=None, options=None):
        self._new_env()
        obs, info = self._env.reset(seed=seed)
        return obs, info

    def step(self, action):
        return self._env.step(action)

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None


# ── Entropy clamp callback ────────────────────────────────────────────────────

def _make_ent_clamp_callback(max_ent=ENT_COEF_MAX, min_ent=ENT_COEF_MIN):
    _max_log = torch.log(torch.tensor(max_ent, dtype=torch.float32)).item()
    _min_log = torch.log(torch.tensor(min_ent, dtype=torch.float32)).item()

    class _Clamp(BaseCallback):
        def _on_step(self):
            if hasattr(self.model, 'log_ent_coef') and self.model.log_ent_coef is not None:
                with torch.no_grad():
                    self.model.log_ent_coef.data.clamp_(_min_log, _max_log)
            return True

    return _Clamp(verbose=0)


# ── Main training function ────────────────────────────────────────────────────

def train(
    reward_config_name: str = 'safety_first',
    normalize: bool         = False,
    step_size: float        = 0.01041667,
    seed: int               = 42,
    obs_mode: str           = 'scada',
    total_timesteps: int    = DEFAULT_TIMESTEPS,
    output_dir: str         = 'models_std_paper',
    eval_scenario: str      = 'nominal',
    eval_freq: int          = 10_000,
    n_eval_episodes: int    = 5,
    device: str             = 'auto',
    verbose: int            = 1,
) -> Path:
    """
    Train SAC on multi-scenario ADM1Env_Std curriculum.

    Returns:
        Path to the run directory.
    """
    if reward_config_name not in REWARD_CONFIGS:
        raise ValueError(f"Unknown reward config '{reward_config_name}'.")

    obs_tag  = f'_{obs_mode}' if obs_mode != 'scada' else ''
    run_name = f'ddpg_std_cur_{reward_config_name}_seed{seed}{obs_tag}' + ('_norm' if normalize else '') + ('_daily' if step_size >= 0.99 else '')
    run_dir  = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'best_model').mkdir(exist_ok=True)
    (run_dir / 'checkpoints').mkdir(exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  DDPG — Standard ADM1, Multi-Scenario Curriculum")
    print(f"  Reward:     {reward_config_name}")
    print(f"  Obs mode:   {obs_mode}")
    print(f"  Scenarios:  {PAPER_SCENARIOS}")
    print(f"  Seed:       {seed}")
    print(f"  Timesteps:  {total_timesteps:,}")
    print(f"  Output:     {run_dir}")
    print(f"{'='*65}")

    reward_config = REWARD_CONFIGS[reward_config_name]

    train_env = Monitor(MultiScenarioEnv(
        PAPER_SCENARIOS, reward_config=reward_config,
        obs_mode=obs_mode, seed=seed, normalize=normalize,
        step_size=step_size,
    ))

    _ev = ADM1Env_Std(
        scenario_name=eval_scenario,
        reward_config=reward_config,
        obs_mode=obs_mode,
        step_size=step_size,
    )
    eval_env = Monitor(NormalizedADM1Env(_ev) if normalize else _ev)
    eval_env.reset(seed=seed + 1)

    n_act = train_env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_act),
                                     sigma=0.1 * np.ones(n_act))

    model = DDPG(
        'MlpPolicy',
        train_env,
        verbose=verbose,
        seed=seed,
        device=device,
        tensorboard_log=str(run_dir / 'tensorboard'),
        action_noise=action_noise,
        **DDPG_HYPERPARAMS,
    )

    callbacks = CallbackList([
        EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / 'best_model'),
            log_path=str(run_dir / 'eval'),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
        ),
        CheckpointCallback(
            save_freq=100_000,
            save_path=str(run_dir / 'checkpoints'),
            name_prefix='sac_model',
        ),
    ])

    t0 = time.time()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=(verbose >= 1),
        )
    except KeyboardInterrupt:
        print("\n  [Interrupted — saving current model]")

    elapsed = time.time() - t0
    model.save(str(run_dir / 'final_model'))

    meta = {
        'algo':            'DDPG',
        'env':             'ADM1Env_Std',
        'scenarios':       PAPER_SCENARIOS,
        'reward_config':   reward_config_name,
        'normalize':       normalize,
        'step_size_days':  step_size,
        'obs_mode':        obs_mode,
        'seed':            seed,
        'total_timesteps': total_timesteps,
        'elapsed_min':     round(elapsed / 60, 1),
        'hyperparams':     {k: str(v) if isinstance(v, dict) else v
                            for k, v in DDPG_HYPERPARAMS.items()},
    }
    with open(run_dir / 'run_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    train_env.close()
    eval_env.close()
    print(f"\n  Done in {elapsed/60:.1f} min → {run_dir}")
    return run_dir


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train SAC on standard ADM1 with 6-scenario curriculum',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--reward-config', type=str, default='safety_first',
                        help='Reward configuration: sf-sac=safety_first, '
                             'st-sac=safety_target, baseline-rl=balanced')
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--timesteps',  type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument('--output-dir', type=str, default='models_std_paper')
    parser.add_argument('--obs-mode',   type=str, default='scada',
                        choices=['scada', 'full'])
    parser.add_argument('--device',     type=str, default='auto')
    parser.add_argument('--verbose',    type=int, default=1)
    parser.add_argument('--normalize', action='store_true',
                        help='Min-max normalise obs/action spaces to [-1,1].')
    parser.add_argument('--step-size', type=float, default=0.01041667,
                        help='Control interval in days (0.01041667=15min, 1.0=daily).')
    args = parser.parse_args()

    train(
        reward_config_name = args.reward_config,
        normalize          = args.normalize,
        step_size          = args.step_size,
        seed               = args.seed,
        obs_mode           = args.obs_mode,
        total_timesteps    = args.timesteps,
        output_dir         = args.output_dir,
        device             = args.device,
        verbose            = args.verbose,
    )


if __name__ == '__main__':
    main()

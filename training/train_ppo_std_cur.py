#!/usr/bin/env python3
"""
PPO Training on Standard ADM1 — Multi-Scenario Curriculum
==========================================================

Trains PPO on ADM1Env_Std with uniform-random scenario sampling over all
six paper scenarios.  PPO serves as the on-policy RL baseline alongside
the off-policy SAC agents (SF-SAC, ST-SAC, Balanced-RL).

Key differences from SAC (train_sac_std_cur.py):
  - On-policy: collects n_steps rollouts per update (no replay buffer)
  - Fixed entropy coefficient (not auto-tuned like SAC)
  - Same reward configs and obs_mode as SAC for fair comparison

Usage:
    python training/train_ppo_std_cur.py --reward-config safety_first --seed 42
    python training/train_ppo_std_cur.py --reward-config safety_target --seed 42
    python training/train_ppo_std_cur.py --reward-config balanced      --seed 42

Output layout:
    models_std_paper_v2/
      ppo_std_cur_<reward>_seed<seed>/
        best_model/best_model.zip
        final_model.zip
        run_meta.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import gymnasium as gym

from env.adm1_gym_env_std import ADM1Env_Std
from env.normalized_wrapper import NormalizedADM1Env
from training.reward_configs import REWARD_CONFIGS

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList
)


# ── Paper scenario set ────────────────────────────────────────────────────────

PAPER_SCENARIOS = ['nominal', 'high_load', 'low_load', 'shock_load',
                   'high_load_real', 'pre_stressed', 'lipid_overload']

# ── Hyperparameters ───────────────────────────────────────────────────────────

PPO_HYPERPARAMS = {
    'learning_rate':  3e-4,
    'n_steps':        2048,   # steps per rollout (≈0.7 episodes at 2880 steps/ep)
    'batch_size':     64,
    'n_epochs':       10,
    'gamma':          0.99,
    'gae_lambda':     0.95,
    'clip_range':     0.2,
    'ent_coef':       0.01,   # fixed; SAC uses auto-tuned ent_coef
    'vf_coef':        0.5,
    'max_grad_norm':  0.5,
    'policy_kwargs':  dict(net_arch=[256, 256]),
}

DEFAULT_TIMESTEPS = 1_000_000


# ── Multi-scenario training wrapper ──────────────────────────────────────────

class MultiScenarioEnv(gym.Env):
    """Samples a new scenario uniformly at random on every episode reset."""

    def __init__(self, scenarios, reward_config, obs_mode='scada', seed=None, normalize=False, step_size=0.01041667):
        super().__init__()
        self.scenarios     = scenarios
        self.reward_config = reward_config
        self.obs_mode      = obs_mode
        self.normalize     = normalize
        self.step_size     = step_size
        self._rng          = np.random.default_rng(seed)

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


# ── Main training function ────────────────────────────────────────────────────

def train(
    reward_config_name: str = 'safety_first',
    normalize: bool         = False,
    step_size: float        = 0.01041667,
    seed: int               = 42,
    obs_mode: str           = 'scada',
    total_timesteps: int    = DEFAULT_TIMESTEPS,
    output_dir: str         = 'models_std_paper_v2',
    eval_scenario: str      = 'nominal',
    eval_freq: int          = 10_000,
    n_eval_episodes: int    = 5,
    device: str             = 'auto',
    verbose: int            = 1,
) -> Path:
    """
    Train PPO on multi-scenario ADM1Env_Std curriculum.

    Returns:
        Path to the run directory.
    """
    if reward_config_name not in REWARD_CONFIGS:
        raise ValueError(f"Unknown reward config '{reward_config_name}'.")

    obs_tag  = f'_{obs_mode}' if obs_mode != 'scada' else ''
    run_name = f'ppo_std_cur_{reward_config_name}_seed{seed}{obs_tag}' + ('_norm' if normalize else '') + ('_daily' if step_size >= 0.99 else '')
    run_dir  = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'best_model').mkdir(exist_ok=True)
    (run_dir / 'checkpoints').mkdir(exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  PPO — Standard ADM1, Multi-Scenario Curriculum")
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

    model = PPO(
        'MlpPolicy',
        train_env,
        verbose=verbose,
        seed=seed,
        device=device,
        tensorboard_log=str(run_dir / 'tensorboard'),
        **PPO_HYPERPARAMS,
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
            name_prefix='ppo_model',
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
        'algo':            'PPO',
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
                            for k, v in PPO_HYPERPARAMS.items()},
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
        description='Train PPO on standard ADM1 with 6-scenario curriculum',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--reward-config', type=str, default='safety_first',
                        )
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--timesteps',  type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument('--output-dir', type=str, default='models_std_paper_v2')
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

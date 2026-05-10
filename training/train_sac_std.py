#!/usr/bin/env python3
"""
SAC Training on Standard (Non-Thermal) ADM1
============================================

Trains a Soft Actor-Critic agent on ADM1Env_Std — the standard ADM1 environment
with a 2-dimensional action space [q_ad, feed_mult] (no Q_HEX heat exchanger).
Used for the cross-model comparison experiments.

Differences vs train_sac.py (thermal env):
  - Uses ADM1Env_Std instead of ADM1Env_v2
  - Action space: 2-dim [q_ad, feed_mult]
  - Observation space: 11-dim full or 4-dim simple (no T_L_norm)
  - Only four scenarios: nominal, high_load, low_load, shock_load

Usage:
    # Single run (paper default)
    python training/train_sac_std.py --scenario nominal --seed 42

    # All four scenarios × 3 seeds (full std comparison)
    for scenario in nominal high_load low_load shock_load; do
        for seed in 42 123 456; do
            python training/train_sac_std.py --scenario $scenario --seed $seed
        done
    done
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from env.adm1_gym_env_std import ADM1Env_Std, STD_SCENARIOS
from training.reward_configs import REWARD_CONFIGS

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList
)


# ── Hyperparameters (same as thermal paper run) ───────────────────────────────

SAC_HYPERPARAMS = {
    'learning_rate':   3e-4,
    'buffer_size':     1_000_000,
    'batch_size':      256,
    'tau':             0.005,
    'gamma':           0.99,
    'learning_starts': 10_000,
    'train_freq':      1,
    'gradient_steps':  1,
    'ent_coef':        'auto',
    'policy_kwargs':   dict(net_arch=[256, 256]),
}

ENT_COEF_MIN        = 0.01
ENT_COEF_MAX        = 5.0
DEFAULT_TIMESTEPS   = 300_000


# ── Entropy clamp callback (identical to train_sac.py) ───────────────────────

def _make_ent_coef_clamp_callback(
    max_ent_coef: float = ENT_COEF_MAX,
    min_ent_coef: float = ENT_COEF_MIN,
):
    from stable_baselines3.common.callbacks import BaseCallback

    _max_log = torch.log(torch.tensor(max_ent_coef, dtype=torch.float32)).item()
    _min_log = torch.log(torch.tensor(min_ent_coef, dtype=torch.float32)).item()

    class _ClampCallback(BaseCallback):
        def _on_step(self) -> bool:
            if hasattr(self.model, 'log_ent_coef') and self.model.log_ent_coef is not None:
                with torch.no_grad():
                    self.model.log_ent_coef.data.clamp_(min=_min_log, max=_max_log)
            return True

    return _ClampCallback(verbose=0)


# ── Training function ─────────────────────────────────────────────────────────

def train_sac_std(
    scenario: str           = 'nominal',
    reward_config_name: str = 'safety_first',
    seed: int               = 42,
    total_timesteps: int    = DEFAULT_TIMESTEPS,
    output_dir: str         = 'models_std',
    obs_mode: str           = 'full',
    eval_freq: int          = 10_000,
    n_eval_episodes: int    = 5,
    device: str             = 'auto',
    verbose: int            = 1,
) -> Path:
    """
    Train SAC on a single std-ADM1 scenario.

    Returns:
        Path to the run directory containing the trained model.
    """
    if scenario not in STD_SCENARIOS:
        raise ValueError(
            f"Scenario '{scenario}' not in STD_SCENARIOS={STD_SCENARIOS}"
        )
    if reward_config_name not in REWARD_CONFIGS:
        raise ValueError(
            f"Unknown reward config '{reward_config_name}'. "
            f"Available: {list(REWARD_CONFIGS.keys())}"
        )

    obs_suffix = f'_{obs_mode}' if obs_mode != 'full' else ''
    run_name   = f'sac_std_{scenario}_{reward_config_name}_seed{seed}{obs_suffix}'
    run_dir    = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'best_model').mkdir(exist_ok=True)
    (run_dir / 'checkpoints').mkdir(exist_ok=True)
    (run_dir / 'eval').mkdir(exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  SAC Training — Standard ADM1")
    print(f"  Scenario:       {scenario}")
    print(f"  Reward config:  {reward_config_name}")
    print(f"  Obs mode:       {obs_mode}")
    print(f"  Seed:           {seed}")
    print(f"  Timesteps:      {total_timesteps:,}")
    print(f"  Output:         {run_dir}")
    print(f"{'='*65}")

    reward_config = REWARD_CONFIGS[reward_config_name]

    train_env = Monitor(ADM1Env_Std(
        scenario_name=scenario,
        reward_config=reward_config,
        obs_mode=obs_mode,
    ))
    train_env.reset(seed=seed)

    eval_env = Monitor(ADM1Env_Std(
        scenario_name=scenario,
        reward_config=reward_config,
        obs_mode=obs_mode,
    ))
    eval_env.reset(seed=seed + 1)

    model = SAC(
        'MlpPolicy',
        train_env,
        verbose=verbose,
        seed=seed,
        device=device,
        tensorboard_log=str(run_dir / 'tensorboard'),
        **SAC_HYPERPARAMS,
    )

    eval_cb  = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / 'best_model'),
        log_path=str(run_dir / 'eval'),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )
    ckpt_cb  = CheckpointCallback(
        save_freq=50_000,
        save_path=str(run_dir / 'checkpoints'),
        name_prefix='sac_model',
    )
    clamp_cb = _make_ent_coef_clamp_callback()

    t0 = time.time()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList([eval_cb, ckpt_cb, clamp_cb]),
            progress_bar=(verbose >= 1),
        )
    except KeyboardInterrupt:
        print("\n  [Training interrupted — saving current model]")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed / 60:.1f} min")

    model.save(str(run_dir / 'final_model'))

    meta = {
        'algo':            'sac',
        'env':             'ADM1Env_Std',
        'scenario':        scenario,
        'reward_config':   reward_config_name,
        'obs_mode':        obs_mode,
        'seed':            seed,
        'total_timesteps': total_timesteps,
        'elapsed_seconds': elapsed,
        'ent_coef_bounds': [ENT_COEF_MIN, ENT_COEF_MAX],
        'hyperparams':     {k: str(v) if isinstance(v, dict) else v
                            for k, v in SAC_HYPERPARAMS.items()},
    }
    with open(run_dir / 'run_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    train_env.close()
    eval_env.close()

    print(f"  Model saved → {run_dir}")
    return run_dir


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train SAC on standard (non-thermal) ADM1',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--scenario',      type=str, default='nominal',
                        choices=STD_SCENARIOS)
    parser.add_argument('--reward-config', type=str, default='safety_first')
    parser.add_argument('--seed',          type=int, default=42)
    parser.add_argument('--timesteps',     type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument('--output-dir',    type=str, default='models_std')
    parser.add_argument('--obs-mode',      type=str, default='full',
                        choices=['full', 'simple'])
    parser.add_argument('--device',        type=str, default='auto')
    args = parser.parse_args()

    train_sac_std(
        scenario           = args.scenario,
        reward_config_name = args.reward_config,
        seed               = args.seed,
        total_timesteps    = args.timesteps,
        output_dir         = args.output_dir,
        obs_mode           = args.obs_mode,
        device             = args.device,
    )


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
SAC Training Script for ADM1 Biogas Control
============================================

Trains a Soft Actor-Critic (SAC) agent on a single ADM1 scenario using the
safety-first reward configuration reported in the paper.

Key implementation detail — clamped auto-entropy tuning:
    SAC uses ent_coef='auto' (dual-variable entropy regularisation), but the
    entropy coefficient α is projected onto [α_min, α_max] after every gradient
    step via _make_ent_coef_clamp_callback.  This is equivalent to projected
    gradient descent on the dual variable and prevents:
      - α → 0 collapse (policy becomes deterministic too early, misses safe region)
      - α → ∞ explosion (NaN critic loss on violation-heavy episodes)
    Bounds used in the paper: α ∈ [0.01, 5.0].

Usage:
    # Train on nominal scenario (paper default)
    python train_sac.py

    # Train on a specific scenario
    python train_sac.py --scenario cold_winter --seed 42

    # Full paper run: all 6 scenarios × 3 seeds
    for scenario in nominal high_load low_load shock_load temperature_drop cold_winter; do
        for seed in 42 123 456; do
            python train_sac.py --scenario $scenario --seed $seed
        done
    done
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from env.adm1_gym_env import ADM1Env_v2
from training.reward_configs import REWARD_CONFIGS

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList


# ── Paper hyperparameters ─────────────────────────────────────────────────────

SAC_HYPERPARAMS = {
    'learning_rate':   3e-4,
    'buffer_size':     1_000_000,
    'batch_size':      256,
    'tau':             0.005,
    'gamma':           0.99,
    'learning_starts': 10_000,
    'train_freq':      1,
    'gradient_steps':  1,
    'ent_coef':        'auto',          # auto-tuned, clamped to [0.01, 5.0]
    'policy_kwargs':   dict(net_arch=[256, 256]),
}

# Entropy coefficient bounds (projected gradient descent on dual variable α)
ENT_COEF_MIN = 0.01
ENT_COEF_MAX = 5.0

# Default training length for this standalone script.
# Paper experiments use 300 000 steps (configured via experiment_config.yaml).
DEFAULT_TOTAL_TIMESTEPS = 300_000

# Paper scenarios
PAPER_SCENARIOS = [
    'nominal', 'high_load', 'low_load',
    'shock_load', 'temperature_drop', 'cold_winter',
]


# ── Entropy clamp callback ────────────────────────────────────────────────────

def _make_ent_coef_clamp_callback(
    max_ent_coef: float = ENT_COEF_MAX,
    min_ent_coef: float = ENT_COEF_MIN,
):
    """
    SB3 callback that clamps SAC log_ent_coef after every training step.

    Mathematical basis: projected gradient descent on the dual variable α.
    Equivalent to constrained dual optimisation:
        max  E[Q(s,a)] - α · (H_target - E[-log π])
        s.t. α_min ≤ α ≤ α_max

    This is strictly stronger than fixing ent_coef to a constant (which disables
    adaptation entirely) while preventing the NaN explosion that occurs with
    unconstrained auto-tuning when the policy collapses on violation-heavy episodes.

    Args:
        max_ent_coef: Upper bound on α (default 5.0).
        min_ent_coef: Lower bound on α (default 0.01).
    """
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

def train_sac(
    scenario: str = 'nominal',
    reward_config_name: str = 'safety_first',
    seed: int = 42,
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
    output_dir: str = 'models',
    obs_mode: str = 'full',
    eval_freq: int = 10_000,
    n_eval_episodes: int = 5,
    device: str = 'auto',
    verbose: int = 1,
) -> Path:
    """
    Train a SAC agent on a single ADM1 scenario.

    Args:
        scenario:           ADM1 scenario name (see env/scenarios.yaml).
        reward_config_name: Key into REWARD_CONFIGS (training/reward_configs.py).
        seed:               Random seed.
        total_timesteps:    Total environment steps for training.
        output_dir:         Root directory for model checkpoints and logs.
        obs_mode:           Observation mode — 'full' (13-dim) or 'simple' (5-dim).
        eval_freq:          Evaluate every N steps (used by EvalCallback).
        n_eval_episodes:    Number of episodes per evaluation.
        device:             PyTorch device ('auto', 'cpu', 'cuda').
        verbose:            SB3 verbosity level (0=silent, 1=info).

    Returns:
        Path to the run directory containing the trained model.
    """
    if reward_config_name not in REWARD_CONFIGS:
        raise ValueError(
            f"Unknown reward config '{reward_config_name}'. "
            f"Available: {list(REWARD_CONFIGS.keys())}"
        )

    # ── Directories ──────────────────────────────────────────────────────────
    obs_suffix = f'_{obs_mode}' if obs_mode != 'full' else ''
    run_name = f'sac_{scenario}_{reward_config_name}_seed{seed}{obs_suffix}'
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'best_model').mkdir(exist_ok=True)
    (run_dir / 'checkpoints').mkdir(exist_ok=True)
    (run_dir / 'eval').mkdir(exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  SAC Training")
    print(f"  Scenario:       {scenario}")
    print(f"  Reward config:  {reward_config_name}")
    print(f"  Obs mode:       {obs_mode}")
    print(f"  Seed:           {seed}")
    print(f"  Timesteps:      {total_timesteps:,}")
    print(f"  Output:         {run_dir}")
    print(f"{'='*65}")

    # ── Environments ─────────────────────────────────────────────────────────
    reward_config = REWARD_CONFIGS[reward_config_name]

    train_env = Monitor(ADM1Env_v2(
        scenario_name=scenario,
        reward_config=reward_config,
        obs_mode=obs_mode,
    ))
    train_env.reset(seed=seed)

    eval_env = Monitor(ADM1Env_v2(
        scenario_name=scenario,
        reward_config=reward_config,
        obs_mode=obs_mode,
    ))
    eval_env.reset(seed=seed + 1)

    # ── Model ────────────────────────────────────────────────────────────────
    model = SAC(
        'MlpPolicy',
        train_env,
        verbose=verbose,
        seed=seed,
        device=device,
        tensorboard_log=None,
        **SAC_HYPERPARAMS,
    )

    # ── Callbacks ────────────────────────────────────────────────────────────
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / 'best_model'),
        log_path=str(run_dir / 'eval'),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=str(run_dir / 'checkpoints'),
        name_prefix='sac_model',
    )
    clamp_cb = _make_ent_coef_clamp_callback(
        max_ent_coef=ENT_COEF_MAX,
        min_ent_coef=ENT_COEF_MIN,
    )

    # ── Train ────────────────────────────────────────────────────────────────
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

    # ── Save ─────────────────────────────────────────────────────────────────
    model.save(str(run_dir / 'final_model'))

    meta = {
        'algo':             'sac',
        'scenario':         scenario,
        'reward_config':    reward_config_name,
        'obs_mode':         obs_mode,
        'seed':             seed,
        'total_timesteps':  total_timesteps,
        'elapsed_seconds':  elapsed,
        'ent_coef_bounds':  [ENT_COEF_MIN, ENT_COEF_MAX],
        'hyperparams':      {k: str(v) if isinstance(v, dict) else v
                             for k, v in SAC_HYPERPARAMS.items()},
    }
    with open(run_dir / 'run_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    train_env.close()
    eval_env.close()

    print(f"  Model saved → {run_dir}")
    print(f"  Best model  → {run_dir / 'best_model' / 'best_model.zip'}")
    return run_dir


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train SAC agent on ADM1 biogas control.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--scenario', type=str, default='nominal',
                        choices=PAPER_SCENARIOS,
                        help='ADM1 scenario name')
    parser.add_argument('--reward-config', type=str, default='safety_first',
                        help='Reward configuration key (training/reward_configs.py)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--timesteps', type=int, default=DEFAULT_TOTAL_TIMESTEPS,
                        help='Total training timesteps')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Root directory for saved models')
    parser.add_argument('--obs-mode', type=str, default='full',
                        choices=['full', 'simple'],
                        help='Observation mode: full (13-dim) or simple (5-dim)')
    parser.add_argument('--device', type=str, default='auto',
                        help='PyTorch device (auto, cpu, cuda)')
    args = parser.parse_args()

    train_sac(
        scenario=args.scenario,
        reward_config_name=args.reward_config,
        seed=args.seed,
        total_timesteps=args.timesteps,
        output_dir=args.output_dir,
        obs_mode=args.obs_mode,
        device=args.device,
    )


if __name__ == '__main__':
    main()

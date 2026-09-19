#!/usr/bin/env python3
"""
PPO Training Script for ADM1 Biogas Control
============================================

Trains a Proximal Policy Optimisation (PPO) agent on a single ADM1 scenario.
PPO serves as an on-policy RL baseline alongside the off-policy SAC agent.

Key differences from SAC:
  - On-policy: collects n_steps rollouts before each update (no replay buffer)
  - Fixed entropy coefficient (not auto-tuned)
  - Typically less sample-efficient but more stable on continuous control tasks

Supports the same τₐ curriculum as train_sac.py via --curriculum flag.

Usage:
    # Train on cold_winter (paper default scenario for curriculum)
    python training/train_ppo.py --scenario cold_winter --seed 42

    # Curriculum + F_K shaping (experiment 3 PPO variant)
    python training/train_ppo.py \\
        --scenario cold_winter \\
        --reward-config safety_first_curriculum \\
        --curriculum --seed 42

    # Full run: all 6 scenarios × 3 seeds
    for scenario in nominal high_load low_load shock_load temperature_drop cold_winter; do
        for seed in 42 123 456; do
            python training/train_ppo.py --scenario $scenario --seed $seed
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
from training.train_sac import TauACurriculumCallback   # reuse curriculum callback

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv


# ── Paper hyperparameters ─────────────────────────────────────────────────────

PPO_HYPERPARAMS = {
    'learning_rate':  3e-4,
    'n_steps':        2048,      # steps per env per rollout (≈0.7 episodes)
    'batch_size':     64,
    'n_epochs':       10,
    'gamma':          0.99,
    'gae_lambda':     0.95,
    'clip_range':     0.2,
    'ent_coef':       0.01,      # fixed (not auto-tuned)
    'vf_coef':        0.5,
    'max_grad_norm':  0.5,
    'policy_kwargs':  dict(net_arch=[256, 256]),
}

DEFAULT_TOTAL_TIMESTEPS = 300_000

PAPER_SCENARIOS = [
    'nominal', 'high_load', 'low_load',
    'shock_load', 'temperature_drop', 'cold_winter',
]


# ── Training function ─────────────────────────────────────────────────────────

def train_ppo(
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
    curriculum_tau_a: bool = False,
    tau_a_start: float = 7.0,
    tau_a_end: float = 30.0,
    curriculum_end_frac: float = 0.8,
    n_envs: int = 1,
) -> Path:
    """
    Train a PPO agent on a single ADM1 scenario.

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
    cur_suffix = '_curriculum' if curriculum_tau_a else ''
    run_name = f'ppo_{scenario}_{reward_config_name}_seed{seed}{obs_suffix}{cur_suffix}'
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'best_model').mkdir(exist_ok=True)
    (run_dir / 'checkpoints').mkdir(exist_ok=True)
    (run_dir / 'eval').mkdir(exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  PPO Training")
    print(f"  Scenario:       {scenario}")
    print(f"  Reward config:  {reward_config_name}")
    print(f"  Obs mode:       {obs_mode}")
    print(f"  Seed:           {seed}")
    print(f"  Timesteps:      {total_timesteps:,}")
    if curriculum_tau_a:
        ramp_end = int(total_timesteps * curriculum_end_frac)
        print(f"  Curriculum τₐ:  {tau_a_start:.0f}d → {tau_a_end:.0f}d "
              f"over first {ramp_end:,} steps, then fixed at {tau_a_end:.0f}d")
    print(f"  Output:         {run_dir}")
    print(f"{'='*65}")

    # ── Environments ─────────────────────────────────────────────────────────
    reward_config = REWARD_CONFIGS[reward_config_name]

    def make_env(seed_offset: int):
        def _init():
            env = Monitor(ADM1Env_v2(
                scenario_name=scenario,
                reward_config=reward_config,
                obs_mode=obs_mode,
            ))
            env.reset(seed=seed + seed_offset)
            return env
        return _init

    if n_envs > 1:
        train_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
        print(f"  Parallel envs:  {n_envs} (SubprocVecEnv)")
    else:
        train_env = DummyVecEnv([make_env(0)])

    eval_env = Monitor(ADM1Env_v2(
        scenario_name=scenario,
        reward_config=reward_config,
        obs_mode=obs_mode,
    ))
    eval_env.reset(seed=seed + 1000)

    # ── Model ────────────────────────────────────────────────────────────────
    model = PPO(
        'MlpPolicy',
        train_env,
        verbose=verbose,
        seed=seed,
        device=device,
        tensorboard_log=str(run_dir / 'tensorboard'),
        **PPO_HYPERPARAMS,
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
        name_prefix='ppo_model',
    )

    callbacks = [eval_cb, ckpt_cb]
    if curriculum_tau_a:
        curriculum_cb = TauACurriculumCallback(
            total_timesteps=total_timesteps,
            tau_a_start=tau_a_start,
            tau_a_end=tau_a_end,
            curriculum_end_frac=curriculum_end_frac,
            verbose=verbose,
        )
        callbacks.append(curriculum_cb)

    # ── Train ────────────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=(verbose >= 1),
        )
    except KeyboardInterrupt:
        print("\n  [Training interrupted — saving current model]")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed / 60:.1f} min")

    # ── Save ─────────────────────────────────────────────────────────────────
    model.save(str(run_dir / 'final_model'))

    meta = {
        'algo':             'ppo',
        'scenario':         scenario,
        'reward_config':    reward_config_name,
        'obs_mode':         obs_mode,
        'seed':             seed,
        'total_timesteps':  total_timesteps,
        'elapsed_seconds':  elapsed,
        'hyperparams':      {k: str(v) if isinstance(v, dict) else v
                             for k, v in PPO_HYPERPARAMS.items()},
        'curriculum': {
            'enabled':             curriculum_tau_a,
            'tau_a_start':         tau_a_start if curriculum_tau_a else None,
            'tau_a_end':           tau_a_end if curriculum_tau_a else None,
            'curriculum_end_frac': curriculum_end_frac if curriculum_tau_a else None,
        },
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
        description='Train PPO agent on ADM1 biogas control.',
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
    parser.add_argument('--curriculum', action='store_true', default=False,
                        help='Enable τₐ curriculum: ramp from tau-a-start to tau-a-end')
    parser.add_argument('--tau-a-start', type=float, default=7.0,
                        help='Initial τₐ (days) for curriculum — easy phase')
    parser.add_argument('--tau-a-end', type=float, default=30.0,
                        help='Final τₐ (days) for curriculum — hard phase (paper default)')
    parser.add_argument('--curriculum-end-frac', type=float, default=0.8,
                        help='Fraction of training steps used for the linear τₐ ramp')
    parser.add_argument('--n-envs', type=int, default=1,
                        help='Number of parallel envs (SubprocVecEnv when >1)')
    args = parser.parse_args()

    train_ppo(
        scenario=args.scenario,
        reward_config_name=args.reward_config,
        seed=args.seed,
        total_timesteps=args.timesteps,
        output_dir=args.output_dir,
        obs_mode=args.obs_mode,
        device=args.device,
        curriculum_tau_a=args.curriculum,
        tau_a_start=args.tau_a_start,
        tau_a_end=args.tau_a_end,
        curriculum_end_frac=args.curriculum_end_frac,
        n_envs=args.n_envs,
    )


if __name__ == '__main__':
    main()

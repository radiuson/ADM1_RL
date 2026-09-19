#!/usr/bin/env python3
"""
SAC Training with Scenario-Difficulty Curriculum
=================================================

Trains SAC using ScenarioCurriculumEnv, which progressively introduces
harder scenarios based on step count rather than using a fixed scenario.

The eval env always uses 'nominal' so learning curves are comparable.

Model is saved to:
    models/sac_scenario_cur_<reward_config>_<stages_tag>_seed<seed>/

Usage:
    python training/train_sac_scenario_cur.py --seed 42
    python training/train_sac_scenario_cur.py --seed 42 --stages high_load_first
    python training/train_sac_scenario_cur.py --seed 42 --reward-config safety_first_curriculum
"""

import argparse
import json
import time
from pathlib import Path

import torch

from env.adm1_gym_env import ADM1Env_v2
from env.scenario_curriculum_env import (
    ScenarioCurriculumEnv,
    DEFAULT_STAGES,
    HIGH_LOAD_FIRST_STAGES,
    FAST_STAGES,
    SLOW_STAGES,
    UNIFORM_STAGES,
    EF_STAGES,
)
from training.reward_configs import REWARD_CONFIGS
from training.train_sac import SAC_HYPERPARAMS, ENT_COEF_MIN, ENT_COEF_MAX, _make_ent_coef_clamp_callback

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList,
)


STAGES_MAP = {
    'default':         DEFAULT_STAGES,
    'high_load_first': HIGH_LOAD_FIRST_STAGES,
    'fast':            FAST_STAGES,
    'slow':            SLOW_STAGES,
    'uniform_random':  UNIFORM_STAGES,
    'ef':              EF_STAGES,
}

DEFAULT_TOTAL_TIMESTEPS = 300_000


def train_sac_scenario_cur(
    reward_config_name: str = 'safety_first',
    seed: int = 42,
    stages_name: str = 'default',
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
    output_dir: str = 'models',
    obs_mode: str = 'full',
    eval_freq: int = 10_000,
    n_eval_episodes: int = 5,
    device: str = 'auto',
    verbose: int = 1,
) -> Path:
    """
    Train SAC with scenario curriculum.

    Returns:
        Path to the run directory containing final_model.zip.
    """
    if reward_config_name not in REWARD_CONFIGS:
        raise ValueError(
            f"Unknown reward config '{reward_config_name}'. "
            f"Available: {list(REWARD_CONFIGS.keys())}"
        )
    if stages_name not in STAGES_MAP:
        raise ValueError(
            f"Unknown stages '{stages_name}'. "
            f"Available: {list(STAGES_MAP.keys())}"
        )

    stages = STAGES_MAP[stages_name]
    reward_config = REWARD_CONFIGS[reward_config_name]

    run_name = f'sac_scenario_cur_{reward_config_name}_{stages_name}_seed{seed}'
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'best_model').mkdir(exist_ok=True)
    (run_dir / 'checkpoints').mkdir(exist_ok=True)
    (run_dir / 'eval').mkdir(exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  SAC + Scenario Curriculum Training")
    print(f"  Reward config:  {reward_config_name}")
    print(f"  Stages:         {stages_name}")
    print(f"  Seed:           {seed}")
    print(f"  Timesteps:      {total_timesteps:,}")
    print(f"  Output:         {run_dir}")
    for entry in stages:
        thresh, pool = entry[0], entry[1]
        rname = entry[2] if len(entry) > 2 else None
        suffix = f'  reward={rname}' if rname else ''
        print(f"    [{thresh:>7,}]  {pool}{suffix}")
    print(f"{'='*65}")

    # ── Training env: scenario curriculum ─────────────────────────────────────
    train_env = Monitor(ScenarioCurriculumEnv(
        reward_config=reward_config,
        stages=stages,
        obs_mode=obs_mode,
        rng_seed=seed,
    ))
    train_env.reset(seed=seed)

    # ── Eval env: fixed nominal scenario (interpretable learning curve) ───────
    eval_env = Monitor(ADM1Env_v2(
        scenario_name='nominal',
        reward_config=reward_config,
        obs_mode=obs_mode,
    ))
    eval_env.reset(seed=seed + 1)

    # ── SAC model ─────────────────────────────────────────────────────────────
    model = SAC(
        'MlpPolicy',
        train_env,
        verbose=verbose,
        seed=seed,
        device=device,
        tensorboard_log=str(run_dir / 'tensorboard'),
        **SAC_HYPERPARAMS,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
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

    # ── Train ─────────────────────────────────────────────────────────────────
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
        'algo':            'sac',
        'reward_config':   reward_config_name,
        'stages_name':     stages_name,
        'stages':          [(e[0], list(e[1]), e[2] if len(e) > 2 else None)
                            for e in stages],
        'obs_mode':        obs_mode,
        'seed':            seed,
        'total_timesteps': total_timesteps,
        'elapsed_seconds': elapsed,
        'curriculum_type': 'scenario_difficulty',
        'hyperparams':     {k: str(v) if isinstance(v, dict) else v
                            for k, v in SAC_HYPERPARAMS.items()},
    }
    with open(run_dir / 'run_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    train_env.close()
    eval_env.close()

    print(f"  Model saved → {run_dir}")
    return run_dir


def main():
    parser = argparse.ArgumentParser(
        description='Train SAC with scenario-difficulty curriculum.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--reward-config', type=str, default='safety_first',
                        help='Reward config key')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stages', type=str, default='default',
                        choices=list(STAGES_MAP.keys()),
                        help='Curriculum stage schedule')
    parser.add_argument('--timesteps', type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument('--output-dir', type=str, default='models')
    parser.add_argument('--obs-mode', type=str, default='full',
                        choices=['full', 'simple'])
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--eval-freq', type=int, default=10_000,
                        help='Evaluate every N training steps')
    parser.add_argument('--n-eval-episodes', type=int, default=5)
    args = parser.parse_args()

    train_sac_scenario_cur(
        reward_config_name=args.reward_config,
        seed=args.seed,
        stages_name=args.stages,
        total_timesteps=args.timesteps,
        output_dir=args.output_dir,
        obs_mode=args.obs_mode,
        device=args.device,
        verbose=args.verbose,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
    )


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Parallel Training Launcher for ADM1 RL Ablation
================================================

Runs scenario×seed combinations in parallel using ThreadPoolExecutor.
Replaces the serial loops in run_ablation.py / run_ppo_*.py.

Usage (from ADM1_RL/ directory):
    # PPO baseline only, 4 parallel envs per run, 6 simultaneous jobs
    python training/run_parallel.py --groups ppo_baseline --n-envs 4 --max-workers 6

    # Full 2×2 ablation for both SAC and PPO (all 8 groups × 6 scenarios × 3 seeds = 144 jobs)
    python training/run_parallel.py --groups all --n-envs 4 --max-workers 6

    # Only SAC variants, 12 simultaneous jobs (SAC uses single env, so more parallelism is fine)
    python training/run_parallel.py --groups sac_baseline sac_cur_only sac_fk_only sac_cur_fk --max-workers 12

    # Single group, dry run to see commands
    python training/run_parallel.py --groups ppo_cur_only --dry-run
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCENARIOS = [
    'nominal', 'high_load', 'low_load',
    'shock_load', 'temperature_drop', 'cold_winter',
]
SEEDS = [42, 123, 456]
DEFAULT_TIMESTEPS = 300_000

# (script, reward_config, use_curriculum, algo)
GROUPS = {
    'sac_baseline': ('training/train_sac.py', 'safety_first',            False, 'sac'),
    'sac_cur_only': ('training/train_sac.py', 'safety_first',            True,  'sac'),
    'sac_fk_only':  ('training/train_sac.py', 'safety_first_curriculum', False, 'sac'),
    'sac_cur_fk':   ('training/train_sac.py', 'safety_first_curriculum', True,  'sac'),
    'ppo_baseline': ('training/train_ppo.py', 'safety_first',            False, 'ppo'),
    'ppo_cur_only': ('training/train_ppo.py', 'safety_first',            True,  'ppo'),
    'ppo_fk_only':  ('training/train_ppo.py', 'safety_first_curriculum', False, 'ppo'),
    'ppo_cur_fk':   ('training/train_ppo.py', 'safety_first_curriculum', True,  'ppo'),
}
ALL_GROUP_NAMES = list(GROUPS.keys())


def build_cmd(
    group: str,
    scenario: str,
    seed: int,
    timesteps: int,
    device: str,
    n_envs: int,
) -> list:
    script, reward_config, curriculum, algo = GROUPS[group]
    cmd = [
        sys.executable, script,
        '--scenario',      scenario,
        '--reward-config', reward_config,
        '--seed',          str(seed),
        '--timesteps',     str(timesteps),
        '--device',        device,
    ]
    if curriculum:
        cmd.append('--curriculum')
    if algo == 'ppo' and n_envs > 1:
        cmd.extend(['--n-envs', str(n_envs)])
    return cmd


def run_one(cmd: list, desc: str) -> tuple:
    t0 = time.time()
    ret = subprocess.run(cmd, cwd=str(ROOT)).returncode
    return desc, ret, time.time() - t0


def main():
    parser = argparse.ArgumentParser(
        description='Parallel training launcher for ADM1 RL ablation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--groups', nargs='+', default=['all'],
        help=f'Training groups to run (or "all"). Choices: {ALL_GROUP_NAMES}',
    )
    parser.add_argument('--scenarios', nargs='+', default=SCENARIOS,
                        help='ADM1 scenario names')
    parser.add_argument('--seeds',     nargs='+', type=int, default=SEEDS,
                        help='Random seeds')
    parser.add_argument('--timesteps', type=int, default=DEFAULT_TIMESTEPS,
                        help='Total training timesteps per run')
    parser.add_argument('--device',    default='cuda',
                        help='PyTorch device')
    parser.add_argument('--n-envs',    type=int, default=4,
                        help='Parallel envs per PPO run (SubprocVecEnv); ignored for SAC')
    parser.add_argument('--max-workers', type=int, default=6,
                        help='Max simultaneous training jobs')
    parser.add_argument('--dry-run',   action='store_true',
                        help='Print commands without running them')
    args = parser.parse_args()

    groups = ALL_GROUP_NAMES if 'all' in args.groups else args.groups
    for g in groups:
        if g not in GROUPS:
            parser.error(f"Unknown group '{g}'. Choices: {ALL_GROUP_NAMES}")

    jobs = []
    for group in groups:
        for scenario in args.scenarios:
            for seed in args.seeds:
                desc = f"{group} | {scenario} | seed={seed}"
                cmd  = build_cmd(group, scenario, seed,
                                 args.timesteps, args.device, args.n_envs)
                jobs.append((cmd, desc))

    total = len(jobs)
    print(f"\n{'='*65}")
    print(f"  Parallel Training Launcher")
    print(f"  Groups:       {groups}")
    print(f"  Scenarios:    {args.scenarios}")
    print(f"  Seeds:        {args.seeds}")
    print(f"  Timesteps:    {args.timesteps:,}")
    print(f"  PPO n_envs:   {args.n_envs}")
    print(f"  max_workers:  {args.max_workers}")
    print(f"  Total jobs:   {total}")
    print(f"{'='*65}\n")

    if args.dry_run:
        for cmd, desc in jobs:
            print(f"  [DRY] {desc}")
            print(f"        {' '.join(cmd)}\n")
        return

    t_start = time.time()
    results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(run_one, cmd, desc): desc for cmd, desc in jobs}
        for future in as_completed(futures):
            desc, ret, elapsed = future.result()
            completed += 1
            status = 'OK' if ret == 0 else f'FAILED({ret})'
            print(f"  [{completed:3d}/{total}] {status:<12} {elapsed/60:5.1f}min  {desc}")
            results.append((desc, ret, elapsed))

    total_elapsed = time.time() - t_start
    failures = [(d, r) for d, r, _ in results if r != 0]

    print(f"\n{'='*65}")
    print(f"  DONE — {total} jobs in {total_elapsed/60:.1f} min")
    if failures:
        print(f"  {len(failures)} failure(s):")
        for d, r in failures:
            print(f"    FAILED({r}): {d}")
        sys.exit(1)
    else:
        print(f"  All {total} jobs completed successfully.")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Experiment 3: SAC-Curriculum + F_K Shaping vs PPO Baseline
===========================================================

Sequentially trains:
  1. PPO baseline       — safety_first reward, no curriculum (6 scenarios × 3 seeds)
  2. SAC-Curriculum     — safety_first_curriculum reward + τₐ curriculum (6 × 3)

All models are saved under models/ with standard naming:
  ppo_<scenario>_safety_first_seed<N>/
  sac_<scenario>_safety_first_curriculum_seed<N>_curriculum/

Usage (from ADM1_RL/ directory):
    python training/run_experiment3.py
    python training/run_experiment3.py --only ppo       # PPO only
    python training/run_experiment3.py --only sac_cur   # SAC-Curriculum only
    python training/run_experiment3.py --scenarios cold_winter nominal
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCENARIOS = [
    'nominal', 'high_load', 'low_load',
    'shock_load', 'temperature_drop', 'cold_winter',
]
SEEDS = [42, 123, 456]
TIMESTEPS = 300_000


def run(cmd: list, desc: str) -> int:
    print(f"\n{'='*65}")
    print(f"  {desc}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*65}")
    t0 = time.time()
    ret = subprocess.run(cmd, cwd=str(ROOT)).returncode
    elapsed = time.time() - t0
    status = 'OK' if ret == 0 else f'FAILED (code {ret})'
    print(f"\n  → {status}  [{elapsed/60:.1f} min]")
    return ret


def main():
    parser = argparse.ArgumentParser(
        description='Run Experiment 3: PPO baseline + SAC-Curriculum',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--only', choices=['ppo', 'sac_cur', 'all'], default='all',
                        help='Which training to run')
    parser.add_argument('--scenarios', nargs='+', default=SCENARIOS,
                        help='Scenarios to train on')
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS,
                        help='Random seeds')
    parser.add_argument('--timesteps', type=int, default=TIMESTEPS,
                        help='Total training steps per run')
    parser.add_argument('--device', default='auto',
                        help='PyTorch device for SAC (PPO always uses cpu — SB3 recommendation)')
    parser.add_argument('--ppo-device', default='cpu',
                        help='PyTorch device for PPO (default: cpu)')
    args = parser.parse_args()

    py = sys.executable
    failures = []

    total_ppo = len(args.scenarios) * len(args.seeds)
    total_sac = len(args.scenarios) * len(args.seeds)

    # ── 1. PPO baseline ───────────────────────────────────────────────────────
    if args.only in ('ppo', 'all'):
        print(f"\n{'#'*65}")
        print(f"  PHASE 1: PPO baseline ({total_ppo} runs)")
        print(f"{'#'*65}")
        for i, scenario in enumerate(args.scenarios):
            for j, seed in enumerate(args.seeds):
                n = i * len(args.seeds) + j + 1
                desc = f"PPO [{n}/{total_ppo}] {scenario} seed={seed}"
                cmd = [
                    py, 'training/train_ppo.py',
                    '--scenario', scenario,
                    '--reward-config', 'safety_first',
                    '--seed', str(seed),
                    '--timesteps', str(args.timesteps),
                    '--device', args.ppo_device,   # cpu recommended for MLP PPO
                ]
                ret = run(cmd, desc)
                if ret != 0:
                    failures.append(desc)

    # ── 2. SAC-Curriculum + F_K shaping ──────────────────────────────────────
    if args.only in ('sac_cur', 'all'):
        print(f"\n{'#'*65}")
        print(f"  PHASE 2: SAC-Curriculum + F_K shaping ({total_sac} runs)")
        print(f"{'#'*65}")
        for i, scenario in enumerate(args.scenarios):
            for j, seed in enumerate(args.seeds):
                n = i * len(args.seeds) + j + 1
                desc = f"SAC-Curriculum [{n}/{total_sac}] {scenario} seed={seed}"
                cmd = [
                    py, 'training/train_sac.py',
                    '--scenario', scenario,
                    '--reward-config', 'safety_first_curriculum',
                    '--curriculum',
                    '--seed', str(seed),
                    '--timesteps', str(args.timesteps),
                    '--device', args.device,
                ]
                ret = run(cmd, desc)
                if ret != 0:
                    failures.append(desc)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    if failures:
        print(f"  DONE with {len(failures)} failure(s):")
        for f in failures:
            print(f"    ✗ {f}")
    else:
        total = (total_ppo if args.only in ('ppo','all') else 0) + \
                (total_sac if args.only in ('sac_cur','all') else 0)
        print(f"  DONE — all {total} runs completed successfully")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()

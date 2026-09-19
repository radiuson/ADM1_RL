#!/usr/bin/env python3
"""
PPO-Cur+FK Training: τₐ Curriculum + F_K Shaping
=================================================

Trains PPO with both curriculum scheduling (τₐ) AND F_K safety shaping
(safety_first_curriculum reward).  This completes the 2×2 ablation matrix
for the PPO baseline, mirroring SAC-Cur+FK (Exp3).

Model output path: models/ppo_{scenario}_safety_first_curriculum_seed{seed}_curriculum/

Usage (from ADM1_RL/ directory):
    python training/run_ppo_cur_fk.py
    python training/run_ppo_cur_fk.py --scenarios cold_winter nominal
    python training/run_ppo_cur_fk.py --device cpu
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
SEEDS     = [42, 123, 456]
TIMESTEPS = 300_000


def run(cmd: list, desc: str) -> int:
    print(f"\n{'='*65}")
    print(f"  {desc}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*65}")
    t0  = time.time()
    ret = subprocess.run(cmd, cwd=str(ROOT)).returncode
    elapsed = time.time() - t0
    status = 'OK' if ret == 0 else f'FAILED (code {ret})'
    print(f"\n  → {status}  [{elapsed/60:.1f} min]")
    return ret


def main():
    parser = argparse.ArgumentParser(
        description='PPO-Cur+FK: curriculum + F_K shaping training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--scenarios', nargs='+', default=SCENARIOS)
    parser.add_argument('--seeds',     nargs='+', type=int, default=SEEDS)
    parser.add_argument('--timesteps', type=int, default=TIMESTEPS)
    parser.add_argument('--device',    default='cuda')
    args = parser.parse_args()

    py       = sys.executable
    failures = []
    total    = len(args.scenarios) * len(args.seeds)

    print(f"\n{'#'*65}")
    print(f"  PPO-Cur+FK ({total} runs)")
    print(f"  reward=safety_first_curriculum  curriculum=True")
    print(f"{'#'*65}")

    for i, scenario in enumerate(args.scenarios):
        for j, seed in enumerate(args.seeds):
            n    = i * len(args.seeds) + j + 1
            desc = f"PPO-Cur+FK [{n}/{total}] {scenario} seed={seed}"
            # Skip if already trained
            model_path = ROOT / 'models' / f'ppo_{scenario}_safety_first_curriculum_seed{seed}_curriculum' / 'final_model.zip'
            if model_path.exists():
                print(f"\n  [SKIP] already exists: {model_path.parent.name}")
                continue
            cmd = [
                py, 'training/train_ppo.py',
                '--scenario',      scenario,
                '--reward-config', 'safety_first_curriculum',
                '--curriculum',
                '--seed',          str(seed),
                '--timesteps',     str(args.timesteps),
                '--device',        args.device,
            ]
            if run(cmd, desc) != 0:
                failures.append(desc)

    print(f"\n{'='*65}")
    if failures:
        print(f"  DONE with {len(failures)} failure(s):")
        for f in failures:
            print(f"    ✗ {f}")
    else:
        print(f"  DONE — all PPO-Cur+FK runs completed successfully")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()

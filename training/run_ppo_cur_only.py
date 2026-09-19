#!/usr/bin/env python3
"""
PPO-CurOnly Training: τₐ Curriculum only (no F_K shaping)
==========================================================
reward_config = safety_first  +  --curriculum flag
Model output: models/ppo_{scenario}_safety_first_seed{seed}_curriculum/
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCENARIOS = ['nominal', 'high_load', 'low_load', 'shock_load', 'temperature_drop', 'cold_winter']
SEEDS     = [42, 123, 456]
TIMESTEPS = 300_000


def run(cmd, desc):
    print(f"\n{'='*65}\n  {desc}\n{'='*65}")
    t0  = time.time()
    ret = subprocess.run(cmd, cwd=str(ROOT)).returncode
    status = 'OK' if ret == 0 else f'FAILED ({ret})'
    print(f"\n  → {status}  [{(time.time()-t0)/60:.1f} min]")
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenarios', nargs='+', default=SCENARIOS)
    parser.add_argument('--seeds',     nargs='+', type=int, default=SEEDS)
    parser.add_argument('--timesteps', type=int, default=TIMESTEPS)
    parser.add_argument('--device',    default='cuda')
    args = parser.parse_args()

    py       = sys.executable
    failures = []
    total    = len(args.scenarios) * len(args.seeds)

    print(f"\n{'#'*65}")
    print(f"  PPO-CurOnly ({total} runs)")
    print(f"  reward=safety_first  curriculum=True")
    print(f"{'#'*65}")

    for i, scenario in enumerate(args.scenarios):
        for j, seed in enumerate(args.seeds):
            n    = i * len(args.seeds) + j + 1
            desc = f"PPO-CurOnly [{n}/{total}] {scenario} seed={seed}"
            model_path = ROOT / 'models' / f'ppo_{scenario}_safety_first_seed{seed}_curriculum' / 'final_model.zip'
            if model_path.exists():
                print(f"\n  [SKIP] already exists: {model_path.parent.name}")
                continue
            cmd = [py, 'training/train_ppo.py',
                   '--scenario',      scenario,
                   '--reward-config', 'safety_first',
                   '--curriculum',
                   '--seed',          str(seed),
                   '--timesteps',     str(args.timesteps),
                   '--device',        args.device]
            if run(cmd, desc) != 0:
                failures.append(desc)

    print(f"\n{'='*65}")
    if failures:
        print(f"  DONE with {len(failures)} failure(s):")
        for f in failures:
            print(f"    ✗ {f}")
    else:
        print(f"  DONE — all PPO-CurOnly runs completed successfully")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()

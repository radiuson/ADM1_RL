#!/usr/bin/env python3
"""
Ablation Study: τₐ Curriculum × F_K Shaping
=============================================

Completes the 2×2 ablation matrix for the paper:

    reward\\curriculum  |  No Curriculum       |  With Curriculum
    ───────────────────────────────────────────────────────────────
    safety_first       |  SAC (baseline ✓)    |  SAC-Cur ONLY   ← Phase 3
    safety_first_cur   |  SAC-FK ONLY         |  SAC-Cur+FK ✓  ← Phase 4

Phase 3: SAC + τₐ Curriculum ONLY (no F_K shaping, safety_first reward)
Phase 4: SAC + F_K Shaping ONLY   (no curriculum, safety_first_curriculum reward)

Usage (from ADM1_RL/ directory):
    python training/run_ablation.py               # both phases
    python training/run_ablation.py --only phase3
    python training/run_ablation.py --only phase4
    python training/run_ablation.py --scenarios cold_winter nominal
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
        description='Ablation: curriculum-only and F_K-only SAC variants',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--only', choices=['phase3', 'phase4', 'both'],
                        default='both', help='Which ablation phase to run')
    parser.add_argument('--scenarios', nargs='+', default=SCENARIOS)
    parser.add_argument('--seeds',     nargs='+', type=int, default=SEEDS)
    parser.add_argument('--timesteps', type=int, default=TIMESTEPS)
    parser.add_argument('--device',    default='cuda')
    args = parser.parse_args()

    py       = sys.executable
    failures = []
    total    = len(args.scenarios) * len(args.seeds)

    # ── Phase 3: SAC + Curriculum ONLY (reward = safety_first, no F_K) ──────
    if args.only in ('phase3', 'both'):
        print(f"\n{'#'*65}")
        print(f"  PHASE 3: SAC + τₐ Curriculum ONLY ({total} runs)")
        print(f"  reward=safety_first  curriculum=True  fk_bonus=0")
        print(f"{'#'*65}")
        for i, scenario in enumerate(args.scenarios):
            for j, seed in enumerate(args.seeds):
                n    = i * len(args.seeds) + j + 1
                desc = f"SAC-CurOnly [{n}/{total}] {scenario} seed={seed}"
                cmd  = [
                    py, 'training/train_sac.py',
                    '--scenario',      scenario,
                    '--reward-config', 'safety_first',   # NO F_K shaping
                    '--curriculum',                       # WITH τₐ curriculum
                    '--seed',          str(seed),
                    '--timesteps',     str(args.timesteps),
                    '--device',        args.device,
                ]
                if run(cmd, desc) != 0:
                    failures.append(desc)

    # ── Phase 4: SAC + F_K Shaping ONLY (reward = safety_first_curriculum) ──
    if args.only in ('phase4', 'both'):
        print(f"\n{'#'*65}")
        print(f"  PHASE 4: SAC + F_K Shaping ONLY ({total} runs)")
        print(f"  reward=safety_first_curriculum  curriculum=False")
        print(f"{'#'*65}")
        for i, scenario in enumerate(args.scenarios):
            for j, seed in enumerate(args.seeds):
                n    = i * len(args.seeds) + j + 1
                desc = f"SAC-FKOnly [{n}/{total}] {scenario} seed={seed}"
                cmd  = [
                    py, 'training/train_sac.py',
                    '--scenario',      scenario,
                    '--reward-config', 'safety_first_curriculum',  # WITH F_K
                    # NO --curriculum flag
                    '--seed',          str(seed),
                    '--timesteps',     str(args.timesteps),
                    '--device',        args.device,
                ]
                if run(cmd, desc) != 0:
                    failures.append(desc)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    if failures:
        print(f"  DONE with {len(failures)} failure(s):")
        for f in failures:
            print(f"    ✗ {f}")
    else:
        n_run = total * (1 if args.only != 'both' else 2)
        print(f"  DONE — all {n_run} ablation runs completed successfully")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()

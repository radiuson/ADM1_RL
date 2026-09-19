#!/usr/bin/env python3
"""
Experiment 3 Resume — Smart restart with device optimisation.

Skips runs where final_model.zip already exists.
PPO uses CPU (SB3 recommendation for MLP — GPU adds transfer overhead with <2% utilisation).
SAC-Curriculum uses CUDA (off-policy, large replay buffer, GPU-efficient).

Usage (from ADM1_RL/ directory):
    python training/run_experiment3_resume.py
"""

import subprocess, sys, time
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
MODELS    = ROOT / 'models'
SCENARIOS = ['nominal','high_load','low_load','shock_load','temperature_drop','cold_winter']
SEEDS     = [42, 123, 456]
STEPS     = 300_000
PY        = sys.executable


def done(run_dir: Path) -> bool:
    return (run_dir / 'final_model.zip').exists()


def run(cmd, desc):
    print(f"\n{'='*65}\n  {desc}\n{'='*65}")
    t0  = time.time()
    ret = subprocess.run(cmd, cwd=str(ROOT)).returncode
    print(f"  → {'OK' if ret==0 else f'FAILED({ret})'}  [{(time.time()-t0)/60:.1f} min]")
    return ret


failures = []

# ── Phase 1: PPO on CPU ───────────────────────────────────────────────────────
print(f"\n{'#'*65}\n  PHASE 1: PPO baseline (CPU) — skipping completed runs\n{'#'*65}")
total = len(SCENARIOS) * len(SEEDS)
n = 0
for scenario in SCENARIOS:
    for seed in SEEDS:
        n += 1
        run_dir = MODELS / f'ppo_{scenario}_safety_first_seed{seed}'
        if done(run_dir):
            print(f"  [skip] PPO [{n}/{total}] {scenario} seed={seed} — already done")
            continue
        desc = f"PPO [{n}/{total}] {scenario} seed={seed}"
        cmd  = [PY, 'training/train_ppo.py',
                '--scenario', scenario, '--reward-config', 'safety_first',
                '--seed', str(seed), '--timesteps', str(STEPS),
                '--device', 'cpu']   # CPU for MLP PPO
        if run(cmd, desc) != 0:
            failures.append(desc)

# ── Phase 2: SAC-Curriculum + F_K on CUDA ────────────────────────────────────
print(f"\n{'#'*65}\n  PHASE 2: SAC-Curriculum+FK (CUDA) — skipping completed runs\n{'#'*65}")
n = 0
for scenario in SCENARIOS:
    for seed in SEEDS:
        n += 1
        run_dir = MODELS / f'sac_{scenario}_safety_first_curriculum_seed{seed}_curriculum'
        if done(run_dir):
            print(f"  [skip] SAC-Cur [{n}/{total}] {scenario} seed={seed} — already done")
            continue
        desc = f"SAC-Curriculum [{n}/{total}] {scenario} seed={seed}"
        cmd  = [PY, 'training/train_sac.py',
                '--scenario', scenario, '--reward-config', 'safety_first_curriculum',
                '--curriculum',
                '--seed', str(seed), '--timesteps', str(STEPS),
                '--device', 'cuda']  # CUDA for off-policy SAC
        if run(cmd, desc) != 0:
            failures.append(desc)

print(f"\n{'='*65}")
if failures:
    print(f"  DONE with {len(failures)} failure(s): {failures}")
else:
    print(f"  ALL RUNS COMPLETE")
print(f"{'='*65}")

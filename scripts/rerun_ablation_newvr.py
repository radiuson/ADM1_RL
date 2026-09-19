#!/usr/bin/env python3
"""
Re-evaluate ablation models (sf_linear_only, sf_constant_only) with
the new MetricsCalculator (pH+VFA only, NH3 excluded from VR).

Prints new VR per scenario for comparison with paper's ablation table.

Usage:
    cd /home/ihpc/code/biogas/ADM1_RL
    python scripts/rerun_ablation_newvr.py
"""

import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.run_experiment import evaluate_model_on_scenario

REPRO_DIR = Path('/home/ihpc/code/biogas/ADM1/rl/results_repro')
SCENARIOS  = ['nominal', 'high_load', 'cold_winter']
SEEDS      = [42, 123, 456]
N_EPISODES = 10
NUM_STEPS  = 2880
EVAL_SEED  = 42

configs = [
    {
        'reward_config': 'sf_linear_only',
        'training_dir':  REPRO_DIR / 'paper_direction_a_ablation' / 'training',
    },
    {
        'reward_config': 'sf_constant_only',
        'training_dir':  REPRO_DIR / 'paper_direction_a_ablation_const' / 'training',
    },
]

all_results = {}

for cfg in configs:
    rc  = cfg['reward_config']
    td  = cfg['training_dir']
    by_scen = defaultdict(list)

    print(f"\n{'='*55}")
    print(f"  {rc}")
    print(f"{'='*55}")

    for train_sc in SCENARIOS:
        for seed in SEEDS:
            run_name   = f"sac_{train_sc}_{rc}_seed{seed}"
            model_path = td / run_name / 'best_model' / 'best_model'
            if not (model_path.parent / 'best_model.zip').exists():
                print(f"  [SKIP] {model_path}")
                continue

            # evaluate in-distribution only
            t0 = time.time()
            metrics = evaluate_model_on_scenario(
                model_path=str(model_path),
                test_scenario=train_sc,
                reward_config_name=rc,
                num_steps=NUM_STEPS,
                n_eval_episodes=N_EPISODES,
                seed=EVAL_SEED,
                algo='sac',
                obs_mode='full',
            )
            elapsed = time.time() - t0
            vr  = metrics.get('violation_rate', float('nan'))
            vfa = metrics.get('vfa_max_mean', float('nan'))
            sc  = metrics.get('overall_score', float('nan'))
            print(f"  {train_sc:<20} seed={seed}: VR={vr*100:.1f}%  vfa_max={vfa:.4f}  "
                  f"score={sc:.3f}  [{elapsed:.0f}s]")
            by_scen[train_sc].append({'vr': vr, 'vfa_max': vfa, 'score': sc})

    print(f"\n  === {rc} averages ===")
    for scen in SCENARIOS:
        recs = by_scen[scen]
        if recs:
            avg_vr  = np.mean([r['vr'] for r in recs])
            avg_sc  = np.mean([r['score'] for r in recs])
            print(f"  {scen:<20}: VR={avg_vr*100:.1f}%  score={avg_sc:.3f}  (n={len(recs)})")

    all_results[rc] = dict(by_scen)

print("\n\n=== PAPER TABLE COMPARISON ===")
print("Ablation table shows (OLD VR definition):")
print("  Safety-first Nominal=0.007, High Load=0.139, Cold Winter=0.003")
print("  Constant-only Nominal=0.000, High Load=0.066, Cold Winter=0.001")
print("  Linear-only   Nominal=0.728, High Load=0.830, Cold Winter=0.066")
print()
print("Computed above is NEW VR definition (pH+VFA only).")

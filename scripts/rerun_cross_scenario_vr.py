#!/usr/bin/env python3
"""
Re-evaluate single-scenario SAC (seeds 42/123/456) across all 6 test scenarios
using the new MetricsCalculator (pH+VFA only, NH3 excluded from VR).

Writes results to: results/evaluation_60d/cross_scenario_newvr.json
Also prints the new deployment-wide average VR for use in the paper.

Usage:
    cd /home/ihpc/code/biogas/ADM1_RL
    python scripts/rerun_cross_scenario_vr.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.run_experiment import evaluate_model_on_scenario

TRAIN_SCENARIOS = ['nominal', 'high_load', 'low_load', 'shock_load', 'temperature_drop', 'cold_winter']
TEST_SCENARIOS  = ['nominal', 'high_load', 'low_load', 'shock_load', 'temperature_drop', 'cold_winter']
SEEDS           = [42, 123, 456]
REWARD_KEY      = 'safety_first'
N_EPISODES      = 10
NUM_STEPS       = 2880   # 30-day episodes, same as original cross-scenario eval
EVAL_SEED       = 42
MODELS_DIR      = ROOT / 'models'
OUT_FILE        = ROOT / 'results' / 'evaluation_60d' / 'cross_scenario_newvr.json'

records = []
total = len(TRAIN_SCENARIOS) * len(TEST_SCENARIOS) * len(SEEDS)
idx = 0

for train_sc in TRAIN_SCENARIOS:
    for seed in SEEDS:
        model_path = MODELS_DIR / f'sac_{train_sc}_{REWARD_KEY}_seed{seed}' / 'final_model'
        if not (model_path.parent / 'final_model.zip').exists():
            print(f"  [SKIP] model not found: {model_path}")
            continue

        for test_sc in TEST_SCENARIOS:
            idx += 1
            t0 = time.time()
            metrics = evaluate_model_on_scenario(
                model_path=str(model_path),
                test_scenario=test_sc,
                reward_config_name=REWARD_KEY,
                num_steps=NUM_STEPS,
                n_eval_episodes=N_EPISODES,
                seed=EVAL_SEED,
                algo='sac',
                obs_mode='full',
            )
            elapsed = time.time() - t0

            rec = {
                'train_scenario': train_sc,
                'test_scenario':  test_sc,
                'seed':           seed,
                'violation_rate': metrics.get('violation_rate', float('nan')),
                'ch4_avg':        metrics.get('ch4_avg', float('nan')),
                'overall_score':  metrics.get('overall_score', float('nan')),
                'terminated_rate':metrics.get('terminated_rate', float('nan')),
                'vfa_max_mean':   metrics.get('vfa_max_mean', float('nan')),
            }
            records.append(rec)

            in_out = 'in ' if train_sc == test_sc else 'off'
            print(f"[{idx:3d}/{total}] train={train_sc:<16} test={test_sc:<16} "
                  f"seed={seed} [{in_out}] VR={rec['violation_rate']*100:.1f}%  [{elapsed:.0f}s]")

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_FILE, 'w') as f:
    json.dump(records, f, indent=2)
print(f"\nSaved {len(records)} records → {OUT_FILE}")

# Summary
all_vr = [r['violation_rate'] for r in records]
in_vr  = [r['violation_rate'] for r in records if r['train_scenario'] == r['test_scenario']]
off_vr = [r['violation_rate'] for r in records if r['train_scenario'] != r['test_scenario']]

print(f"\n=== NEW VR SUMMARY (pH+VFA only, NH3 excluded) ===")
print(f"All (108):              avg VR = {np.mean(all_vr)*100:.2f}%")
print(f"In-distribution (18):   avg VR = {np.mean(in_vr)*100:.2f}%")
print(f"Cross-scenario (90):    avg VR = {np.mean(off_vr)*100:.2f}%")

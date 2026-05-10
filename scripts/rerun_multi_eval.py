#!/usr/bin/env python3
"""
Re-evaluation script for multi-scenario SAC results.

Recomputes all per-run JSON files in sac_multi_scenario/evaluation/per_run/
using the fixed MetricsCalculator (union-based violation_rate, strictly in [0,1]).

Usage:
    python scripts/rerun_multi_eval.py --results-dir /path/to/ADM1/rl/results
"""

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from training.run_experiment import evaluate_model_on_scenario


TRAIN_SCENARIOS_KEY = (
    "nominal_high_load_shock_load_low_load_temperature_drop_cold_winter"
)
SEEDS      = [42, 123, 456]
OBS_MODES  = ['full', 'simple']
TEST_SCENARIOS = [
    'nominal', 'high_load', 'shock_load',
    'low_load', 'temperature_drop', 'cold_winter',
]
N_EPISODES = 10
NUM_STEPS  = 2880
EVAL_SEED  = 42


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Re-evaluate multi-scenario SAC results with fixed MetricsCalculator.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--results-dir', required=True,
        help='Root results directory containing sac_multi_scenario/.',
    )
    args = parser.parse_args()

    results_dir  = pathlib.Path(args.results_dir).resolve()
    multi_dir    = results_dir / 'sac_multi_scenario'
    training_dir = multi_dir / 'training'
    per_run_dir  = multi_dir / 'evaluation' / 'per_run'
    per_run_dir.mkdir(parents=True, exist_ok=True)

    total   = len(SEEDS) * len(OBS_MODES) * len(TEST_SCENARIOS)
    done    = 0
    errors  = 0
    t_start = time.time()

    print(f"=== rerun_multi_eval.py started ===")
    print(f"  results_dir : {results_dir}")
    print(f"  total files : {total}")
    print()

    for seed in SEEDS:
        for obs_mode in OBS_MODES:
            obs_suffix = f'_{obs_mode}' if obs_mode != 'full' else ''
            run_name   = f"sac_{TRAIN_SCENARIOS_KEY}_safety_first_seed{seed}{obs_suffix}"
            model_path = str(training_dir / run_name / 'best_model' / 'best_model')

            if not pathlib.Path(model_path + '.zip').exists():
                print(f"  [SKIP] model not found: {model_path}.zip")
                continue

            for test_sc in TEST_SCENARIOS:
                out_file = per_run_dir / f"{run_name}_on_{test_sc}.json"
                t0 = time.time()
                done += 1
                print(f"  [{done}/{total}]  seed{seed} {obs_mode:6s} -> {test_sc:<20} ... ",
                      end='', flush=True)

                try:
                    metrics = evaluate_model_on_scenario(
                        model_path         = model_path,
                        test_scenario      = test_sc,
                        reward_config_name = 'safety_first',
                        num_steps          = NUM_STEPS,
                        n_eval_episodes    = N_EPISODES,
                        seed               = EVAL_SEED,
                        algo               = 'sac',
                        obs_mode           = obs_mode,
                    )

                    record = {
                        'obs_mode':       obs_mode,
                        'train_scenario': TRAIN_SCENARIOS_KEY,
                        'test_scenario':  test_sc,
                        'algo':           'sac',
                        'reward_config':  'safety_first',
                        'seed':           seed,
                        **metrics,
                    }
                    with open(out_file, 'w') as f:
                        json.dump({'record': record, 'raw': metrics}, f, indent=2)

                    vr    = metrics.get('violation_rate', float('nan'))
                    score = metrics.get('overall_score',  float('nan'))
                    flag  = '  WARN: vr>1' if vr > 1.0 else ''
                    print(f"{time.time()-t0:.1f}s  score={score:.3f}  vr={vr:.4f}{flag}")

                except Exception as e:
                    errors += 1
                    print(f"ERROR: {e}")

    elapsed = time.time() - t_start
    print(f"\n=== Done: {done} rerun, {errors} errors in {elapsed/3600:.2f}h ===")


if __name__ == '__main__':
    main()

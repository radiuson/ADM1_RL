#!/usr/bin/env python3
"""
Ablation Reward Evaluation
===========================

Evaluates trained ablation models (``sf_linear_only``,
``sf_constant_only``) using MetricsCalculator and saves per-run JSON files.

This script is used to run or re-run evaluation when the metrics
definition changes without re-training the models.

Expected directory layout (set via --results-dir):

    <results-dir>/
        paper_direction_a_ablation/training/
            sac_<scenario>_sf_linear_only_seed<seed>/best_model/best_model.zip
        paper_direction_a_ablation_const/training/
            sac_<scenario>_sf_constant_only_seed<seed>/best_model/best_model.zip

Output is written to:

    <results-dir>/
        paper_direction_a_ablation_rerun/evaluation/per_run/
        paper_direction_a_ablation_const_rerun/evaluation/per_run/

Usage:
    python rerun_ablation_eval.py --results-dir /path/to/results
"""

import argparse
import json
import pathlib

import numpy as np

from training.run_experiment import evaluate_model_on_scenario


# ── Evaluation settings ───────────────────────────────────────────────────────

SCENARIOS  = ['nominal', 'high_load', 'cold_winter']
SEEDS      = [42, 123, 456]
OBS_MODE   = 'full'
N_EPISODES = 10
NUM_STEPS  = 2880   # 30 days at 15-min intervals
EVAL_SEED  = 42


def run_ablation_eval(results_dir: pathlib.Path) -> None:
    """
    Re-evaluate both ablation reward configurations.

    Args:
        results_dir: Root directory that contains the training subdirectories
                     and where output will be written.
    """
    configs = [
        {
            'reward_config': 'sf_linear_only',
            'training_dir':  results_dir / 'paper_direction_a_ablation' / 'training',
            'out_dir':       results_dir / 'paper_direction_a_ablation_rerun' / 'evaluation' / 'per_run',
        },
        {
            'reward_config': 'sf_constant_only',
            'training_dir':  results_dir / 'paper_direction_a_ablation_const' / 'training',
            'out_dir':       results_dir / 'paper_direction_a_ablation_const_rerun' / 'evaluation' / 'per_run',
        },
    ]

    for cfg in configs:
        rc           = cfg['reward_config']
        training_dir = cfg['training_dir']
        out_dir      = cfg['out_dir']
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  reward_config: {rc}")
        print(f"  training_dir:  {training_dir}")
        print(f"  out_dir:       {out_dir}")
        print(f"{'='*60}")

        for train_sc in SCENARIOS:
            for seed in SEEDS:
                run_name   = f"sac_{train_sc}_{rc}_seed{seed}"
                model_path = training_dir / run_name / 'best_model' / 'best_model'
                model_zip  = model_path.with_suffix('.zip')

                if not model_zip.exists():
                    print(f"  [SKIP] model not found: {model_zip}")
                    continue

                for test_sc in SCENARIOS:
                    out_file = out_dir / f"{run_name}_on_{test_sc}.json"
                    print(f"  {run_name} -> {test_sc} ... ", end='', flush=True)

                    metrics = evaluate_model_on_scenario(
                        model_path         = str(model_path),
                        test_scenario      = test_sc,
                        reward_config_name = rc,
                        num_steps          = NUM_STEPS,
                        n_eval_episodes    = N_EPISODES,
                        seed               = EVAL_SEED,
                        algo               = 'sac',
                        obs_mode           = OBS_MODE,
                    )

                    record = {
                        'obs_mode':       OBS_MODE,
                        'train_scenario': train_sc,
                        'test_scenario':  test_sc,
                        'algo':           'sac',
                        'reward_config':  rc,
                        'seed':           seed,
                        **metrics,
                    }
                    with open(out_file, 'w') as f:
                        json.dump({'record': record, 'raw': metrics}, f, indent=2)

                    score = metrics.get('overall_score', float('nan'))
                    viol  = metrics.get('violation_rate', float('nan'))
                    print(f"score={score:.3f}  viol={viol:.3f}")

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Re-evaluate ablation reward models with current metrics.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--results-dir', type=str, required=True,
        help='Root results directory containing the ablation training subdirectories.',
    )
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir).resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"results-dir not found: {results_dir}")

    run_ablation_eval(results_dir)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Evaluate scenario-curriculum SAC models (cross-scenario).

Finds all models matching:
    models/sac_scenario_cur_<reward_config>_<stages_name>_seed<seed>/

Evaluates each on all 6 test scenarios (10 episodes each, 60 days = 5760 steps).
Results are appended to a dedicated JSON file:
    results/evaluation_60d/scenario_cur_results.json

Usage (from ADM1_RL/ directory):
    python scripts/eval_scenario_cur.py
    python scripts/eval_scenario_cur.py --stages-name high_load_first
    python scripts/eval_scenario_cur.py --reward-config safety_first_curriculum
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.run_experiment import evaluate_model_on_scenario

SCENARIOS   = ['nominal', 'high_load', 'low_load', 'shock_load', 'temperature_drop', 'cold_winter']
SEEDS       = [42, 123, 456]
N_EPISODES  = 10
NUM_STEPS   = 5760   # 60 days at 15-min step
EVAL_SEED   = 42
MODELS_DIR  = ROOT / 'models'
RESULTS_DIR = ROOT / 'results' / 'evaluation_60d'


def flat_record(stages_name, test_sc, reward_config, seed, metrics) -> dict:
    def g(k):
        v = metrics.get(k, float('nan'))
        return float(v) if v is not None else float('nan')

    return {
        'obs_mode':           'full',
        'train_scenario':     f'scenario_cur:{stages_name}',
        'test_scenario':      test_sc,
        'algo':               'sac',
        'reward_config':      reward_config,
        'curriculum':         'scenario_difficulty',
        'stages_name':        stages_name,
        'seed':               seed,
        'reward_mean':        g('reward_mean'),
        'reward_std':         g('reward_std'),
        'ch4_avg':            g('ch4_avg'),
        'ch4_std':            g('ch4_std'),
        'violation_rate':     g('violation_rate'),
        'violation_rate_std': g('violation_rate_std'),
        'overall_score':      g('overall_score'),
        'terminated_rate':    g('terminated_rate'),
        'ph_mean':            g('ph_mean'),
        'vfa_max_mean':       g('vfa_max_mean'),
        'vfa_cvar95':         g('vfa_cvar95'),
        'vfa_worst_episode':  g('vfa_worst_episode'),
        'score_worst':        g('score_worst'),
        'score_cvar95':       g('score_cvar95'),
    }


def find_models(reward_config: str, stages_name: str, models_dir: Path = MODELS_DIR):
    """Return list of (stages_name, reward_config, seed, model_path)."""
    entries = []
    for seed in SEEDS:
        name = f'sac_scenario_cur_{reward_config}_{stages_name}_seed{seed}'
        mp = models_dir / name / 'final_model'
        entries.append((stages_name, reward_config, seed, mp))
    return entries


def load_existing_keys(json_path: Path):
    if not json_path.exists():
        return set()
    with open(json_path) as f:
        data = json.load(f)
    return {(r.get('stages_name', ''), r['test_scenario'],
             r['reward_config'], r['seed']) for r in data}


def run_eval(reward_config: str, stages_name: str, results_dir: Path,
             models_dir: Path = MODELS_DIR, output_tag: str = None) -> list:
    """Evaluate all seeds for a given (reward_config, stages_name). Return new records.

    output_tag: if set, overrides stages_name in the saved JSON (lets you tag
                fast_500k differently from fast_300k without renaming model dirs).
    """
    tag = output_tag or stages_name
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / 'scenario_cur_results.json'
    existing_keys = load_existing_keys(json_path)

    entries = find_models(reward_config, stages_name, models_dir)
    total = len(entries) * len(SCENARIOS)
    done = 0
    new_records = []

    print(f"\n{'#'*65}")
    print(f"  Evaluating: {reward_config} / {stages_name}")
    print(f"  Models: {len(entries)} × 6 scenarios = {total} evals")
    print(f"{'#'*65}")

    for (sn, rc, seed, model_path) in entries:
        mp_zip = Path(str(model_path) + '.zip')
        if not mp_zip.exists() and not model_path.exists():
            print(f"  [SKIP] model not found: {model_path}")
            continue

        for test_sc in SCENARIOS:
            done += 1
            key = (tag, test_sc, rc, seed)
            if key in existing_keys:
                print(f"  [{done:3d}/{total}] SKIP (exists): seed={seed} →{test_sc}")
                continue

            t0 = time.time()
            print(f"  [{done:3d}/{total}] {sn:<20} → {test_sc:<20} seed={seed} ... ",
                  end='', flush=True)
            try:
                metrics = evaluate_model_on_scenario(
                    model_path=str(model_path),
                    test_scenario=test_sc,
                    reward_config_name=rc,
                    num_steps=NUM_STEPS,
                    n_eval_episodes=N_EPISODES,
                    seed=EVAL_SEED,
                    algo='sac',
                    obs_mode='full',
                )
                rec = flat_record(tag, test_sc, rc, seed, metrics)
                new_records.append(rec)
                print(f"score={rec['overall_score']:.3f}  "
                      f"viol={rec['violation_rate']:.1%}  "
                      f"CH4={rec['ch4_avg']:.0f}  [{time.time()-t0:.0f}s]")
            except Exception as e:
                print(f"ERROR: {e}")

    # Append to JSON
    existing_all = []
    if json_path.exists():
        with open(json_path) as f:
            existing_all = json.load(f)

    combined = existing_all + new_records
    with open(json_path, 'w') as f:
        json.dump(combined, f, indent=2)

    print(f"\n  Saved {len(new_records)} new records → {json_path}")
    print(f"  Total records in file: {len(combined)}")
    return new_records


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate scenario-curriculum SAC models.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--reward-config', type=str, default='safety_first')
    parser.add_argument('--stages-name', type=str, default='default')
    parser.add_argument('--results-dir', type=str, default=str(RESULTS_DIR))
    parser.add_argument('--models-dir', type=str, default=str(ROOT / 'models'),
                        help='Directory containing trained model folders')
    parser.add_argument('--output-tag', type=str, default=None,
                        help='Override stages_name label in saved JSON (e.g. fast_500k)')
    args = parser.parse_args()

    run_eval(
        reward_config=args.reward_config,
        stages_name=args.stages_name,
        results_dir=Path(args.results_dir),
        models_dir=Path(args.models_dir),
        output_tag=args.output_tag,
    )


if __name__ == '__main__':
    main()

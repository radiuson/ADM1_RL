#!/usr/bin/env python3
"""
Cross-scenario evaluation for seeds 789 and 1234.

Train scenarios: cold_winter, high_load, shock_load (the ones with 5 trained seeds).
Test  scenarios: all 6.
Seeds:           789, 1234.
Obs modes:       full, simple.

Skips files that already exist (covers in-distribution files written by
eval_missing_seeds.py).  Produces 72 files total (60 new + 12 already done).

Usage:
    python scripts/eval_missing_seeds_cross.py \
        --results-dir /home/ihpc/code/biogas/ADM1/rl/results_repro
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.adm1_gym_env import ADM1Env_v2
from training.reward_configs import REWARD_CONFIGS
from evaluation.metrics_calculator import MetricsCalculator
from stable_baselines3 import SAC

TRAIN_SCENARIOS = ['cold_winter', 'high_load', 'shock_load']
TEST_SCENARIOS  = ['nominal', 'high_load', 'shock_load', 'low_load',
                   'temperature_drop', 'cold_winter']
SEEDS           = [789, 1234]
OBS_MODES       = ['full', 'simple']
REWARD_KEY      = 'safety_first'
N_EPISODES      = 10
NUM_STEPS       = 2880
EVAL_SEED_BASE  = 42


def run_episode(model, env, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    calc = MetricsCalculator()
    for step in range(NUM_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        calc.add_step(obs, action, reward, info)
        if terminated:
            calc.set_terminated(step)
            break
        if truncated:
            break
    return calc.compute_metrics()


def aggregate(episode_metrics):
    def extract(key_path):
        vals = []
        for m in episode_metrics:
            v = m
            for k in key_path.split('.'):
                v = v.get(k, {}) if isinstance(v, dict) else float('nan')
            vals.append(float(v) if not isinstance(v, dict) else float('nan'))
        return np.array(vals, dtype=float)

    def cvar(arr, alpha=0.05):
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return float('nan')
        k = max(1, int(np.ceil(alpha * len(arr))))
        return float(np.mean(np.sort(arr)[:k]))

    scores  = extract('summary.overall_score')
    vr      = extract('safety.violation_rate')
    rewards = extract('reward.mean')
    term    = extract('episode_info.terminated_early')

    return {
        'reward_mean':        float(np.nanmean(rewards)),
        'reward_std':         float(np.nanstd(rewards, ddof=1)),
        'ch4_avg':            float(np.nanmean(extract('production.avg_ch4_flow'))),
        'ch4_std':            float(np.nanstd(extract('production.avg_ch4_flow'), ddof=1)),
        'violation_rate':     float(np.nanmean(vr)),
        'violation_rate_std': float(np.nanstd(vr, ddof=1)),
        'vfa_cvar95':         cvar(vr, alpha=0.05),
        'vfa_worst_episode':  float(np.nanmax(vr)) if len(vr) > 0 else float('nan'),
        'overall_score':      float(np.nanmean(scores)),
        'score_worst':        float(np.nanmin(scores)) if len(scores) > 0 else float('nan'),
        'score_cvar95':       cvar(-scores, alpha=0.05),
        'terminated_rate':    float(np.nanmean(term)),
        'ph_mean':            float(np.nanmean(extract('safety.ph_mean'))),
        'vfa_max_mean':       float(np.nanmean(extract('safety.vfa_max'))),
        'n_episodes':         len(episode_metrics),
    }


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--results-dir', required=True)
    args = parser.parse_args()

    results_dir  = Path(args.results_dir).resolve()
    training_dir = results_dir / 'single_scenario' / 'training'
    per_run_dir  = results_dir / 'single_scenario' / 'evaluation' / 'per_run'
    per_run_dir.mkdir(parents=True, exist_ok=True)

    total = len(TRAIN_SCENARIOS) * len(SEEDS) * len(OBS_MODES) * len(TEST_SCENARIOS)
    done = skipped = errors = 0
    t_start = time.time()

    print(f"=== eval_missing_seeds_cross.py ===")
    print(f"  results_dir    : {results_dir}")
    print(f"  train_scenarios: {TRAIN_SCENARIOS}")
    print(f"  test_scenarios : {TEST_SCENARIOS}")
    print(f"  seeds          : {SEEDS}")
    print(f"  total runs     : {total}")
    print()

    for train_sc in TRAIN_SCENARIOS:
        for seed in SEEDS:
            for obs_mode in OBS_MODES:
                obs_suffix = '_simple' if obs_mode == 'simple' else ''
                run_name   = f'sac_{train_sc}_{REWARD_KEY}_seed{seed}{obs_suffix}'
                model_path = training_dir / run_name / 'best_model' / 'best_model.zip'

                if not model_path.exists():
                    for test_sc in TEST_SCENARIOS:
                        done += 1
                    print(f'  SKIP (no model) {run_name}')
                    continue

                # Load model once per (train_sc, seed, obs_mode)
                model = SAC.load(str(model_path))

                for test_sc in TEST_SCENARIOS:
                    done += 1
                    out_file = per_run_dir / f'{run_name}_on_{test_sc}.json'

                    if out_file.exists():
                        skipped += 1
                        print(f'  SKIP [{done}/{total}]  {run_name}_on_{test_sc}  (exists)')
                        continue

                    elapsed_so_far = time.time() - t_start
                    runs_done = done - skipped - 1
                    eta = (elapsed_so_far / runs_done * (total - done)) if runs_done > 0 else 0
                    print(f'  [{done}/{total}]  {run_name}_on_{test_sc}  ETA={eta/60:.0f}min ... ',
                          end='', flush=True)
                    t0 = time.time()
                    try:
                        env = ADM1Env_v2(
                            scenario_name=test_sc,
                            reward_config=REWARD_CONFIGS[REWARD_KEY],
                            obs_mode=obs_mode,
                        )
                        ep_metrics = [
                            run_episode(model, env, seed=EVAL_SEED_BASE + ep * 100)
                            for ep in range(N_EPISODES)
                        ]
                        env.close()

                        agg = aggregate(ep_metrics)
                        record = {
                            'obs_mode':       obs_mode,
                            'train_scenario': train_sc,
                            'test_scenario':  test_sc,
                            'algo':           'sac',
                            'reward_config':  REWARD_KEY,
                            'seed':           seed,
                            **agg,
                        }
                        out_file.write_text(json.dumps({'record': record, 'raw': agg}, indent=2))
                        print(f"{time.time()-t0:.1f}s  score={agg['overall_score']:.4f}  vr={agg['violation_rate']:.4f}")

                    except Exception:
                        errors += 1
                        print(f'ERROR')
                        traceback.print_exc()

    elapsed = time.time() - t_start
    print(f"\n=== Done: {done} total, {skipped} skipped, {errors} errors in {elapsed/60:.1f} min ===")
    print(f"Next: python analysis/plot_combo.py --results-dir {results_dir}")
    print(f"      python analysis/plot_generalization.py --results-dir {results_dir}")


if __name__ == '__main__':
    main()

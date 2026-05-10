#!/usr/bin/env python3
"""
Re-evaluation script: recompute all per-run SAC result files using the
updated MetricsCalculator (union-based violation_rate, strictly in [0,1]).

Overwrites existing per-run JSON files in-place.

Directories processed (under --results-dir):
  - single_scenario/evaluation/per_run                       (SAC files only)
  - ablation_linear_only/evaluation/per_run
  - ablation_constant_only/evaluation/per_run

Usage:
    python scripts/rerun_eval.py --results-dir /path/to/ADM1/rl/results
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# ── Package root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.adm1_gym_env import ADM1Env_v2
from training.reward_configs import REWARD_CONFIGS
from evaluation.metrics_calculator import MetricsCalculator

from stable_baselines3 import SAC

N_EVAL_EPISODES = 10
NUM_STEPS       = 2880
EVAL_SEED_BASE  = 42


# ── Core evaluation ───────────────────────────────────────────────────────────

def _run_episode(model, env, seed: int) -> dict:
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


def _aggregate(episode_metrics):
    def extract(key_path):
        vals = []
        for m in episode_metrics:
            v = m
            for k in key_path.split('.'):
                v = v.get(k, {}) if isinstance(v, dict) else float('nan')
            vals.append(float(v) if not isinstance(v, dict) else float('nan'))
        return np.array(vals, dtype=float)

    def _cvar(arr, alpha=0.05):
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return float('nan')
        k = max(1, int(np.ceil(alpha * len(arr))))
        return float(np.mean(np.sort(arr)[:k]))

    vfa_rates = extract('safety.violation_rate')
    scores    = extract('summary.overall_score')
    rewards   = extract('reward.mean')
    terminated = extract('episode_info.terminated_early')

    return {
        'reward_mean':        float(np.nanmean(rewards)),
        'reward_std':         float(np.nanstd(rewards, ddof=1)),
        'ch4_avg':            float(np.nanmean(extract('production.avg_ch4_flow'))),
        'ch4_std':            float(np.nanstd(extract('production.avg_ch4_flow'), ddof=1)),
        'violation_rate':     float(np.nanmean(vfa_rates)),
        'violation_rate_std': float(np.nanstd(vfa_rates, ddof=1)),
        'vfa_cvar95':         _cvar(vfa_rates, alpha=0.05),
        'vfa_worst_episode':  float(np.nanmax(vfa_rates)) if len(vfa_rates) > 0 else float('nan'),
        'overall_score':      float(np.nanmean(scores)),
        'score_worst':        float(np.nanmin(scores)) if len(scores) > 0 else float('nan'),
        'score_cvar95':       _cvar(-scores, alpha=0.05),
        'terminated_rate':    float(np.nanmean(terminated)),
        'ph_mean':            float(np.nanmean(extract('safety.ph_mean'))),
        'vfa_max_mean':       float(np.nanmean(extract('safety.vfa_max'))),
        'n_episodes':         len(episode_metrics),
    }


def evaluate_and_save(json_path: Path, training_dir: Path, reward_config_key: str):
    with open(json_path) as f:
        data = json.load(f)
    rec = data.get('record', data)

    algo        = rec.get('algo', 'sac').lower()
    if algo != 'sac':
        return 'skip_not_sac'

    train_sc    = rec['train_scenario']
    test_sc     = rec['test_scenario']
    seed        = rec['seed']
    obs_mode    = rec.get('obs_mode', 'full')
    rc          = rec.get('reward_config', reward_config_key)

    obs_suffix  = '_simple' if obs_mode == 'simple' else ''
    run_name    = f'sac_{train_sc}_{rc}_seed{seed}{obs_suffix}'
    model_path  = training_dir / run_name / 'best_model' / 'best_model.zip'

    if not model_path.exists():
        return f'skip_no_model:{model_path}'

    if rc not in REWARD_CONFIGS:
        return f'skip_unknown_reward_config:{rc}'

    model = SAC.load(str(model_path))
    env   = ADM1Env_v2(
        scenario_name=test_sc,
        reward_config=REWARD_CONFIGS[rc],
        obs_mode=obs_mode,
    )

    episode_metrics = []
    for ep in range(N_EVAL_EPISODES):
        ep_metrics = _run_episode(model, env, seed=EVAL_SEED_BASE + ep * 100)
        episode_metrics.append(ep_metrics)

    env.close()

    aggregated = _aggregate(episode_metrics)

    new_record = {
        'obs_mode':       obs_mode,
        'train_scenario': train_sc,
        'test_scenario':  test_sc,
        'algo':           algo,
        'reward_config':  rc,
        'seed':           seed,
        **aggregated,
    }

    with open(json_path, 'w') as f:
        json.dump({'record': new_record, 'raw': aggregated}, f, indent=2)

    return f"ok  vr={aggregated['violation_rate']:.4f}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Re-evaluate per-run SAC results with fixed MetricsCalculator.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--results-dir', required=True,
        help='Root results directory (contains single_scenario/, etc.).',
    )
    args = parser.parse_args()

    results_base = Path(args.results_dir).resolve()
    targets = [
        {
            'per_run_dir':   results_base / 'single_scenario/evaluation/per_run',
            'training_dir':  results_base / 'single_scenario/training',
            'reward_config': 'safety_first',
        },
        {
            'per_run_dir':   results_base / 'ablation_linear_only/evaluation/per_run',
            'training_dir':  results_base / 'ablation_linear_only/training',
            'reward_config': 'sf_linear_only',
        },
        {
            'per_run_dir':   results_base / 'ablation_constant_only/evaluation/per_run',
            'training_dir':  results_base / 'ablation_constant_only/training',
            'reward_config': 'sf_constant_only',
        },
    ]

    log_path = results_base / 'rerun_eval.log'
    log = open(log_path, 'w', buffering=1)

    def log_print(msg):
        print(msg)
        log.write(msg + '\n')

    log_print(f"=== rerun_eval.py started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log_print(f"  results_dir : {results_base}\n")

    total = done = skipped = errors = 0
    t_start = time.time()

    for target in targets:
        per_run_dir  = target['per_run_dir']
        training_dir = target['training_dir']
        rc_key       = target['reward_config']

        files = sorted(per_run_dir.glob('sac_*.json'))
        log_print(f"\n[{per_run_dir.parent.parent.name}]  {len(files)} SAC files")

        for json_path in files:
            total += 1
            t0 = time.time()
            try:
                status = evaluate_and_save(json_path, training_dir, rc_key)
                elapsed = time.time() - t0
                if status.startswith('skip'):
                    skipped += 1
                    log_print(f"  SKIP  {json_path.name}  ({status})")
                else:
                    done += 1
                    elapsed_total = time.time() - t_start
                    remaining = (elapsed_total / done) * (total - done) if done > 0 else 0
                    log_print(
                        f"  OK    {json_path.name}  {elapsed:.1f}s  "
                        f"[{done}/{total}  ETA {remaining/3600:.1f}h]  {status}"
                    )
            except Exception:
                errors += 1
                log_print(f"  ERR   {json_path.name}")
                log_print(traceback.format_exc())

    elapsed_total = time.time() - t_start
    log_print(
        f"\n=== Done: {done} rerun, {skipped} skipped, {errors} errors "
        f"in {elapsed_total/3600:.2f}h ==="
    )
    log.close()


if __name__ == '__main__':
    main()

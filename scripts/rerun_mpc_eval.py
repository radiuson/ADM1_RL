#!/usr/bin/env python3
"""
MPC / NMPC (oracle) re-evaluation script.

Runs one episode per (controller, scenario, seed) and saves results to:
    <results_dir>/single_scenario/evaluation/per_run/
        mpc_<sc>_seed<N>_on_<sc>.json
        nmpc_oracle_<sc>_seed<N>_on_<sc>.json

File format matches the original paper results (flat dict, no 'record' wrapper).

Usage:
    python scripts/rerun_mpc_eval.py --results-dir /path/to/results_repro
    python scripts/rerun_mpc_eval.py --results-dir /path/to/results_repro --skip-nmpc
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from env.adm1_gym_env import ADM1Env_v2
from training.reward_configs import REWARD_CONFIGS
from evaluation.metrics_calculator import MetricsCalculator

# ── Constants ─────────────────────────────────────────────────────────────────

SCENARIOS = [
    'nominal', 'high_load', 'low_load',
    'shock_load', 'temperature_drop', 'cold_winter',
]
SEEDS    = [42, 123, 456]
REWARD_CONFIG = 'safety_first'
HORIZON  = 4
MAX_ITER = 10


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(ctrl_class, scenario: str, seed: int,
                horizon: int, max_iter: int) -> dict:
    env = ADM1Env_v2(
        scenario_name=scenario,
        reward_config=REWARD_CONFIGS[REWARD_CONFIG],
        obs_mode='full',
    )
    obs, _ = env.reset(seed=seed)
    ctrl   = ctrl_class(env=env, horizon=horizon, max_iter=max_iter, verbose=0)

    calc  = MetricsCalculator()
    done  = False
    steps = 0
    t0    = time.time()

    while not done:
        action = ctrl.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        calc.add_step(obs, action, reward, info)
        steps += 1
        done = terminated or truncated

    duration_s = time.time() - t0
    env.close()

    metrics = calc.compute_metrics()
    summary    = metrics.get('summary',    {})
    production = metrics.get('production', {})
    safety     = metrics.get('safety',     {})
    stability  = metrics.get('stability',  {})

    return {
        'controller':         ctrl.name,
        'train_scenario':     scenario,
        'test_scenario':      scenario,
        'obs_mode':           'full',
        'seed':               seed,
        'horizon':            horizon,
        'max_iter':           max_iter,
        'disturbance_model':  'persistent' if ctrl.name == 'MPC' else 'oracle',
        'episode_steps':      steps,
        'terminated_early':   calc.terminated_early,
        'termination_step':   calc.termination_step,
        'episode_duration_s': round(duration_s, 1),
        'overall_score':      float(summary.get('overall_score', float('nan'))),
        'score_worst':        float(summary.get('overall_score', float('nan'))),
        'score_cvar95':       float(summary.get('overall_score', float('nan'))),
        'avg_ch4_flow':       float(production.get('avg_ch4_flow', float('nan'))),
        'total_ch4_m3':       float(production.get('total_ch4_m3', float('nan'))),
        'violation_rate':     float(safety.get('violation_rate', float('nan'))),
        'ph_violation_count': int(len(calc.ph_violations)),
        'vfa_violation_count':int(len(calc.vfa_violations)),
        'nh3_violation_count':int(len(calc.nh3_violations)),
        'production_score':   float(summary.get('production_score', float('nan'))),
        'safety_score':       float(summary.get('safety_score',     float('nan'))),
        'stability_score':    float(summary.get('stability_score',  float('nan'))),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run MPC and NMPC episodes and save evaluation JSONs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--results-dir', type=pathlib.Path, required=True,
                        help='Root results directory (contains single_scenario/).')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--skip-nmpc', action='store_true',
                        help='Skip NMPC (oracle) — much faster.')
    parser.add_argument('--skip-mpc', action='store_true',
                        help='Skip MPC.')
    parser.add_argument('--horizon',  type=int, default=HORIZON)
    parser.add_argument('--max-iter', type=int, default=MAX_ITER)
    args = parser.parse_args()

    out_dir = (args.results_dir / 'single_scenario' / 'evaluation' / 'per_run')
    out_dir.mkdir(parents=True, exist_ok=True)

    from baselines.mpc_controller  import MPCController
    from baselines.nmpc_controller import NMPCController

    controllers = []
    if not args.skip_mpc:
        controllers.append(('mpc',         MPCController,  'MPC'))
    if not args.skip_nmpc:
        controllers.append(('nmpc_oracle', NMPCController, 'NMPC (oracle)'))

    total = len(controllers) * len(SCENARIOS) * len(args.seeds)
    done_n = 0

    for prefix, ctrl_class, ctrl_label in controllers:
        for sc in SCENARIOS:
            for seed in args.seeds:
                fname  = f'{prefix}_{sc}_seed{seed}_on_{sc}.json'
                fpath  = out_dir / fname

                if fpath.exists():
                    print(f'  SKIP  {fname}  (exists)')
                    done_n += 1
                    continue

                print(f'  [{done_n+1}/{total}]  {ctrl_label}  {sc}  seed{seed} ...', flush=True)
                t0 = time.time()
                try:
                    result = run_episode(ctrl_class, sc, seed,
                                         args.horizon, args.max_iter)
                    fpath.write_text(json.dumps(result, indent=2))
                    elapsed = time.time() - t0
                    print(f'    OK  score={result["overall_score"]:.3f}  '
                          f'viol={result["violation_rate"]:.3f}  '
                          f'{elapsed/60:.1f} min')
                except Exception as e:
                    print(f'    FAIL: {e}')
                done_n += 1

    print(f'\nDone. Results in: {out_dir}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
MPC extended-horizon evaluation.

Evaluates MPCController with configurable horizon/max_iter across all 6 paper
scenarios (single seed=42, 60-day episodes) and appends results to:

    results/baselines_mpc_extended/mpc_H<horizon>_results.json

Usage:
    cd ADM1_RL/
    python scripts/eval_mpc_extended.py --horizon 8 --max-iter 10
    python scripts/eval_mpc_extended.py --horizon 8 --max-iter 10 --scenario nominal
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.adm1_gym_env import ADM1Env_v2
from baselines.mpc_controller import MPCController
from evaluation.metrics_calculator import MetricsCalculator
from training.reward_configs import REWARD_CONFIGS

SCENARIOS   = ['nominal', 'high_load', 'low_load',
               'shock_load', 'temperature_drop', 'cold_winter']
NUM_STEPS   = 5760   # 60 days at 15-min step
REWARD_CFG  = 'safety_first'
SEED        = 42
RESULTS_DIR = ROOT / 'results' / 'baselines_mpc_extended'


def run_episode(scenario: str, horizon: int, max_iter: int) -> dict:
    env  = ADM1Env_v2(
        scenario_name=scenario,
        reward_config=REWARD_CONFIGS[REWARD_CFG],
        obs_mode='full',
    )
    obs, _ = env.reset(seed=SEED)
    ctrl   = MPCController(env=env, horizon=horizon, max_iter=max_iter, verbose=0)
    calc   = MetricsCalculator()

    t0 = time.time()
    for step in range(NUM_STEPS):
        action = ctrl.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        calc.add_step(obs, action, reward, info)
        if step % 96 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (step + 1) * (NUM_STEPS - step - 1)
            print(f'  step {step:4d}/{NUM_STEPS}  VFA={info["total_vfa"]:.3f}  '
                  f'NH3={info.get("S_NH3", float("nan")):.4f}  '
                  f'pH={info["pH"]:.2f}  CH4={info["q_ch4"]:.0f}'
                  f'  ETA={eta/60:.0f}min', flush=True)
        if terminated:
            calc.set_terminated(step)
            print(f'  [TERMINATED at step {step}]', flush=True)
            break
        if truncated:
            break

    env.close()
    m  = calc.compute_metrics()
    ep = m.get('episode_info', {})
    prod   = m['production']
    safety = m['safety']
    summ   = m['summary']
    steps  = m.get('episode_info', {}).get('steps', step + 1)

    vfa_vr  = safety.get('vfa_violation_count',  0) / steps if steps else 0
    nh3_vr  = safety.get('nh3_violation_count',  0) / steps if steps else 0

    rec = {
        'scenario':            scenario,
        'horizon':             horizon,
        'max_iter':            max_iter,
        'seed':                SEED,
        'reward_config':       REWARD_CFG,
        'terminated_early':    calc.terminated_early,
        'termination_step':    calc.termination_step,
        'steps':               steps,
        'overall_score':       float(summ.get('overall_score', float('nan'))),
        'ch4_avg':             float(prod.get('avg_ch4_flow', float('nan'))),
        'ch4_total_m3':        float(prod.get('total_ch4_m3', float('nan'))),
        'violation_rate':      float(safety.get('violation_rate', float('nan'))),
        'vfa_violation_rate':  float(vfa_vr),
        'nh3_violation_rate':  float(nh3_vr),
        'ph_violation_rate':   float(safety.get('ph_violation_count', 0) / steps if steps else 0),
        'vfa_max':             float(safety.get('vfa_max', float('nan'))),
        'nh3_max':             float(safety.get('nh3_max', float('nan'))),
        'ph_mean':             float(safety.get('ph_mean', float('nan'))),
        'production_score':    float(summ.get('production_score', float('nan'))),
        'safety_score':        float(summ.get('safety_score',     float('nan'))),
    }
    return rec


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate MPC with extended prediction horizon.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--horizon',   type=int, default=8)
    parser.add_argument('--max-iter',  type=int, default=10)
    parser.add_argument('--scenario',  type=str, default=None,
                        help='Single scenario (default: all 6)')
    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario else SCENARIOS
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f'mpc_H{args.horizon}_maxiter{args.max_iter}_results.json'

    existing = []
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
    done_keys = {r['scenario'] for r in existing}

    print(f'\n{"="*65}')
    print(f'  MPC Extended Horizon Evaluation')
    print(f'  horizon={args.horizon} steps ({args.horizon*0.25:.1f}h)  max_iter={args.max_iter}')
    print(f'  scenarios={scenarios}')
    print(f'  output → {out_path}')
    print(f'{"="*65}\n')

    new_records = []
    for sc in scenarios:
        if sc in done_keys:
            print(f'  SKIP {sc} (already in results)')
            continue
        print(f'\n--- {sc} ---', flush=True)
        t0 = time.time()
        rec = run_episode(sc, args.horizon, args.max_iter)
        elapsed = time.time() - t0
        print(f'  => score={rec["overall_score"]:.3f}  '
              f'CH4={rec["ch4_avg"]:.0f}  '
              f'VFA_VR={rec["vfa_violation_rate"]:.1%}  '
              f'NH3_VR={rec["nh3_violation_rate"]:.1%}  '
              f'term={rec["terminated_early"]}  '
              f'[{elapsed/60:.1f}min]')
        new_records.append(rec)

        # Save after each scenario (crash-safe)
        with open(out_path, 'w') as f:
            json.dump(existing + new_records, f, indent=2)
        print(f'  saved → {out_path}')

    print(f'\nDone. {len(new_records)} new records saved.')

    # Summary
    all_recs = existing + new_records
    if all_recs:
        print(f'\n{"Scenario":<22} {"CH4":>8} {"VFA_VR":>8} {"NH3_VR":>8} {"Score":>8}  term')
        for r in all_recs:
            print(f'  {r["scenario"]:<20} {r["ch4_avg"]:>8.0f} '
                  f'{r["vfa_violation_rate"]:>7.1%} '
                  f'{r["nh3_violation_rate"]:>7.1%} '
                  f'{r["overall_score"]:>8.3f}  {r["terminated_early"]}')


if __name__ == '__main__':
    main()

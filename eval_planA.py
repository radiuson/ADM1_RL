#!/usr/bin/env python3
"""
Plan A Comprehensive Evaluation
================================
Evaluates MS-SAC, ST-SAC (and optionally single-scenario SAC)
on all 6 paper scenarios, aggregating over 3 seeds.

Produces:
  results/eval_planA/rl_results.csv      — per-seed rows
  results/eval_planA/summary_table.csv   — mean±std over seeds (paper table)
  results/eval_planA/summary_table.json

Usage (from ADM1_RL/ directory):
    python eval_planA.py
    python eval_planA.py --agents ms_sac st_sac
    python eval_planA.py --n-episodes 10 --output-dir results/eval_planA
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from env.adm1_gym_env import ADM1Env_v2
from evaluation.metrics_calculator import MetricsCalculator
from training.reward_configs import REWARD_CONFIGS


PAPER_SCENARIOS = [
    'nominal', 'high_load', 'low_load',
    'shock_load', 'temperature_drop', 'cold_winter',
]
SEEDS = [42, 123, 456]

# Agent configs: (label, model_dir_pattern, reward_config)
AGENT_CONFIGS = {
    'ms_sac': (
        'MS-SAC',
        'models_planA/sac_scenario_cur_safety_first_uniform_random_seed{seed}',
        'safety_first',
    ),
    'st_sac': (
        'ST-SAC',
        'models_planA/sac_scenario_cur_safety_target_uniform_random_seed{seed}',
        'safety_target',
    ),
}


def evaluate_model_on_scenario(
    model_path: Path,
    scenario: str,
    reward_config_name: str,
    n_episodes: int = 10,
    seed: int = 42,
) -> Dict:
    """Run n_episodes with a deterministic SAC policy and return aggregated metrics."""
    model = SAC.load(str(model_path))
    rc = REWARD_CONFIGS[reward_config_name]
    env = ADM1Env_v2(scenario_name=scenario, reward_config=rc, obs_mode='full')

    vr_list, ch4_list, score_list, terminated_list = [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        mc = MetricsCalculator()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            prev_obs = obs
            obs, reward, terminated, truncated, info = env.step(action)
            mc.add_step(prev_obs, action, reward, info)
            if terminated:
                mc.set_terminated(mc.step_count)
            done = terminated or truncated

        m = mc.compute_metrics()
        vr_list.append(m['safety']['violation_rate'])
        ch4_list.append(m['production']['avg_ch4_flow'])
        score_list.append(m['summary']['overall_score'])
        terminated_list.append(m['episode_info']['terminated_early'])

    env.close()
    return {
        'vr_mean':   float(np.mean(vr_list)),
        'vr_std':    float(np.std(vr_list)),
        'ch4_mean':  float(np.mean(ch4_list)),
        'ch4_std':   float(np.std(ch4_list)),
        'score_mean': float(np.mean(score_list)),
        'score_std':  float(np.std(score_list)),
        'term_rate': float(np.mean(terminated_list)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', nargs='+', default=list(AGENT_CONFIGS.keys()),
                        choices=list(AGENT_CONFIGS.keys()))
    parser.add_argument('--scenarios', nargs='+', default=PAPER_SCENARIOS)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--n-episodes', type=int, default=10)
    parser.add_argument('--output-dir', type=str, default='results/eval_planA')
    args = parser.parse_args()

    root = Path(__file__).parent
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for agent_key in args.agents:
        label, pattern, rc_name = AGENT_CONFIGS[agent_key]
        print(f"\n{'='*60}")
        print(f"  Agent: {label}  (reward={rc_name})")
        print(f"{'='*60}")

        # Per-seed × per-scenario results
        seed_scenario_results: Dict[int, Dict[str, Dict]] = {}

        for seed in args.seeds:
            model_dir = root / pattern.format(seed=seed)
            best_zip  = model_dir / 'best_model' / 'best_model.zip'
            final_zip = model_dir / 'final_model.zip'

            if best_zip.exists():
                model_path = best_zip
            elif final_zip.exists():
                model_path = final_zip
            else:
                print(f"  [WARN] No model found at {model_dir} — skipping seed {seed}")
                continue

            print(f"\n  seed={seed}  model={model_path}")
            seed_scenario_results[seed] = {}

            for scenario in args.scenarios:
                print(f"    {scenario:<22s}", end=' ', flush=True)
                res = evaluate_model_on_scenario(
                    model_path, scenario, rc_name,
                    n_episodes=args.n_episodes, seed=seed
                )
                seed_scenario_results[seed][scenario] = res
                print(f"VR={res['vr_mean']*100:5.1f}%  CH4={res['ch4_mean']:6.0f}  "
                      f"score={res['score_mean']:.3f}  term={res['term_rate']*100:.0f}%")

                rows.append({
                    'agent':    label,
                    'seed':     seed,
                    'scenario': scenario,
                    'vr':       res['vr_mean'],
                    'ch4':      res['ch4_mean'],
                    'score':    res['score_mean'],
                    'terminated': res['term_rate'],
                })

        # Aggregate over seeds per scenario
        print(f"\n  --- {label} mean over {len(args.seeds)} seeds ---")
        for scenario in args.scenarios:
            seed_data = [
                seed_scenario_results[s][scenario]
                for s in args.seeds
                if s in seed_scenario_results and scenario in seed_scenario_results[s]
            ]
            if not seed_data:
                continue
            vr_vals    = [d['vr_mean']    for d in seed_data]
            ch4_vals   = [d['ch4_mean']   for d in seed_data]
            score_vals = [d['score_mean'] for d in seed_data]
            term_vals  = [d['term_rate']  for d in seed_data]
            print(f"    {scenario:<22s}  "
                  f"VR={np.mean(vr_vals)*100:5.1f}±{np.std(vr_vals)*100:.1f}%  "
                  f"CH4={np.mean(ch4_vals):6.0f}±{np.std(ch4_vals):.0f}  "
                  f"score={np.mean(score_vals):.3f}±{np.std(score_vals):.3f}  "
                  f"term={np.mean(term_vals)*100:.0f}%")

    # Save per-seed CSV
    df = pd.DataFrame(rows)
    csv_path = out_dir / 'rl_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nPer-seed results → {csv_path}")

    # Build summary table (mean±std over seeds, averaged over all scenarios)
    summary_rows = []
    for agent_key in args.agents:
        label = AGENT_CONFIGS[agent_key][0]
        sub = df[df['agent'] == label]
        for scenario in args.scenarios:
            s = sub[sub['scenario'] == scenario]
            if s.empty:
                continue
            summary_rows.append({
                'agent':    label,
                'scenario': scenario,
                'vr_mean':  s['vr'].mean(),
                'vr_std':   s['vr'].std(),
                'ch4_mean': s['ch4'].mean(),
                'ch4_std':  s['ch4'].std(),
                'score_mean': s['score'].mean(),
                'score_std':  s['score'].std(),
                'term_rate':  s['terminated'].mean(),
            })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(out_dir / 'summary_table.csv', index=False)
    df_sum.to_json(out_dir / 'summary_table.json', orient='records', indent=2)

    # Print macro-average (across all scenarios) for the paper
    print(f"\n{'='*60}")
    print(f"  MACRO-AVERAGE (all 6 scenarios combined)")
    print(f"{'='*60}")
    for agent_key in args.agents:
        label = AGENT_CONFIGS[agent_key][0]
        sub = df_sum[df_sum['agent'] == label]
        print(f"  {label}:  "
              f"VR={sub['vr_mean'].mean()*100:.1f}%  "
              f"CH4={sub['ch4_mean'].mean():.0f} m³/d  "
              f"score={sub['score_mean'].mean():.3f}  "
              f"term={sub['term_rate'].mean()*100:.1f}%")

    print(f"\nSummary → {out_dir}/summary_table.csv")


if __name__ == '__main__':
    main()

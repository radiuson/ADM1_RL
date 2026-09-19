#!/usr/bin/env python3
"""
Reward Ablation Evaluation (Plan A)
=====================================
Evaluates the 2×2 reward ablation:
  - sf_constant_only  (c only,  w=0)  → what if no linear term?
  - sf_linear_only    (w only,  c=0)  → what if no constant term?
  - safety_first      (w + c)         → MS-SAC baseline
  - safety_target     (large w, c=0)  → ST-SAC (proposed)

Ablation scenarios: nominal, high_load, cold_winter  (3 × 3 seeds = 9 models each)

Output:
  results/eval_planA/ablation_results.csv
  results/eval_planA/ablation_summary.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from env.adm1_gym_env import ADM1Env_v2
from evaluation.metrics_calculator import MetricsCalculator
from training.reward_configs import REWARD_CONFIGS


ABLATION_SCENARIOS = ['nominal', 'high_load', 'cold_winter']
SEEDS = [42, 123, 456]

# (display_label, model_dir_pattern, reward_config_name)
ABLATION_CONFIGS = [
    ('Const-only\n(c, w=0)',
     'results/ablation_planA/sac_{scenario}_sf_constant_only_seed{seed}',
     'sf_constant_only'),
    ('Linear-only\n(w, c=0)',
     'results/ablation_planA/sac_{scenario}_sf_linear_only_seed{seed}',
     'sf_linear_only'),
    ('MS-SAC\n(w+c)',
     'models_planA/sac_scenario_cur_safety_first_uniform_random_seed{seed}',
     'safety_first'),
    ('ST-SAC\n(large w, c=0)',
     'models_planA/sac_scenario_cur_safety_target_uniform_random_seed{seed}',
     'safety_target'),
]


def evaluate_model(model_path: Path, scenario: str, rc_name: str,
                   n_episodes: int = 10, seed: int = 42):
    model = SAC.load(str(model_path))
    rc = REWARD_CONFIGS[rc_name]
    env = ADM1Env_v2(scenario_name=scenario, reward_config=rc, obs_mode='full')
    vr_list, ch4_list, score_list, term_list = [], [], [], []
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
        term_list.append(m['episode_info']['terminated_early'])
    env.close()
    return dict(vr=np.mean(vr_list), ch4=np.mean(ch4_list),
                score=np.mean(score_list), term=np.mean(term_list))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', type=int, default=10)
    parser.add_argument('--output-dir', default='results/eval_planA')
    args = parser.parse_args()

    root    = Path(__file__).parent
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, pattern, rc_name in ABLATION_CONFIGS:
        clean_label = label.replace('\n', ' ')
        print(f"\n{'='*55}")
        print(f"  {clean_label}  ({rc_name})")
        print(f"{'='*55}")

        for scenario in ABLATION_SCENARIOS:
            seed_vr, seed_ch4, seed_score, seed_term = [], [], [], []
            for seed in SEEDS:
                model_dir = root / pattern.format(scenario=scenario, seed=seed)
                best = model_dir / 'best_model' / 'best_model.zip'
                final = model_dir / 'final_model.zip'
                path  = best if best.exists() else (final if final.exists() else None)
                if path is None:
                    print(f"  [SKIP] {scenario} seed={seed} — no model at {model_dir}")
                    continue
                res = evaluate_model(path, scenario, rc_name,
                                     n_episodes=args.n_episodes, seed=seed)
                seed_vr.append(res['vr'])
                seed_ch4.append(res['ch4'])
                seed_score.append(res['score'])
                seed_term.append(res['term'])
                rows.append(dict(config=clean_label, scenario=scenario, seed=seed, **res))

            if seed_vr:
                print(f"  {scenario:<18s}  VR={np.mean(seed_vr)*100:5.1f}±{np.std(seed_vr)*100:.1f}%  "
                      f"CH4={np.mean(seed_ch4):6.0f}  score={np.mean(seed_score):.3f}  "
                      f"term={np.mean(seed_term)*100:.0f}%")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'ablation_results.csv', index=False)

    # Summary: mean over seeds per config × scenario
    summary = df.groupby(['config', 'scenario']).agg(
        vr_mean=('vr','mean'), vr_std=('vr','std'),
        ch4_mean=('ch4','mean'), ch4_std=('ch4','std'),
        score_mean=('score','mean'), score_std=('score','std'),
        term_rate=('term','mean'),
    ).reset_index()
    summary.to_csv(out_dir / 'ablation_summary.csv', index=False)

    # Macro-average across 3 scenarios
    print(f"\n{'='*55}")
    print("  MACRO (nominal + high_load + cold_winter avg)")
    print(f"{'='*55}")
    macro = df.groupby('config').agg(VR=('vr','mean'), CH4=('ch4','mean'),
                                      Score=('score','mean'), Term=('term','mean'))
    for name, row in macro.iterrows():
        print(f"  {name:<28s}  VR={row.VR*100:5.1f}%  CH4={row.CH4:6.0f}  "
              f"score={row.Score:.3f}  term={row.Term*100:.0f}%")

    print(f"\n→ {out_dir}/ablation_summary.csv")


if __name__ == '__main__':
    main()

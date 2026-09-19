#!/usr/bin/env python3
"""
Evaluate all (w_VFA, c_VFA) sweep models across 6 scenarios.

Produces:
  results/eval_planA/sweep_results.csv   — per-seed rows
  results/eval_planA/sweep_summary.csv   — mean±std per (config, scenario)
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from env.adm1_gym_env import ADM1Env_v2
from training.reward_configs import REWARD_CONFIGS
from evaluation.metrics_calculator import MetricsCalculator

SCENARIOS = ['nominal', 'high_load', 'low_load', 'cold_winter', 'warm_summer', 'mixed']

# Ordered sweep configs: (label, config_key, w_VFA, c_VFA)
# Reference point is safety_target (w=20, c=0), trained under models_planA
SWEEP_META = [
    # Axis 1 — c=0, varying w (reference w=20 loaded from models_planA)
    ('w=2,  c=0', 'sweep_w2',    2,  0.0),
    ('w=3,  c=0', 'sweep_w3',    3,  0.0),
    ('w=5,  c=0', 'sweep_w5',    5,  0.0),
    ('w=10, c=0', 'sweep_w10',  10,  0.0),
    ('w=15, c=0', 'sweep_w15',  15,  0.0),
    ('w=20, c=0', 'safety_target', 20, 0.0),   # reference (already trained)
    ('w=25, c=0', 'sweep_w25',  25,  0.0),
    ('w=30, c=0', 'sweep_w30',  30,  0.0),
    ('w=40, c=0', 'sweep_w40',  40,  0.0),
    ('w=50, c=0', 'sweep_w50',  50,  0.0),
    ('w=60, c=0', 'sweep_w60',  60,  0.0),
    # Axis 2 — w=20, varying c (reference c=0 = safety_target)
    ('w=20, c=0.25', 'sweep_w20c025', 20, 0.25),
    ('w=20, c=0.50', 'sweep_w20c050', 20, 0.50),
    ('w=20, c=1.00', 'sweep_w20c100', 20, 1.00),
    ('w=20, c=1.50', 'sweep_w20c150', 20, 1.50),
    ('w=20, c=2.00', 'sweep_w20c200', 20, 2.00),
]

# Where each config's models live
def _model_dir(cfg_key: str, seed: int) -> Path:
    if cfg_key == 'safety_target':
        return ROOT / f'models_planA/sac_scenario_cur_safety_target_uniform_random_seed{seed}'
    return ROOT / f'models_sweep/sac_scenario_cur_{cfg_key}_uniform_random_seed{seed}'


def load_model(model_dir: Path):
    from stable_baselines3 import SAC
    for name in ['best_model.zip', 'final_model.zip']:
        p = model_dir / name
        if p.exists():
            return SAC.load(str(p), device='cpu')
    raise FileNotFoundError(f"No model zip in {model_dir}")


def evaluate_one(model, scenario_name: str, reward_cfg: dict,
                 n_episodes: int = 10, max_steps: int = 365) -> dict:
    env = ADM1Env_v2(scenario_name=scenario_name, reward_config=reward_cfg)
    mc = MetricsCalculator()

    vr_list, ch4_list, score_list, term_list = [], [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        mc.reset()
        done = False
        while not done:
            prev_obs = obs
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            mc.add_step(prev_obs, action, reward, info)
            done = term or trunc

        m = mc.compute_metrics()
        vr_list.append(m['safety']['violation_rate'])
        ch4_list.append(m['production']['avg_ch4_flow'])
        score_list.append(m['summary']['overall_score'])
        term_list.append(float(mc.terminated_early))

    env.close()
    return {
        'vr':   np.mean(vr_list),  'vr_std':   np.std(vr_list),
        'ch4':  np.mean(ch4_list), 'ch4_std':  np.std(ch4_list),
        'score':np.mean(score_list),'score_std':np.std(score_list),
        'term': np.mean(term_list),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--n-episodes', type=int, default=10)
    parser.add_argument('--scenarios', nargs='+', default=SCENARIOS)
    parser.add_argument('--skip-missing', action='store_true',
                        help='Skip configs whose model dirs are missing')
    args = parser.parse_args()

    out_dir = ROOT / 'results/eval_planA'
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, cfg_key, w, c in SWEEP_META:
        reward_cfg = REWARD_CONFIGS[cfg_key]
        for seed in args.seeds:
            mdir = _model_dir(cfg_key, seed)
            if not mdir.exists():
                if args.skip_missing:
                    print(f"  [SKIP] {cfg_key} seed{seed} — dir not found")
                    continue
                else:
                    print(f"  [WAIT] {cfg_key} seed{seed} — {mdir} not found, skipping")
                    continue

            print(f"\n{'='*55}")
            print(f"  {label}  (seed {seed})")
            print(f"{'='*55}")
            try:
                model = load_model(mdir)
            except FileNotFoundError as e:
                print(f"  [SKIP] {e}")
                continue

            for sc in args.scenarios:
                res = evaluate_one(model, sc, reward_cfg, n_episodes=args.n_episodes)
                print(f"  {sc:<15} VR={res['vr']*100:5.1f}%  "
                      f"CH4={res['ch4']:6.0f}  score={res['score']:.3f}  "
                      f"term={res['term']*100:.0f}%")
                rows.append({
                    'label':    label,
                    'config':   cfg_key,
                    'w_vfa':    w,
                    'c_vfa':    c,
                    'seed':     seed,
                    'scenario': sc,
                    'vr':       res['vr'],
                    'ch4':      res['ch4'],
                    'score':    res['score'],
                    'term':     res['term'],
                })

    if not rows:
        print("No results — check model directories.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'sweep_results.csv', index=False)
    print(f"\n  Saved {len(df)} rows → sweep_results.csv")

    # Summary: mean±std across seeds
    grp = df.groupby(['label','config','w_vfa','c_vfa','scenario'])
    summary = grp.agg(
        vr_mean=('vr','mean'), vr_std=('vr','std'),
        ch4_mean=('ch4','mean'), ch4_std=('ch4','std'),
        score_mean=('score','mean'), score_std=('score','std'),
        term_rate=('term','mean'),
    ).reset_index()
    summary.to_csv(out_dir / 'sweep_summary.csv', index=False)
    print(f"  Saved → sweep_summary.csv")

    # Quick macro-average table by config
    macro = summary.groupby(['label','w_vfa','c_vfa']).agg(
        VR=('vr_mean','mean'), CH4=('ch4_mean','mean'),
        Score=('score_mean','mean'), Term=('term_rate','mean')
    ).reset_index().sort_values(['c_vfa','w_vfa'])
    print("\nMacro-average sweep results:")
    print(f"  {'Label':<18} {'w':>4} {'c':>5}  VR%    CH4    Score  Term%")
    for _, r in macro.iterrows():
        print(f"  {r['label']:<18} {r['w_vfa']:>4.0f} {r['c_vfa']:>5.2f}  "
              f"{r['VR']*100:5.1f}  {r['CH4']:6.0f}  {r['Score']:.3f}  {r['Term']*100:.0f}")


if __name__ == '__main__':
    main()

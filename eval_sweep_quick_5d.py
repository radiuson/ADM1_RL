#!/usr/bin/env python3
"""
Evaluate 5D sweep models (obs_mode='simple') on 3 scenarios.

Scans models_sweep_5d/, evaluates on nominal / high_load / cold_winter,
writes results/eval_planA/sweep_quick_5d.csv.

Safe to re-run incrementally — skips already-evaluated (config, seed) pairs.
"""
import sys, argparse, re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from env.adm1_gym_env import ADM1Env_v2
from training.reward_configs import REWARD_CONFIGS
from evaluation.metrics_calculator import MetricsCalculator

SCENARIOS = ['nominal', 'high_load', 'cold_winter']
OUT_CSV   = ROOT / 'results/eval_planA/sweep_quick_5d.csv'

W_C_MAP = {
    'sweep_w2':     ( 2, 0.0), 'sweep_w3':     ( 3, 0.0),
    'sweep_w5':     ( 5, 0.0), 'sweep_w10':    (10, 0.0),
    'sweep_w11':    (11, 0.0), 'sweep_w12':    (12, 0.0),
    'sweep_w13':    (13, 0.0), 'sweep_w14':    (14, 0.0),
    'sweep_w15':    (15, 0.0), 'sweep_w16':    (16, 0.0),
    'sweep_w18':    (18, 0.0), 'sweep_w22':    (22, 0.0),
    'sweep_w25':    (25, 0.0),
    'sweep_w30':    (30, 0.0), 'sweep_w40':    (40, 0.0),
    'sweep_w50':    (50, 0.0), 'sweep_w60':    (60, 0.0),
    'sweep_w20c025':(20, 0.25),'sweep_w20c050':(20, 0.50),
    'sweep_w20c100':(20, 1.00),'sweep_w20c150':(20, 1.50),
    'sweep_w20c200':(20, 2.00),
}


def find_model(run_dir: Path):
    for candidate in [
        run_dir / 'final_model.zip',
        run_dir / 'best_model' / 'best_model.zip',
    ]:
        if candidate.exists():
            return candidate
    return None


def scan_models(models_dir: Path):
    found = []
    for d in sorted(models_dir.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r'sac_scenario_cur_(.+)_uniform_random_seed(\d+)$', d.name)
        if not m:
            continue
        cfg_key = m.group(1)
        seed    = int(m.group(2))
        zip_p   = find_model(d)
        if zip_p:
            found.append((cfg_key, seed, zip_p))
    return found


def evaluate(zip_path: Path, reward_cfg: dict, n_ep: int = 5) -> dict:
    from stable_baselines3 import SAC
    model = SAC.load(str(zip_path), device='cpu')
    rows  = {}
    for sc in SCENARIOS:
        env = ADM1Env_v2(scenario_name=sc, reward_config=reward_cfg, obs_mode='simple')
        mc  = MetricsCalculator()
        vr_l, ch4_l, sc_l, tr_l = [], [], [], []
        for _ in range(n_ep):
            obs, _ = env.reset()
            mc.reset()
            done = False
            while not done:
                prev = obs
                act, _ = model.predict(obs, deterministic=True)
                obs, rew, t, tr, info = env.step(act)
                mc.add_step(prev, act, rew, info)
                done = t or tr
            m2 = mc.compute_metrics()
            vr_l.append(m2['safety']['violation_rate'])
            ch4_l.append(m2['production']['avg_ch4_flow'])
            sc_l.append(m2['summary']['overall_score'])
            tr_l.append(float(mc.terminated_early))
        env.close()
        rows[sc] = dict(vr=np.mean(vr_l), ch4=np.mean(ch4_l),
                        score=np.mean(sc_l), term=np.mean(tr_l))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-episodes', type=int, default=5)
    ap.add_argument('--models-dir', default='models_sweep_5d')
    args = ap.parse_args()

    models_dir = ROOT / args.models_dir
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    if OUT_CSV.exists():
        existing  = pd.read_csv(OUT_CSV)
        done_keys = set(zip(existing['config'], existing['seed']))
    else:
        existing  = pd.DataFrame()
        done_keys = set()

    candidates = scan_models(models_dir)
    print(f"Found {len(candidates)} models in {models_dir}")

    new_rows = []
    for cfg_key, seed, zip_path in candidates:
        if (cfg_key, seed) in done_keys:
            print(f"  [skip] {cfg_key} seed{seed}")
            continue
        if cfg_key not in W_C_MAP:
            print(f"  [skip] {cfg_key} — not in W_C_MAP")
            continue
        w, c = W_C_MAP[cfg_key]
        reward_cfg = REWARD_CONFIGS.get(cfg_key)
        if reward_cfg is None:
            print(f"  [skip] {cfg_key} — not in REWARD_CONFIGS")
            continue

        print(f"  eval  {cfg_key} seed{seed} ...", flush=True)
        try:
            sc_results = evaluate(zip_path, reward_cfg, n_ep=args.n_episodes)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        for sc, metrics in sc_results.items():
            new_rows.append({
                'config': cfg_key, 'w_vfa': w, 'c_vfa': c,
                'seed': seed, 'scenario': sc, 'model_type': 'final',
                **metrics,
            })

    if new_rows:
        combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
        combined.to_csv(OUT_CSV, index=False)
        print(f"\nSaved {len(new_rows)} new rows → {OUT_CSV}")
    else:
        print("No new results to save.")


if __name__ == '__main__':
    main()

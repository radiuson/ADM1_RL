#!/usr/bin/env python3
"""
统一评估脚本：评估所有新训练的模型（PPO、SAC-Cur+FK、Phase3、Phase4）。

每个模型在全部6个场景上做交叉评估（10 episodes），结果追加写入：
  results/evaluation/cross_scenario_results.csv / .json

用法（从 ADM1_RL/ 目录运行）：
    python scripts/eval_all_new_models.py                     # 评估所有
    python scripts/eval_all_new_models.py --only ppo
    python scripts/eval_all_new_models.py --only sac_cur_fk
    python scripts/eval_all_new_models.py --only phase3
    python scripts/eval_all_new_models.py --only phase4
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
NUM_STEPS   = 5760
EVAL_SEED   = 42
MODELS_DIR  = ROOT / 'models'
RESULTS_DIR = ROOT / 'results' / 'evaluation'


def flat_record(train_sc, test_sc, algo, reward_config, seed, metrics, curriculum=False) -> dict:
    """把 evaluate_model_on_scenario 返回的平铺 metrics 转为 CSV 行。"""
    def g(key):
        v = metrics.get(key, float('nan'))
        return float(v) if v is not None else float('nan')

    return {
        'obs_mode':           'full',
        'train_scenario':     train_sc,
        'test_scenario':      test_sc,
        'algo':               algo,
        'reward_config':      reward_config,
        'curriculum':         curriculum,
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


def eval_group(group_name, model_entries, existing_keys, curriculum=False):
    """
    model_entries: list of (train_sc, algo, reward_config, seed, model_path)
    existing_keys: set of (train_sc, test_sc, algo, reward_config, curriculum, seed) already in results
    Returns list of new records.
    """
    total = len(model_entries) * len(SCENARIOS)
    done = 0
    new_records = []

    print(f"\n{'#'*65}")
    print(f"  GROUP: {group_name}  ({len(model_entries)} models × 6 scenarios = {total} evals)")
    print(f"{'#'*65}")

    for (train_sc, algo, rc, seed, model_path) in model_entries:
        if not Path(str(model_path) + '.zip').exists() and not Path(model_path).exists():
            print(f"  [SKIP] model not found: {model_path}")
            continue

        for test_sc in SCENARIOS:
            done += 1
            key = (train_sc, test_sc, algo, rc, curriculum, seed)
            if key in existing_keys:
                print(f"  [{done:3d}/{total}] SKIP (already exists): {algo} {train_sc}→{test_sc} seed={seed}")
                continue

            t0 = time.time()
            print(f"  [{done:3d}/{total}] {algo:4s} {train_sc:<18s} → {test_sc:<18s} seed={seed} ... ",
                  end='', flush=True)
            try:
                metrics = evaluate_model_on_scenario(
                    model_path=str(model_path),
                    test_scenario=test_sc,
                    reward_config_name=rc,
                    num_steps=NUM_STEPS,
                    n_eval_episodes=N_EPISODES,
                    seed=EVAL_SEED,
                    algo=algo,
                    obs_mode='full',
                )
                rec = flat_record(train_sc, test_sc, algo, rc, seed, metrics, curriculum=curriculum)
                new_records.append(rec)
                score = rec['overall_score']
                term  = rec['terminated_rate']
                print(f"score={score:.3f}  term={term:.0%}  [{time.time()-t0:.0f}s]")
            except Exception as e:
                print(f"ERROR: {e}")

    return new_records


def save_results(new_records, results_dir):
    """追加写入 CSV 和 JSON。"""
    import csv

    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = results_dir / 'cross_scenario_results.csv'
    json_path = results_dir / 'cross_scenario_results.json'

    # Load existing JSON
    existing = []
    if json_path.exists():
        with open(json_path) as f:
            existing = json.load(f)

    combined = existing + new_records

    # Write JSON
    with open(json_path, 'w') as f:
        json.dump(combined, f, indent=2)

    # Write CSV (use union of all keys to handle records with/without 'curriculum')
    if combined:
        fieldnames = list(dict.fromkeys(k for r in combined for k in r.keys()))
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for row in combined:
                writer.writerow({k: row.get(k, '') for k in fieldnames})

    print(f"\n  Saved {len(new_records)} new records → {csv_path}")
    print(f"  Total records: {len(combined)}")


def load_existing_keys(results_dir):
    json_path = results_dir / 'cross_scenario_results.json'
    if not json_path.exists():
        return set()
    with open(json_path) as f:
        data = json.load(f)
    return {(r['train_scenario'], r['test_scenario'], r['algo'],
             r['reward_config'], r.get('curriculum', False), r['seed']) for r in data}


def build_sac_baseline_entries():
    entries = []
    for sc in SCENARIOS:
        for seed in SEEDS:
            mp = MODELS_DIR / f'sac_{sc}_safety_first_seed{seed}' / 'final_model'
            entries.append((sc, 'sac', 'safety_first', seed, mp))
    return entries


def build_ppo_entries():
    entries = []
    for sc in SCENARIOS:
        for seed in SEEDS:
            mp = MODELS_DIR / f'ppo_{sc}_safety_first_seed{seed}' / 'final_model'
            entries.append((sc, 'ppo', 'safety_first', seed, mp))
    return entries


def build_sac_cur_fk_entries():
    entries = []
    for sc in SCENARIOS:
        for seed in SEEDS:
            mp = MODELS_DIR / f'sac_{sc}_safety_first_curriculum_seed{seed}_curriculum' / 'final_model'
            entries.append((sc, 'sac', 'safety_first_curriculum', seed, mp))
    return entries


def build_phase3_entries():
    # SAC + Curriculum ONLY (safety_first reward, _curriculum suffix)
    entries = []
    for sc in SCENARIOS:
        for seed in SEEDS:
            mp = MODELS_DIR / f'sac_{sc}_safety_first_seed{seed}_curriculum' / 'final_model'
            entries.append((sc, 'sac', 'safety_first', seed, mp))
    return entries


def build_phase4_entries():
    # SAC + F_K ONLY (safety_first_curriculum reward, no _curriculum suffix)
    entries = []
    for sc in SCENARIOS:
        for seed in SEEDS:
            mp = MODELS_DIR / f'sac_{sc}_safety_first_curriculum_seed{seed}' / 'final_model'
            entries.append((sc, 'sac', 'safety_first_curriculum', seed, mp))
    return entries


def build_ppo_cur_fk_entries():
    # PPO + Curriculum + F_K (safety_first_curriculum reward + _curriculum suffix)
    entries = []
    for sc in SCENARIOS:
        for seed in SEEDS:
            mp = MODELS_DIR / f'ppo_{sc}_safety_first_curriculum_seed{seed}_curriculum' / 'final_model'
            entries.append((sc, 'ppo', 'safety_first_curriculum', seed, mp))
    return entries


def build_ppo_cur_only_entries():
    # PPO + Curriculum ONLY (safety_first reward + _curriculum suffix)
    entries = []
    for sc in SCENARIOS:
        for seed in SEEDS:
            mp = MODELS_DIR / f'ppo_{sc}_safety_first_seed{seed}_curriculum' / 'final_model'
            entries.append((sc, 'ppo', 'safety_first', seed, mp))
    return entries


def build_ppo_fk_only_entries():
    # PPO + F_K ONLY (safety_first_curriculum reward, no _curriculum suffix)
    entries = []
    for sc in SCENARIOS:
        for seed in SEEDS:
            mp = MODELS_DIR / f'ppo_{sc}_safety_first_curriculum_seed{seed}' / 'final_model'
            entries.append((sc, 'ppo', 'safety_first_curriculum', seed, mp))
    return entries


def main():
    parser = argparse.ArgumentParser(
        description='评估所有新模型（PPO、SAC-Cur+FK、Phase3、Phase4、PPO 2x2）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--only', choices=[
        'ppo', 'sac_baseline', 'sac_cur_fk', 'phase3', 'phase4', 'ppo_cur_fk',
        'ppo_cur_only', 'ppo_fk_only', 'all'],
                        default='all')
    parser.add_argument('--results-dir', default=str(RESULTS_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    existing_keys = load_existing_keys(results_dir)
    print(f"已有评估记录: {len(existing_keys)} 条")

    groups = {
        'sac_baseline': ('SAC Baseline',          build_sac_baseline_entries,  False),
        'ppo':          ('PPO Baseline',          build_ppo_entries,           False),
        'sac_cur_fk':   ('SAC-Cur+FK',           build_sac_cur_fk_entries,    True),
        'phase3':       ('SAC-CurOnly',          build_phase3_entries,        True),
        'phase4':       ('SAC-FKOnly',           build_phase4_entries,        False),
        'ppo_cur_fk':   ('PPO-Cur+FK',            build_ppo_cur_fk_entries,    True),
        'ppo_cur_only': ('PPO-CurOnly',           build_ppo_cur_only_entries,  True),
        'ppo_fk_only':  ('PPO-FKOnly',            build_ppo_fk_only_entries,   False),
    }

    to_run = list(groups.keys()) if args.only == 'all' else [args.only]
    all_new = []

    for key in to_run:
        name, builder, cur_flag = groups[key]
        entries = builder()
        # filter: only entries where model exists
        valid = [(sc, algo, rc, seed, mp) for (sc, algo, rc, seed, mp) in entries
                 if Path(str(mp) + '.zip').exists()]
        if not valid:
            print(f"\n  [{name}] 暂无可用模型，跳过")
            continue
        print(f"\n  [{name}] 找到 {len(valid)}/{len(entries)} 个模型")
        recs = eval_group(name, valid, existing_keys, curriculum=cur_flag)
        all_new.extend(recs)
        # 每组结束后立即保存
        if recs:
            save_results(recs, results_dir)
            existing_keys = load_existing_keys(results_dir)

    print(f"\n{'='*65}")
    print(f"  全部完成，共新增 {len(all_new)} 条评估记录")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()

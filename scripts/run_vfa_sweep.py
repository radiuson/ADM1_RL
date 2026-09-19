#!/usr/bin/env python3
"""
VFA penalty sensitivity sweep — two axes:
  Plan A: c_VFA=0, w_VFA in {5,10,15,25,30}  (ST-SAC w=20 is reference)
  Plan B: w_VFA=10, c_VFA in {0,0.25,0.5,1.5} (MS-SAC c=1.0 is reference)

Runs in two batches of 12 parallel jobs each (~101 min / batch).
After both batches, evaluates all new models on 6 test scenarios.

Usage:
    cd /home/ihpc/code/biogas/ADM1_RL
    nohup python scripts/run_vfa_sweep.py > /tmp/vfa_sweep.log 2>&1 &
"""

import json
import multiprocessing as mp
import pathlib
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCENARIOS = ['nominal', 'high_load', 'low_load', 'shock_load', 'temperature_drop', 'cold_winter']
SEEDS     = [42, 123, 456]
TOTAL_STEPS = 500_000
OUT_DIR   = ROOT / 'results' / 'sweep_vfa'

# Plan A: c=0, varying w
PLAN_A = [
    ('sweep_w5',  'A'),
    ('sweep_w10', 'A'),
    ('sweep_w15', 'A'),
    ('sweep_w25', 'A'),
    ('sweep_w30', 'A'),
]
# Plan B: w=10, varying c (sweep_w10 doubles as c=0)
PLAN_B = [
    ('sweep_w10',  'B'),   # c=0 baseline, reuses Plan A model
    ('sweep_c025', 'B'),
    ('sweep_c050', 'B'),
    ('sweep_c150', 'B'),
]

ALL_CONFIGS = list(dict.fromkeys([rc for rc, _ in PLAN_A + PLAN_B]))  # unique, ordered


def train_one(args):
    reward_config, seed, output_dir_str = args
    import sys, pathlib as _pl, time, json, traceback
    sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    from training.run_experiment import train_single

    output_dir = _pl.Path(output_dir_str)
    run_name   = f'sac_allscen_{reward_config}_seed{seed}'
    run_dir    = output_dir / 'training' / run_name

    if (run_dir / 'best_model' / 'best_model.zip').exists():
        print(f'[SKIP] {run_name} already trained', flush=True)
        return run_name

    print(f'[START] {run_name}', flush=True)
    t0 = time.time()
    try:
        train_single(
            scenario           = ['nominal','high_load','low_load','shock_load','temperature_drop','cold_winter'],
            reward_config_name = reward_config,
            seed               = seed,
            algo               = 'sac',
            total_timesteps    = TOTAL_STEPS,
            output_dir         = output_dir,
            obs_mode           = 'full',
            hyperparams_cfg    = {},
            eval_freq          = 10_000,
            n_eval_episodes    = 5,
            verbose            = 0,
        )
        elapsed = time.time() - t0
        meta = {'reward_config': reward_config, 'seed': seed, 'elapsed_seconds': round(elapsed, 1)}
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / 'run_meta.json').write_text(json.dumps(meta, indent=2))
        print(f'[DONE]  {run_name}  {elapsed/60:.1f} min', flush=True)
    except Exception:
        print(f'[ERR]   {run_name}', flush=True)
        traceback.print_exc()
    return run_name


def eval_all(output_dir):
    from training.run_experiment import evaluate_model_on_scenario
    results = []
    total = len(ALL_CONFIGS) * len(SEEDS) * len(SCENARIOS)
    idx = 0
    for rc in ALL_CONFIGS:
        for seed in SEEDS:
            model_path = output_dir / 'training' / f'sac_allscen_{rc}_seed{seed}' / 'best_model' / 'best_model'
            if not (model_path.parent / 'best_model.zip').exists():
                print(f'[EVAL SKIP] {rc} seed={seed} — model not found', flush=True)
                continue
            for test_sc in SCENARIOS:
                idx += 1
                t0 = time.time()
                try:
                    m = evaluate_model_on_scenario(
                        model_path=str(model_path), test_scenario=test_sc,
                        reward_config_name=rc, num_steps=2880, n_eval_episodes=10,
                        seed=42, algo='sac', obs_mode='full',
                    )
                    results.append({
                        'reward_config': rc, 'seed': seed, 'test_scenario': test_sc,
                        'violation_rate': m.get('violation_rate'), 'ch4_avg': m.get('ch4_avg'),
                        'overall_score':  m.get('overall_score'),
                        'terminated_rate': m.get('terminated_rate'),
                    })
                    print(f'[EVAL {idx:3d}/{total}] {rc} seed={seed} → {test_sc:<20} '
                          f'VR={m.get("violation_rate",0)*100:.1f}%  '
                          f'CH4={m.get("ch4_avg",0):.0f}  [{time.time()-t0:.0f}s]', flush=True)
                except Exception:
                    print(f'[EVAL ERR] {rc} seed={seed} {test_sc}', flush=True)
                    traceback.print_exc()

    out_file = output_dir / 'sweep_results.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved {len(results)} eval records → {out_file}', flush=True)
    return results


def run_batch(jobs, batch_name, n_workers=12):
    print(f'\n{"="*60}', flush=True)
    print(f'  {batch_name}: {len(jobs)} jobs, {n_workers} parallel workers', flush=True)
    print(f'{"="*60}', flush=True)
    t0 = time.time()
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=n_workers) as pool:
        pool.map(train_one, jobs)
    elapsed = time.time() - t0
    print(f'\n{batch_name} done in {elapsed/60:.1f} min', flush=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build all training jobs
    all_jobs = [(rc, seed, str(OUT_DIR)) for rc in ALL_CONFIGS for seed in SEEDS]

    print(f'VFA Sweep — {len(ALL_CONFIGS)} configs × {len(SEEDS)} seeds = {len(all_jobs)} training jobs')
    print(f'Output: {OUT_DIR}')
    print(f'Configs: {ALL_CONFIGS}')
    print()

    # Batch 1: first 12 jobs
    batch1 = all_jobs[:12]
    batch2 = all_jobs[12:]

    run_batch(batch1, 'Batch 1 (12 jobs)', n_workers=12)
    run_batch(batch2, 'Batch 2 (12 jobs)', n_workers=12)

    print('\n=== All training done. Starting evaluation ===', flush=True)
    results = eval_all(OUT_DIR)

    # Print summary
    import numpy as np
    print('\n=== SWEEP SUMMARY ===')
    for rc in ALL_CONFIGS:
        recs = [r for r in results if r['reward_config'] == rc]
        if recs:
            # mean-of-scenario-means
            by_sc = {}
            for r in recs:
                by_sc.setdefault(r['test_scenario'], []).append(r)
            avg_vr  = np.mean([np.mean([x['violation_rate'] for x in v]) for v in by_sc.values()])
            avg_ch4 = np.mean([np.mean([x['ch4_avg']        for x in v]) for v in by_sc.values()])
            avg_sc  = np.mean([np.mean([x['overall_score']  for x in v]) for v in by_sc.values()])
            print(f'  {rc:<12}: VR={avg_vr*100:.1f}%  CH4={avg_ch4:.0f}  score={avg_sc:.3f}')

    print('\nDone.', flush=True)


if __name__ == '__main__':
    main()

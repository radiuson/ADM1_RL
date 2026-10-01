#!/usr/bin/env python3
"""
Evaluate all VFA sweep models and produce sweep_results.json.

The sweep training script (run_vfa_sweep.py) saves models under the full
scenario-list name:
  sac_nominal_high_load_low_load_shock_load_temperature_drop_cold_winter_{rc}_seed{seed}/

This script uses the correct paths (unlike the broken eval_all in run_vfa_sweep.py
which wrongly looks for sac_allscen_* directories).

Usage:
    cd /home/ihpc/code/biogas/ADM1_RL
    nohup python scripts/eval_sweep_results.py > /tmp/eval_sweep.log 2>&1 &
"""

import json
import pathlib
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCENARIOS  = ['nominal', 'high_load', 'low_load', 'shock_load', 'temperature_drop', 'cold_winter']
SEEDS      = [42, 123, 456]
SWEEP_DIR  = ROOT / 'results' / 'sweep_vfa'
TRAIN_DIR  = SWEEP_DIR / 'training'

# same order as run_vfa_sweep.py ALL_CONFIGS
ALL_CONFIGS = ['sweep_w5', 'sweep_w10', 'sweep_w15', 'sweep_w25',
               'sweep_w30', 'sweep_c025', 'sweep_c050', 'sweep_c150']

SCENARIO_KEY = 'nominal_high_load_low_load_shock_load_temperature_drop_cold_winter'


def model_path_for(rc, seed):
    run_name = f'sac_{SCENARIO_KEY}_{rc}_seed{seed}'
    return TRAIN_DIR / run_name / 'best_model' / 'best_model'


def main():
    from training.run_experiment import evaluate_model_on_scenario

    results = []
    total = len(ALL_CONFIGS) * len(SEEDS) * len(SCENARIOS)
    idx = 0

    for rc in ALL_CONFIGS:
        for seed in SEEDS:
            mp = model_path_for(rc, seed)
            if not (mp.parent / 'best_model.zip').exists():
                print(f'[SKIP] {rc} seed={seed} — not found at {mp}', flush=True)
                continue
            for test_sc in SCENARIOS:
                idx += 1
                t0 = time.time()
                try:
                    m = evaluate_model_on_scenario(
                        model_path=str(mp),
                        test_scenario=test_sc,
                        reward_config_name=rc,
                        num_steps=2880,
                        n_eval_episodes=10,
                        seed=42,
                        algo='sac',
                        obs_mode='full',
                    )
                    results.append({
                        'reward_config': rc,
                        'seed': seed,
                        'test_scenario': test_sc,
                        'violation_rate': m.get('violation_rate'),
                        'ch4_avg': m.get('ch4_avg'),
                        'overall_score': m.get('overall_score'),
                        'terminated_rate': m.get('terminated_rate'),
                    })
                    elapsed = time.time() - t0
                    print(
                        f'[EVAL {idx:3d}/{total}] {rc} seed={seed} → {test_sc:<22}'
                        f'VR={m.get("violation_rate", 0)*100:.1f}%  '
                        f'CH4={m.get("ch4_avg", 0):.0f}  [{elapsed:.0f}s]',
                        flush=True,
                    )
                except Exception:
                    print(f'[ERR] {rc} seed={seed} {test_sc}', flush=True)
                    traceback.print_exc()

    out_file = SWEEP_DIR / 'sweep_results.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved {len(results)} records → {out_file}', flush=True)

    # Print summary
    import numpy as np
    print('\n=== SWEEP SUMMARY ===')
    for rc in ALL_CONFIGS:
        recs = [r for r in results if r['reward_config'] == rc]
        if not recs:
            print(f'  {rc:<12}: no data')
            continue
        by_sc = {}
        for r in recs:
            by_sc.setdefault(r['test_scenario'], []).append(r)
        avg_vr  = np.mean([np.mean([x['violation_rate'] for x in v]) for v in by_sc.values()])
        avg_ch4 = np.mean([np.mean([x['ch4_avg']        for x in v]) for v in by_sc.values()])
        avg_sc  = np.mean([np.mean([x['overall_score']  for x in v]) for v in by_sc.values()])
        print(f'  {rc:<12}: VR={avg_vr*100:.1f}%  CH4={avg_ch4:.0f}  score={avg_sc:.3f}')

    print('\nDone.')


if __name__ == '__main__':
    main()

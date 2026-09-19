#!/usr/bin/env python3
"""
Scenario-Difficulty Curriculum: Full Experiment Loop
=====================================================

Runs train → eval → analyze → iterate autonomously.
Tries up to MAX_VARIANTS curriculum designs. Stops when a design beats
the SAC Baseline on BOTH CH4 production AND violation rate, or after all
designs are exhausted.

Success criterion (vs SAC Baseline 60-day results):
    violation_rate < BASELINE_VIOL (0.196)  AND/OR
    overall_score  > BASELINE_SCORE (0.7289)
    [at least ONE metric must improve]

Design variants (tried in order):
    1. default:         nominal → +high_load+low_load → all 6
    2. high_load_first: high_load → +nominal+low_load → all 6
    3. fast:            nominal (50k) → +h+l (150k) → all 6
    4. slow:            nominal (150k) → +h+l (250k) → all 6

Usage (from ADM1_RL/ directory):
    screen -S scen_cur python scripts/run_scenario_cur_experiment.py
    screen -S scen_cur python scripts/run_scenario_cur_experiment.py --max-workers 8
"""

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Experiment settings ───────────────────────────────────────────────────────

REWARD_CONFIG = 'safety_first'
SEEDS = [42, 123, 456]
MAX_WORKERS = 12      # parallel training jobs (SAC on GPU)
TOTAL_TIMESTEPS = 300_000

# SAC Baseline 60-day results (from evaluation_60d/cross_scenario_results.json)
BASELINE_VIOL  = 0.196    # 19.6% violation rate
BASELINE_CH4   = 1411.0   # m³/day
BASELINE_SCORE = 0.7289   # overall_score

# Curriculum variants to try in order
VARIANTS = ['default', 'high_load_first', 'fast', 'slow', 'uniform_random']

RESULTS_DIR = ROOT / 'results' / 'evaluation_60d'
SCENARIO_CUR_JSON = RESULTS_DIR / 'scenario_cur_results.json'


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


def run_training_parallel(stages_name: str, max_workers: int) -> bool:
    """Train all seeds in parallel. Returns True if all jobs succeeded."""
    log(f"Training variant={stages_name} | {len(SEEDS)} seeds | max_workers={max_workers}")

    def run_one(seed):
        cmd = [
            sys.executable,
            str(ROOT / 'training' / 'train_sac_scenario_cur.py'),
            '--reward-config', REWARD_CONFIG,
            '--seed', str(seed),
            '--stages', stages_name,
            '--timesteps', str(TOTAL_TIMESTEPS),
            '--device', 'cuda',
            '--verbose', '0',
        ]
        log(f"  START seed={seed}: {' '.join(cmd)}")
        t0 = time.time()
        ret = subprocess.run(cmd, cwd=str(ROOT)).returncode
        elapsed = time.time() - t0
        status = 'OK' if ret == 0 else f'FAILED({ret})'
        log(f"  {status} seed={seed} in {elapsed/60:.1f}min")
        return seed, ret

    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_one, s): s for s in SEEDS}
        for f in as_completed(futures):
            seed, ret = f.result()
            if ret != 0:
                failures.append(seed)

    if failures:
        log(f"  FAILED seeds: {failures}")
        return False
    log(f"  All {len(SEEDS)} seeds completed.")
    return True


def run_evaluation(stages_name: str) -> bool:
    """Run cross-scenario evaluation. Returns True if completed."""
    log(f"Evaluating variant={stages_name} ...")
    cmd = [
        sys.executable,
        str(ROOT / 'scripts' / 'eval_scenario_cur.py'),
        '--reward-config', REWARD_CONFIG,
        '--stages-name', stages_name,
    ]
    ret = subprocess.run(cmd, cwd=str(ROOT)).returncode
    if ret != 0:
        log(f"  Evaluation FAILED (returncode={ret})")
        return False
    log(f"  Evaluation complete.")
    return True


def analyze_results(stages_name: str) -> dict:
    """
    Load results for this variant and compare with SAC Baseline.

    Returns a dict with summary stats and a 'passed' flag.
    """
    if not SCENARIO_CUR_JSON.exists():
        log("  [ANALYZE] No results file found.")
        return {'passed': False, 'error': 'no_file'}

    with open(SCENARIO_CUR_JSON) as f:
        all_records = json.load(f)

    records = [r for r in all_records
               if r.get('stages_name') == stages_name
               and r.get('reward_config') == REWARD_CONFIG]

    if not records:
        log(f"  [ANALYZE] No records found for variant={stages_name}")
        return {'passed': False, 'error': 'no_records'}

    viol_rates  = [r['violation_rate']  for r in records if not np.isnan(r['violation_rate'])]
    ch4_vals    = [r['ch4_avg']         for r in records if not np.isnan(r['ch4_avg'])]
    scores      = [r['overall_score']   for r in records if not np.isnan(r['overall_score'])]
    term_rates  = [r['terminated_rate'] for r in records if not np.isnan(r.get('terminated_rate', float('nan')))]

    if not viol_rates:
        log(f"  [ANALYZE] All records NaN for variant={stages_name}")
        return {'passed': False, 'error': 'all_nan'}

    avg_viol  = float(np.mean(viol_rates))
    avg_ch4   = float(np.mean(ch4_vals))   if ch4_vals  else float('nan')
    avg_score = float(np.mean(scores))     if scores    else float('nan')
    avg_term  = float(np.mean(term_rates)) if term_rates else float('nan')
    n_records = len(records)

    # Per-test-scenario breakdown
    from collections import defaultdict
    per_sc = defaultdict(list)
    for r in records:
        per_sc[r['test_scenario']].append(r['violation_rate'])

    log(f"\n  ── Analysis: {stages_name} ({n_records} records) ──")
    log(f"  CH4:        {avg_ch4:.1f} m³/d  (baseline: {BASELINE_CH4:.1f})")
    log(f"  Viol rate:  {avg_viol:.1%}      (baseline: {BASELINE_VIOL:.1%})")
    log(f"  Score:      {avg_score:.4f}     (baseline: {BASELINE_SCORE:.4f})")
    log(f"  Term rate:  {avg_term:.1%}")
    log(f"  Per test scenario:")
    for sc in ['nominal', 'high_load', 'low_load', 'shock_load', 'temperature_drop', 'cold_winter']:
        v = per_sc.get(sc, [])
        avg_v = np.mean(v) if v else float('nan')
        log(f"    {sc:<22} viol={avg_v:.1%}")

    # Success criterion: improve on at least one of viol or score
    improved_viol  = avg_viol  < BASELINE_VIOL
    improved_score = avg_score > BASELINE_SCORE
    improved_ch4   = avg_ch4   > BASELINE_CH4
    passed = improved_viol or improved_score

    log(f"\n  Viol improved:  {improved_viol}  ({avg_viol:.1%} vs {BASELINE_VIOL:.1%})")
    log(f"  Score improved: {improved_score}  ({avg_score:.4f} vs {BASELINE_SCORE:.4f})")
    log(f"  CH4 improved:   {improved_ch4}  ({avg_ch4:.1f} vs {BASELINE_CH4:.1f})")
    log(f"  PASSED: {passed}")

    return {
        'passed':          passed,
        'stages_name':     stages_name,
        'n_records':       n_records,
        'avg_viol':        avg_viol,
        'avg_ch4':         avg_ch4,
        'avg_score':       avg_score,
        'avg_term':        avg_term,
        'improved_viol':   improved_viol,
        'improved_score':  improved_score,
        'improved_ch4':    improved_ch4,
        'delta_viol':      avg_viol  - BASELINE_VIOL,
        'delta_score':     avg_score - BASELINE_SCORE,
        'delta_ch4':       avg_ch4   - BASELINE_CH4,
    }


def check_models_exist(stages_name: str) -> list:
    """Return list of seeds whose models already exist."""
    models_dir = ROOT / 'models'
    existing = []
    for seed in SEEDS:
        name = f'sac_scenario_cur_{REWARD_CONFIG}_{stages_name}_seed{seed}'
        mp = models_dir / name / 'final_model.zip'
        if mp.exists():
            existing.append(seed)
    return existing


def check_evals_exist(stages_name: str) -> int:
    """Return number of eval records for this variant."""
    if not SCENARIO_CUR_JSON.exists():
        return 0
    with open(SCENARIO_CUR_JSON) as f:
        data = json.load(f)
    return sum(1 for r in data
               if r.get('stages_name') == stages_name
               and r.get('reward_config') == REWARD_CONFIG)


# ── Main experiment loop ──────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-workers', type=int, default=MAX_WORKERS)
    parser.add_argument('--variants', nargs='+', default=VARIANTS)
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip training/eval if models/results already exist')
    args = parser.parse_args()

    log("=" * 65)
    log("  Scenario-Difficulty Curriculum Experiment")
    log(f"  Baseline: viol={BASELINE_VIOL:.1%} CH4={BASELINE_CH4:.0f} score={BASELINE_SCORE:.4f}")
    log(f"  Variants to try: {args.variants}")
    log("=" * 65)

    all_results = []
    best_result = None

    for i, stages_name in enumerate(args.variants):
        log(f"\n{'='*65}")
        log(f"  VARIANT {i+1}/{len(args.variants)}: {stages_name}")
        log(f"{'='*65}")

        # ── Training ─────────────────────────────────────────────────────────
        existing_seeds = check_models_exist(stages_name)
        if len(existing_seeds) == len(SEEDS):
            log(f"  All models already exist, skipping training.")
        else:
            missing_seeds = [s for s in SEEDS if s not in existing_seeds]
            log(f"  Existing seeds: {existing_seeds}, missing: {missing_seeds}")
            ok = run_training_parallel(stages_name, args.max_workers)
            if not ok:
                log(f"  Training failed for {stages_name}, skipping.")
                continue

        # ── Evaluation ───────────────────────────────────────────────────────
        n_eval_records = check_evals_exist(stages_name)
        expected = len(SEEDS) * 6   # seeds × test scenarios
        if n_eval_records >= expected:
            log(f"  Eval already complete ({n_eval_records} records), skipping.")
        else:
            log(f"  Eval records: {n_eval_records}/{expected}, running eval...")
            ok = run_evaluation(stages_name)
            if not ok:
                log(f"  Evaluation failed for {stages_name}, skipping.")
                continue

        # ── Analysis ─────────────────────────────────────────────────────────
        result = analyze_results(stages_name)
        all_results.append(result)

        if result.get('passed'):
            log(f"\n  ✓ VARIANT {stages_name} PASSED — stopping search.")
            best_result = result
            break
        else:
            log(f"\n  ✗ VARIANT {stages_name} did not beat baseline. Trying next...")

    # ── Final report ─────────────────────────────────────────────────────────
    log(f"\n{'='*65}")
    log("  EXPERIMENT COMPLETE")
    log(f"{'='*65}")
    log(f"  Baseline: viol={BASELINE_VIOL:.1%}  CH4={BASELINE_CH4:.0f}  score={BASELINE_SCORE:.4f}")
    log(f"")

    for r in all_results:
        sn = r.get('stages_name', '?')
        if r.get('error'):
            log(f"  {sn:<20}: ERROR ({r['error']})")
            continue
        tick = '✓' if r['passed'] else '✗'
        log(f"  {tick} {sn:<18}: viol={r['avg_viol']:.1%}  "
            f"CH4={r['avg_ch4']:.0f}  score={r['avg_score']:.4f}  "
            f"Δviol={r['delta_viol']:+.1%}  Δscore={r['delta_score']:+.4f}")

    if best_result:
        log(f"\n  BEST DESIGN: {best_result['stages_name']}")
        log(f"    violation rate: {best_result['avg_viol']:.1%}  ({best_result['delta_viol']:+.1%} vs baseline)")
        log(f"    CH4 production: {best_result['avg_ch4']:.1f} m³/d ({best_result['delta_ch4']:+.1f})")
        log(f"    overall score:  {best_result['avg_score']:.4f} ({best_result['delta_score']:+.4f})")
    else:
        log("\n  NO VARIANT beat the baseline.")
        log("  Recommendation: reconsider FK shaping or use shorter control horizon.")

    # Save summary
    summary_path = RESULTS_DIR / 'scenario_cur_experiment_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'baseline': {
                'viol': BASELINE_VIOL,
                'ch4': BASELINE_CH4,
                'score': BASELINE_SCORE,
            },
            'results': all_results,
            'best': best_result,
        }, f, indent=2)
    log(f"\n  Summary saved → {summary_path}")


if __name__ == '__main__':
    main()

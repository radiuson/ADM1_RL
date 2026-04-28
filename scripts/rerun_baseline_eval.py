#!/usr/bin/env python3
"""
Baseline controller re-evaluation with MetricsCalculator.

Evaluates Constant, PID, and CascadedPID on all six paper scenarios using the
same MetricsCalculator that is used for SAC evaluation, so that violation_rate
(union-based, ∈ [0,1]) and overall_score are directly comparable.

Results are saved to:
    <results_dir>/paper_direction_a/evaluation/baselines/<controller>_on_<scenario>.json

A summary table is printed at the end, including the Python list literals
needed to update the hardcoded values in analysis/plot_combo.py.

Usage:
    python scripts/rerun_baseline_eval.py --results-dir /path/to/ADM1/rl/results
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from env.adm1_gym_env import ADM1Env_v2
from baselines.baseline_controllers import get_controller
from evaluation.metrics_calculator import MetricsCalculator


# ── Constants ─────────────────────────────────────────────────────────────────

SCENARIOS = [
    'nominal', 'high_load', 'shock_load',
    'low_load', 'temperature_drop', 'cold_winter',
]

# Order must match SCENARIOS list in plot_combo.py
PLOT_SCENARIOS = [
    'nominal', 'high_load', 'low_load',
    'shock_load', 'temperature_drop', 'cold_winter',
]

CONTROLLERS = [
    ('Constant',     'constant',     {}),
    ('PID',          'pid',          {'K_p': 0.5, 'K_i': 0.1, 'K_d': 0.05}),
    ('CascadedPID',  'cascaded_pid', {}),
]

NUM_STEPS = 2880
EVAL_SEED = 42


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(controller_type: str, controller_params: dict,
                scenario: str, seed: int = EVAL_SEED) -> dict:
    """Run one episode and return MetricsCalculator output."""
    env = ADM1Env_v2(scenario_name=scenario)
    obs, _ = env.reset(seed=seed)

    ctrl = get_controller(controller_type, **controller_params)
    ctrl.reset()

    action_dim = env.action_space.shape[0]
    calc = MetricsCalculator()
    terminated_early = False

    for step in range(NUM_STEPS):
        action = ctrl.get_action(obs)
        if len(action) < action_dim:
            action = np.append(action, [0.0] * (action_dim - len(action)))

        obs, reward, terminated, truncated, info = env.step(action)
        calc.add_step(obs, action, reward, info)

        if terminated:
            calc.set_terminated(step)
            terminated_early = True
            break
        if truncated:
            break

    env.close()
    metrics = calc.compute_metrics()
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Re-evaluate engineering baselines with fixed MetricsCalculator.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--results-dir', required=True,
        help='Root results directory (contains paper_direction_a/).',
    )
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir).resolve()
    out_dir = results_dir / 'paper_direction_a' / 'evaluation' / 'baselines'
    out_dir.mkdir(parents=True, exist_ok=True)

    total  = len(CONTROLLERS) * len(SCENARIOS)
    done   = 0
    errors = 0
    t_start = time.time()

    # collected[ctrl_name][scenario] = overall_score
    collected: dict = {c[0]: {} for c in CONTROLLERS}

    print(f"=== rerun_baseline_eval.py started ===")
    print(f"  results_dir : {results_dir}")
    print(f"  output_dir  : {out_dir}")
    print(f"  total runs  : {total}")
    print()

    for ctrl_name, ctrl_type, ctrl_params in CONTROLLERS:
        for scenario in SCENARIOS:
            done += 1
            t0 = time.time()
            print(f"  [{done}/{total}]  {ctrl_name:<14} -> {scenario:<20} ... ",
                  end='', flush=True)

            try:
                metrics = run_episode(ctrl_type, ctrl_params, scenario, EVAL_SEED)

                record = {
                    'controller':   ctrl_name,
                    'controller_type': ctrl_type,
                    'scenario':     scenario,
                    'seed':         EVAL_SEED,
                    'num_steps':    NUM_STEPS,
                    **metrics.get('summary', {}),
                    'violation_rate':   metrics['safety']['violation_rate'],
                    'avg_ch4':          metrics['production']['avg_ch4_flow'],
                    'terminated_early': metrics['episode_info']['terminated_early'],
                }

                out_file = out_dir / f"{ctrl_type}_on_{scenario}.json"
                with open(out_file, 'w') as f:
                    json.dump({'record': record, 'full_metrics': metrics}, f, indent=2)

                score = metrics['summary']['overall_score']
                vr    = metrics['safety']['violation_rate']
                flag  = '  WARN: vr>1' if vr > 1.0 else ''
                print(f"{time.time()-t0:.1f}s  score={score:.3f}  vr={vr:.4f}{flag}")

                collected[ctrl_name][scenario] = score

            except Exception as e:
                errors += 1
                print(f"ERROR: {e}")
                collected[ctrl_name][scenario] = float('nan')

    elapsed = time.time() - t_start
    print(f"\n=== Done: {done} runs, {errors} errors in {elapsed:.1f}s ===\n")

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"{'Controller':<16} " + "  ".join(f"{s[:8]:>10}" for s in PLOT_SCENARIOS))
    print("-" * (16 + 12 * len(PLOT_SCENARIOS)))
    for ctrl_name, _, _ in CONTROLLERS:
        row = "  ".join(
            f"{collected[ctrl_name].get(s, float('nan')):>10.3f}"
            for s in PLOT_SCENARIOS
        )
        print(f"{ctrl_name:<16} {row}")

    # ── Print plot_combo.py update snippet ──────────────────────────────────
    print("\n" + "=" * 70)
    print("Paste the following into analysis/plot_combo.py")
    print("(replace the hardcoded 'mean' lists for Constant / PID / CascadedPID):")
    print("=" * 70)
    for ctrl_name, _, _ in CONTROLLERS:
        vals = [collected[ctrl_name].get(s, float('nan')) for s in PLOT_SCENARIOS]
        vals_str = ", ".join(f"{v:7.3f}" for v in vals)
        print(f"  # {ctrl_name}")
        print(f"  mean=[{vals_str}],")
    print("=" * 70)


if __name__ == '__main__':
    main()

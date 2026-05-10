#!/usr/bin/env python3
"""
Thermal Baseline Controller Re-evaluation.

Evaluates ConstantThermal and FullPID on the two thermal scenarios
(temperature_drop, cold_winter) with scenario-appropriate Q_HEX bias values,
providing a fair comparison against SAC which can freely adjust Q_HEX.

Q_HEX bias is set to the steady-state value required to maintain 35°C:
  - temperature_drop : 500  W  (UA=50 W/K, ΔT=10°C at nominal)
  - cold_winter      : 2400 W  (UA=80 W/K, ΔT=30°C)

Results are saved to:
    <results_dir>/single_scenario/evaluation/baselines/
        constant_thermal_on_<scenario>.json
        full_pid_on_<scenario>.json

Usage:
    python scripts/rerun_thermal_baseline_eval.py \\
        --results-dir /path/to/ADM1/rl/results_repro
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from env.adm1_gym_env import ADM1Env_v2
from baselines.baseline_controllers import ConstantThermalController, FullPIDController
from evaluation.metrics_calculator import MetricsCalculator


# ── Scenario-specific Q_HEX bias (steady-state heating required) ──────────────

THERMAL_SCENARIOS = ['temperature_drop', 'cold_winter']

# Steady-state Q_HEX = UA × (T_L_setpoint - T_env)
SCENARIO_QHEX_BIAS = {
    'temperature_drop': 500.0,   # UA=50 × (35-25)=10 → 500 W
    'cold_winter':      2400.0,  # UA=80 × (35- 5)=30 → 2400 W
}

CONTROLLERS = [
    ('ConstantThermal', 'constant_thermal'),
    ('FullPID',         'full_pid'),
]

NUM_STEPS = 2880
EVAL_SEED = 42


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(ctrl, scenario: str, seed: int = EVAL_SEED) -> dict:
    """Run one episode and return MetricsCalculator output."""
    env = ADM1Env_v2(scenario_name=scenario)
    obs, _ = env.reset(seed=seed)
    ctrl.reset()

    calc = MetricsCalculator()

    for step in range(NUM_STEPS):
        action = ctrl.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        calc.add_step(obs, action, reward, info)

        if terminated:
            calc.set_terminated(step)
            break
        if truncated:
            break

    env.close()
    return calc.compute_metrics()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate thermal baselines (ConstantThermal, FullPID) '
                    'on temperature_drop and cold_winter.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--results-dir', required=True,
                        help='Root results directory (contains single_scenario/).')
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir).resolve()
    out_dir = results_dir / 'single_scenario' / 'evaluation' / 'baselines'
    out_dir.mkdir(parents=True, exist_ok=True)

    total   = len(CONTROLLERS) * len(THERMAL_SCENARIOS)
    done    = 0
    errors  = 0
    t_start = time.time()

    print('=== rerun_thermal_baseline_eval.py started ===')
    print(f'  results_dir : {results_dir}')
    print(f'  output_dir  : {out_dir}')
    print(f'  total runs  : {total}')
    print()

    collected: dict = {c[0]: {} for c in CONTROLLERS}

    for ctrl_name, ctrl_type in CONTROLLERS:
        for scenario in THERMAL_SCENARIOS:
            done += 1
            q_bias = SCENARIO_QHEX_BIAS[scenario]

            # Instantiate controller with scenario-appropriate Q_HEX bias
            if ctrl_type == 'constant_thermal':
                ctrl = ConstantThermalController(Q_HEX=q_bias)
            else:  # full_pid
                ctrl = FullPIDController(Q_HEX_bias=q_bias,
                                         Q_HEX_min=0.0, Q_HEX_max=5000.0)

            t0 = time.time()
            print(f'  [{done}/{total}]  {ctrl_name:<18} -> {scenario:<20} '
                  f'(Q_HEX_bias={q_bias:.0f}W) ... ', end='', flush=True)

            try:
                metrics = run_episode(ctrl, scenario, EVAL_SEED)

                record = {
                    'controller':       ctrl_name,
                    'controller_type':  ctrl_type,
                    'scenario':         scenario,
                    'seed':             EVAL_SEED,
                    'num_steps':        NUM_STEPS,
                    'Q_HEX_bias':       q_bias,
                    **metrics.get('summary', {}),
                    'violation_rate':   metrics['safety']['violation_rate'],
                    'avg_ch4':          metrics['production']['avg_ch4_flow'],
                    'terminated_early': metrics['episode_info']['terminated_early'],
                }

                out_file = out_dir / f'{ctrl_type}_on_{scenario}.json'
                with open(out_file, 'w') as f:
                    json.dump({'record': record, 'full_metrics': metrics}, f, indent=2)

                score = metrics['summary']['overall_score']
                vr    = metrics['safety']['violation_rate']
                print(f"{time.time()-t0:.1f}s  score={score:.3f}  vr={vr:.4f}")
                collected[ctrl_name][scenario] = score

            except Exception as e:
                errors += 1
                print(f'ERROR: {e}')
                import traceback; traceback.print_exc()
                collected[ctrl_name][scenario] = float('nan')

    elapsed = time.time() - t_start
    print(f'\n=== Done: {done} runs, {errors} errors in {elapsed:.1f}s ===\n')

    # ── Comparison summary ──────────────────────────────────────────────────
    print(f"{'Controller':<20} {'temp_drop':>12} {'cold_winter':>12}")
    print('-' * 46)
    for ctrl_name, _ in CONTROLLERS:
        td  = collected[ctrl_name].get('temperature_drop', float('nan'))
        cw  = collected[ctrl_name].get('cold_winter',      float('nan'))
        print(f'{ctrl_name:<20} {td:>12.3f} {cw:>12.3f}')

    print()
    print('Files written to:', out_dir)
    print('Next step: re-run python analysis/plot_combo.py to update figures.')


if __name__ == '__main__':
    main()

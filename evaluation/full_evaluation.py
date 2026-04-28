#!/usr/bin/env python3
"""
Full Baseline Evaluation
========================

Evaluates engineering baseline controllers on all 6 paper scenarios using the
MetricsCalculator for comprehensive metrics (production, safety, stability,
economics).

Paper scenarios: nominal, high_load, low_load, shock_load,
                 temperature_drop, cold_winter

Controllers evaluated:
    Constant, PID (tuned), CascadedPID

Usage:
    # Evaluate all controllers on all paper scenarios (default)
    python full_evaluation.py

    # Custom output directory
    python full_evaluation.py --output-dir results/baselines

    # Single scenario
    python full_evaluation.py --scenario nominal

    # Custom episode length
    python full_evaluation.py --steps 1440
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from env.adm1_gym_env import ADM1Env_v2
from baselines.baseline_controllers import get_controller
from evaluation.metrics_calculator import MetricsCalculator
from training.reward_configs import REWARD_CONFIGS
from env.scenario_manager import ScenarioManager


# Paper-reported PID gains (determined via tune_pid.py grid search)
_PID_PARAMS = {'K_p': 0.5, 'K_i': 0.1, 'K_d': 0.05}

# Controllers evaluated in the paper
_CONTROLLERS = [
    ('Constant',     'constant',     {}),
    ('PID',          'pid',          _PID_PARAMS),
    ('CascadedPID',  'cascaded_pid', {}),
]

# Default evaluation settings
DEFAULT_NUM_STEPS       = 2880   # 30 days at 15-min intervals
DEFAULT_REWARD_CONFIG   = 'safety_first'
DEFAULT_SEED            = 42


def evaluate_controller_on_scenario(
    controller_name: str,
    controller_params: Dict,
    scenario_name: str,
    reward_config_name: str = DEFAULT_REWARD_CONFIG,
    num_steps: int = DEFAULT_NUM_STEPS,
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate a single baseline controller on one scenario.

    Args:
        controller_name:    Key accepted by ``get_controller()``
                            (e.g. 'constant', 'pid', 'cascaded_pid').
        controller_params:  Keyword arguments forwarded to the controller.
        scenario_name:      Scenario key from scenarios.yaml.
        reward_config_name: Key into REWARD_CONFIGS.
        num_steps:          Maximum episode length.
        seed:               Environment reset seed.
        verbose:            Print per-200-step progress and summary.

    Returns:
        Metrics dict from MetricsCalculator.compute_metrics(), augmented
        with a 'metadata' sub-dict.
    """
    if reward_config_name not in REWARD_CONFIGS:
        raise ValueError(
            f"Unknown reward config '{reward_config_name}'. "
            f"Available: {list(REWARD_CONFIGS.keys())}"
        )

    if verbose:
        print(f"\n{'='*70}")
        print(f"  {controller_name} on {scenario_name}")
        print(f"{'='*70}")

    reward_config = REWARD_CONFIGS[reward_config_name]
    env = ADM1Env_v2(scenario_name=scenario_name, reward_config=reward_config)
    obs, _ = env.reset(seed=seed)

    controller = get_controller(controller_name, **controller_params)
    controller.reset()

    action_dim = env.action_space.shape[0]
    calc = MetricsCalculator()

    for step in range(num_steps):
        action = controller.get_action(obs)
        # Pad to env action dimension if controller returns fewer dimensions.
        if len(action) < action_dim:
            action = np.append(action, [0.0] * (action_dim - len(action)))

        obs, reward, terminated, truncated, info = env.step(action)
        calc.add_step(obs, action, reward, info)

        if verbose and step % 200 == 0:
            print(f"  step {step:4d}  pH={info['pH']:.2f}  "
                  f"VFA={info['total_vfa']:.4f}  "
                  f"CH4={info['q_ch4']:.0f}  r={reward:.4f}")

        if terminated:
            calc.set_terminated(step)
            if verbose:
                print(f"  [terminated at step {step}]")
            break
        if truncated:
            break

    env.close()
    metrics = calc.compute_metrics()

    metrics['metadata'] = {
        'controller':        controller_name,
        'controller_params': controller_params,
        'scenario':          scenario_name,
        'reward_config':     reward_config_name,
        'seed':              seed,
    }

    if verbose:
        print(f"\n  reward={metrics['reward']['mean']:.4f}  "
              f"CH4={metrics['production']['avg_ch4_flow']:.1f} m3/d  "
              f"viol={metrics['safety']['violation_rate']*100:.1f}%  "
              f"score={metrics['summary']['overall_score']:.4f}")

    return metrics


def full_evaluation(
    scenarios: Optional[List[str]] = None,
    reward_config_name: str = DEFAULT_REWARD_CONFIG,
    num_steps: int = DEFAULT_NUM_STEPS,
    seed: int = DEFAULT_SEED,
    output_dir: str = '.',
    verbose: bool = True,
) -> List[Dict]:
    """
    Evaluate all baseline controllers on the specified scenarios.

    Args:
        scenarios:          List of scenario keys. If None, uses the 'paper'
                            group from ScenarioManager (all 6 paper scenarios).
        reward_config_name: Key into REWARD_CONFIGS.
        num_steps:          Episode length per evaluation.
        seed:               Random seed.
        output_dir:         Directory for JSON and CSV output files.
        verbose:            Per-step progress and summary printing.

    Returns:
        List of metrics dicts, one per (controller, scenario) pair.
    """
    if scenarios is None:
        manager = ScenarioManager()
        scenarios = manager.get_scenario_group('paper')

    print("=" * 70)
    print("  FULL BASELINE EVALUATION")
    print(f"  Scenarios:   {scenarios}")
    print(f"  Controllers: {[c[0] for c in _CONTROLLERS]}")
    print(f"  Steps:       {num_steps}  Seed: {seed}")
    print("=" * 70)

    all_results = []
    for scenario in scenarios:
        for display_name, controller_type, params in _CONTROLLERS:
            result = evaluate_controller_on_scenario(
                controller_name=controller_type,
                controller_params=params,
                scenario_name=scenario,
                reward_config_name=reward_config_name,
                num_steps=num_steps,
                seed=seed,
                verbose=verbose,
            )
            result['metadata']['display_name'] = display_name
            all_results.append(result)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_file = out / 'full_evaluation_results.json'
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {json_file}")

    _generate_summary_report(all_results, scenarios)
    _export_to_csv(all_results, out / 'full_evaluation_results.csv')

    return all_results


# ── Internal helpers ──────────────────────────────────────────────────────────

def _generate_summary_report(results: List[Dict], scenarios: List[str]) -> None:
    """Print a formatted per-scenario and overall summary table."""
    print(f"\n{'='*70}")
    print("  SUMMARY REPORT")
    print(f"{'='*70}")

    for scenario in scenarios:
        sc_results = [r for r in results if r['metadata']['scenario'] == scenario]

        print(f"\n  {scenario.upper()}")
        print(f"  {'Controller':<20} {'Reward':>9} {'CH4':>9} {'Viol%':>8} "
              f"{'Score':>8}  Status")
        print("  " + "-" * 62)

        for r in sc_results:
            name      = r['metadata'].get('display_name', r['metadata']['controller'])
            reward    = r['reward']['mean']
            ch4       = r['production']['avg_ch4_flow']
            viol      = r['safety']['violation_rate'] * 100
            score     = r['summary']['overall_score']
            status    = 'TERM' if r['episode_info']['terminated_early'] else 'ok'
            print(f"  {name:<20} {reward:>9.4f} {ch4:>9.1f} {viol:>7.1f}% "
                  f"{score:>8.4f}  {status}")

        valid = [r for r in sc_results if not r['episode_info']['terminated_early']]
        if valid:
            best = max(valid, key=lambda r: r['summary']['overall_score'])
            best_name = best['metadata'].get('display_name', best['metadata']['controller'])
            print(f"\n  Best: {best_name} "
                  f"(score={best['summary']['overall_score']:.4f})")

    print(f"\n{'='*70}")
    print("  OVERALL (averaged across scenarios)")
    print(f"{'='*70}")
    print(f"\n  {'Controller':<20} {'Avg Reward':>12} {'Avg CH4':>10} "
          f"{'Avg Viol%':>10} {'Avg Score':>10}")
    print("  " + "-" * 66)

    seen = {}
    for r in results:
        k = r['metadata'].get('display_name', r['metadata']['controller'])
        seen.setdefault(k, []).append(r)

    for name, rlist in seen.items():
        avg_reward = float(np.mean([r['reward']['mean'] for r in rlist]))
        avg_ch4    = float(np.mean([r['production']['avg_ch4_flow'] for r in rlist]))
        avg_viol   = float(np.mean([r['safety']['violation_rate'] for r in rlist])) * 100
        avg_score  = float(np.mean([r['summary']['overall_score'] for r in rlist]))
        print(f"  {name:<20} {avg_reward:>12.4f} {avg_ch4:>10.1f} "
              f"{avg_viol:>9.1f}% {avg_score:>10.4f}")


def _export_to_csv(results: List[Dict], csv_path: Path) -> None:
    """Export results to CSV."""
    rows = []
    for r in results:
        rows.append({
            'controller':             r['metadata'].get('display_name', r['metadata']['controller']),
            'scenario':               r['metadata']['scenario'],
            'reward_mean':            r['reward']['mean'],
            'reward_std':             r['reward']['std'],
            'ch4_total_m3':           r['production']['total_ch4_m3'],
            'ch4_avg_flow':           r['production']['avg_ch4_flow'],
            'volumetric_productivity':r['production']['volumetric_productivity'],
            'violation_rate':         r['safety']['violation_rate'],
            'steps_with_violation':   r['safety']['steps_with_violation'],
            'ph_violation_count':     r['safety']['ph_violation_count'],
            'vfa_violation_count':    r['safety']['vfa_violation_count'],
            'nh3_violation_count':    r['safety']['nh3_violation_count'],
            'ph_mean':                r['safety']['ph_mean'],
            'vfa_max':                r['safety']['vfa_max'],
            'ch4_volatility':         r['stability']['ch4_volatility'],
            'net_energy_kwh':         r['economics']['net_energy_kwh'],
            'overall_score':          r['summary']['overall_score'],
            'terminated':             r['episode_info']['terminated_early'],
        })

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"CSV saved to {csv_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate baseline controllers on ADM1 paper scenarios.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--scenario', type=str, default=None,
                        help='Single scenario to evaluate (default: all paper scenarios)')
    parser.add_argument('--reward-config', type=str, default=DEFAULT_REWARD_CONFIG,
                        help='Reward config key (training/reward_configs.py)')
    parser.add_argument('--steps', type=int, default=DEFAULT_NUM_STEPS,
                        help='Episode length (2880 = 30 days at 15-min intervals)')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory for JSON and CSV output')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress per-step progress output')
    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario else None

    full_evaluation(
        scenarios=scenarios,
        reward_config_name=args.reward_config,
        num_steps=args.steps,
        seed=args.seed,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Baseline Controller Evaluation
===============================

Evaluates engineering baseline controllers on ADM1Env_v2 scenarios and
reports the metrics used in the paper (methane yield, safety violation rate,
pH statistics).

Paper-reported baselines:
    constant, pid, cascaded_pid,
    constant_thermal, thermal_pid, full_pid   (thermal scenarios only)

Usage examples:
    # Single controller, full 30-day episode
    python evaluate_baselines.py --controller pid --scenario nominal

    # Compare all paper-relevant controllers on one scenario
    python evaluate_baselines.py --controller all --scenario cold_winter

    # Custom episode length
    python evaluate_baselines.py --controller constant --scenario high_load --steps 2880
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from env.adm1_gym_env import ADM1Env_v2
from baselines.baseline_controllers import get_controller


# Scenarios that involve thermal disturbances and require Q_HEX control.
_THERMAL_SCENARIOS = {'cold_winter', 'temperature_drop'}

# Controllers reported in the paper per scenario type.
_PAPER_CONTROLLERS = {
    'non_thermal': [
        ('Constant',     'constant',     {}),
        ('PID',          'pid',          {'K_p': 0.5, 'K_i': 0.1,  'K_d': 0.05}),
        ('CascadedPID',  'cascaded_pid', {}),
    ],
    'thermal': [
        ('ConstantThermal', 'constant_thermal', {}),
        ('FullPID',         'full_pid',         {}),
    ],
}


def _build_controller_list(scenario_name: str) -> List[tuple]:
    """Return (display_name, type, params) triples for a given scenario."""
    ctrls = list(_PAPER_CONTROLLERS['non_thermal'])
    if scenario_name in _THERMAL_SCENARIOS:
        Q_HEX_bias = 2400.0 if scenario_name == 'cold_winter' else 500.0
        for name, ctype, params in _PAPER_CONTROLLERS['thermal']:
            ctrls.append((name, ctype, {**params, 'Q_HEX_bias': Q_HEX_bias}))
    return ctrls


def evaluate_controller(
    controller_name: str,
    scenario_name: str,
    num_steps: int = 2880,
    seed: int = 42,
    controller_params: Optional[Dict] = None,
    reward_config: Optional[Dict] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run one episode of a single controller on a scenario.

    Args:
        controller_name : Key accepted by ``get_controller()``
                          (e.g. 'pid', 'constant', 'cascaded_pid').
        scenario_name   : Scenario key from scenarios.yaml
                          (e.g. 'nominal', 'cold_winter').
        num_steps       : Maximum episode length (default: 2880 = 30 days).
        seed            : Environment reset seed.
        controller_params: Keyword arguments forwarded to the controller constructor.
        reward_config   : Optional reward config dict for ADM1Env_v2.
                          If None, the environment default is used.
        verbose         : Print per-100-step progress.

    Returns:
        dict with keys: controller, scenario, seed, episode_length,
        terminated_early, avg_reward, avg_ch4, violation_rate, ...
    """
    env_kwargs = dict(scenario_name=scenario_name)
    if reward_config is not None:
        env_kwargs['reward_config'] = reward_config

    env = ADM1Env_v2(**env_kwargs)
    obs, _ = env.reset(seed=seed)

    controller_params = controller_params or {}
    controller = get_controller(controller_name, **controller_params)
    controller.reset()

    action_dim = env.action_space.shape[0]

    rewards, ch4_vals, ph_vals, vfa_vals, nh3_vals = [], [], [], [], []
    n_violations = 0
    terminated_early = False
    episode_length = 0

    for step in range(num_steps):
        action = controller.get_action(obs)
        # Pad to env action dim if controller returns fewer dimensions.
        if len(action) < action_dim:
            action = np.append(action, [0.0] * (action_dim - len(action)))

        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(reward)
        ch4_vals.append(info['q_ch4'])
        ph_vals.append(info['pH'])
        vfa_vals.append(info['total_vfa'])
        nh3_vals.append(info['S_nh3'])

        violated = (
            info['pH'] < 6.8 or info['pH'] > 7.8
            or info['total_vfa'] > 0.2
            or info['S_nh3'] > 0.002
        )
        if violated:
            n_violations += 1

        episode_length += 1

        if verbose and step % 100 == 0:
            print(f"  step {step:4d}/{num_steps}  "
                  f"pH={info['pH']:.2f}  VFA={info['total_vfa']:.4f}  "
                  f"CH4={info['q_ch4']:.1f}  r={reward:.4f}")

        if terminated:
            terminated_early = True
            if verbose:
                print(f"  [terminated early at step {step}]")
            break
        if truncated:
            break

    env.close()

    return {
        'controller':       controller_name,
        'scenario':         scenario_name,
        'seed':             seed,
        'episode_length':   episode_length,
        'terminated_early': terminated_early,
        # Reward
        'avg_reward':       float(np.mean(rewards)),
        'std_reward':       float(np.std(rewards, ddof=1)),
        'total_reward':     float(np.sum(rewards)),
        # Production
        'avg_ch4':          float(np.mean(ch4_vals)),
        'std_ch4':          float(np.std(ch4_vals, ddof=1)),
        'total_ch4_m3':     float(np.sum(ch4_vals) * (15.0 / 1440)),  # m³ over episode
        # Safety
        'avg_ph':           float(np.mean(ph_vals)),
        'min_ph':           float(np.min(ph_vals)),
        'max_ph':           float(np.max(ph_vals)),
        'avg_vfa':          float(np.mean(vfa_vals)),
        'max_vfa':          float(np.max(vfa_vals)),
        'max_nh3':          float(np.max(nh3_vals)),
        'violation_rate':   n_violations / episode_length if episode_length > 0 else 0.0,
        'n_violations':     n_violations,
    }


def compare_controllers(
    scenario_name: str = 'nominal',
    num_steps: int = 2880,
    seed: int = 42,
    output_json: Optional[Path] = None,
) -> List[Dict]:
    """
    Evaluate all paper-relevant controllers on a single scenario and print a
    summary table.

    Args:
        scenario_name : Scenario key from scenarios.yaml.
        num_steps     : Episode length (default: 2880 = 30 days).
        seed          : Random seed.
        output_json   : If given, write results list to this JSON file.

    Returns:
        List of result dicts from ``evaluate_controller``.
    """
    ctrl_list = _build_controller_list(scenario_name)

    print(f"\n{'='*70}")
    print(f"  Scenario: {scenario_name}   steps={num_steps}   seed={seed}")
    print(f"  Controllers: {[c[0] for c in ctrl_list]}")
    print(f"{'='*70}")

    results = []
    for display_name, ctrl_type, params in ctrl_list:
        print(f"\n-- {display_name} --")
        res = evaluate_controller(
            controller_name=ctrl_type,
            scenario_name=scenario_name,
            num_steps=num_steps,
            seed=seed,
            controller_params=params,
            verbose=False,
        )
        res['display_name'] = display_name
        results.append(res)

    # Summary table
    print(f"\n{'Controller':<20} {'AvgCH4':>9} {'ViolRate':>10} {'AvgpH':>8} {'Status'}")
    print('-' * 60)
    for r in results:
        status = 'TERMINATED' if r['terminated_early'] else 'ok'
        print(f"{r['display_name']:<20} {r['avg_ch4']:>9.1f} "
              f"{r['violation_rate']:>10.4f} {r['avg_ph']:>8.3f}  {status}")

    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_json}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate baseline controllers on ADM1 scenarios.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--controller', type=str, default='all',
                        help='Controller type key, or "all" to compare paper baselines')
    parser.add_argument('--scenario', type=str, default='nominal',
                        help='Scenario name (see env/scenarios.yaml)')
    parser.add_argument('--steps', type=int, default=2880,
                        help='Episode length (2880 = 30 days at 15-min intervals)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None,
                        help='Optional path to save results as JSON')
    # PID gains (used when --controller pid)
    parser.add_argument('--K_p', type=float, default=0.5)
    parser.add_argument('--K_i', type=float, default=0.1)
    parser.add_argument('--K_d', type=float, default=0.05)

    args = parser.parse_args()
    out = Path(args.output) if args.output else None

    if args.controller == 'all':
        compare_controllers(
            scenario_name=args.scenario,
            num_steps=args.steps,
            seed=args.seed,
            output_json=out,
        )
    else:
        params = {}
        if args.controller == 'pid':
            params = {'K_p': args.K_p, 'K_i': args.K_i, 'K_d': args.K_d}
        result = evaluate_controller(
            controller_name=args.controller,
            scenario_name=args.scenario,
            num_steps=args.steps,
            seed=args.seed,
            controller_params=params,
        )
        print(f"\navg_ch4={result['avg_ch4']:.1f}  "
              f"violation_rate={result['violation_rate']:.4f}  "
              f"terminated={result['terminated_early']}")
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Result saved to {out}")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # No arguments: run paper-relevant comparison on nominal scenario
        compare_controllers(scenario_name='nominal', num_steps=2880, seed=42)
    else:
        main()

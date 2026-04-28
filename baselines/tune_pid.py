#!/usr/bin/env python3
"""
PID Gain Tuning Utility
========================

Grid search over PID gains (K_p, K_i, K_d) to find good parameters for the
PID and CascadedPID baseline controllers.

The gains reported in the paper (K_p=0.5, K_i=0.1, K_d=0.05) were determined
using this script on the nominal scenario with the safety-first reward.

Usage:
    # Grid search on nominal scenario (default)
    python tune_pid.py

    # Grid search on a specific scenario
    python tune_pid.py --scenario high_load --steps 1440

    # Single evaluation with known gains
    python tune_pid.py --mode eval --K_p 0.5 --K_i 0.1 --K_d 0.05
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from env.adm1_gym_env import ADM1Env_v2
from baselines.baseline_controllers import PIDController
from training.reward_configs import REWARD_CONFIGS


def evaluate_pid_params(
    K_p: float,
    K_i: float,
    K_d: float,
    scenario_name: str = 'nominal',
    reward_config_name: str = 'conservative',
    num_steps: int = 1440,
    seed: int = 42,
    verbose: bool = False,
) -> Dict:
    """
    Evaluate a PID controller with given gains for one episode.

    Args:
        K_p, K_i, K_d  : PID gains.
        scenario_name   : Scenario key from scenarios.yaml.
        reward_config_name: Key into REWARD_CONFIGS (training/reward_configs.py).
        num_steps       : Episode length (default 1440 = 15 days).
        seed            : Random seed.
        verbose         : Print per-step progress.

    Returns:
        dict with keys: K_p, K_i, K_d, avg_reward, avg_ch4,
        max_vfa, violation_rate, terminated.
    """
    reward_config = REWARD_CONFIGS[reward_config_name]
    env = ADM1Env_v2(scenario_name=scenario_name, reward_config=reward_config)
    obs, _ = env.reset(seed=seed)

    pid = PIDController(K_p=K_p, K_i=K_i, K_d=K_d)
    pid.reset()

    action_dim = env.action_space.shape[0]

    rewards, ch4_vals, vfa_vals = [], [], []
    n_violations = 0
    terminated = False

    for step in range(num_steps):
        action = pid.get_action(obs)
        if len(action) < action_dim:
            action = np.append(action, [0.0] * (action_dim - len(action)))

        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(reward)
        ch4_vals.append(info['q_ch4'])
        vfa_vals.append(info['total_vfa'])

        # Safety violation: pH, VFA, NH3 (matches paper definition)
        if (info['pH'] < 6.8 or info['pH'] > 7.8
                or info['total_vfa'] > 0.2
                or info['S_nh3'] > 0.002):
            n_violations += 1

        if verbose and step % 200 == 0:
            print(f"  step {step:4d}  pH={info['pH']:.2f}  "
                  f"VFA={info['total_vfa']:.4f}  CH4={info['q_ch4']:.1f}")

        if terminated:
            if verbose:
                print(f"  [terminated at step {step}]")
            break
        if truncated:
            break

    env.close()
    n = len(rewards)

    return {
        'K_p':            K_p,
        'K_i':            K_i,
        'K_d':            K_d,
        'avg_reward':     float(np.mean(rewards)),
        'std_reward':     float(np.std(rewards, ddof=1)),
        'avg_ch4':        float(np.mean(ch4_vals)),
        'max_vfa':        float(np.max(vfa_vals)),
        'violation_rate': n_violations / n if n > 0 else 0.0,
        'terminated':     bool(terminated),
    }


def grid_search_pid(
    scenario_name: str = 'nominal',
    reward_config_name: str = 'conservative',
    num_steps: int = 1440,
    seed: int = 42,
    output_json: Optional[Path] = None,
) -> Tuple[List[Dict], Optional[Dict]]:
    """
    Grid search over PID gains and report the best configuration.

    Scoring: avg_reward - 0.5 * violation_rate  (penalises unsafe operation).
    Terminated episodes are excluded from the best-selection ranking.

    Args:
        scenario_name      : Scenario key from scenarios.yaml.
        reward_config_name : Key into REWARD_CONFIGS.
        num_steps          : Episode length per evaluation.
        seed               : Random seed.
        output_json        : Optional path to write ranked results as JSON.

    Returns:
        (results_list, best_result) — results sorted by score descending.
    """
    # Parameter grid used in the paper tuning experiments
    K_p_values = [0.2, 0.3, 0.4, 0.5, 0.6]
    K_i_values = [0.02, 0.05, 0.08, 0.10, 0.15]
    K_d_values = [0.01, 0.02, 0.03, 0.05, 0.08]

    total = len(K_p_values) * len(K_i_values) * len(K_d_values)

    print(f"\n{'='*65}")
    print(f"  PID Grid Search  |  scenario={scenario_name}  steps={num_steps}")
    print(f"  K_p: {K_p_values}")
    print(f"  K_i: {K_i_values}")
    print(f"  K_d: {K_d_values}")
    print(f"  Total configs: {total}")
    print(f"{'='*65}")

    results_list: List[Dict] = []
    best_result: Optional[Dict] = None
    best_score = -np.inf
    idx = 0

    for K_p in K_p_values:
        for K_i in K_i_values:
            for K_d in K_d_values:
                idx += 1
                result = evaluate_pid_params(
                    K_p=K_p, K_i=K_i, K_d=K_d,
                    scenario_name=scenario_name,
                    reward_config_name=reward_config_name,
                    num_steps=num_steps,
                    seed=seed,
                )
                results_list.append(result)

                score = result['avg_reward'] - 0.5 * result['violation_rate']
                if score > best_score and not result['terminated']:
                    best_score = score
                    best_result = result

                if idx % 10 == 0 or idx == total:
                    print(f"  [{idx:3d}/{total}]  "
                          f"Kp={K_p:.2f} Ki={K_i:.3f} Kd={K_d:.3f}  "
                          f"reward={result['avg_reward']:.4f}  "
                          f"viol={result['violation_rate']*100:.1f}%")

    # Sort by score
    results_list.sort(
        key=lambda r: r['avg_reward'] - 0.5 * r['violation_rate'],
        reverse=True,
    )

    print(f"\n{'Rank':<6} {'K_p':<8} {'K_i':<8} {'K_d':<8} "
          f"{'AvgReward':<12} {'AvgCH4':<10} {'ViolRate':<10} {'Status'}")
    print('-' * 70)
    for rank, r in enumerate(results_list[:10], 1):
        status = 'TERM' if r['terminated'] else 'ok'
        print(f"{rank:<6} {r['K_p']:<8.2f} {r['K_i']:<8.3f} {r['K_d']:<8.3f} "
              f"{r['avg_reward']:<12.4f} {r['avg_ch4']:<10.1f} "
              f"{r['violation_rate']*100:<9.1f}%  {status}")

    if best_result:
        print(f"\nBest: K_p={best_result['K_p']:.3f}  "
              f"K_i={best_result['K_i']:.3f}  K_d={best_result['K_d']:.3f}  "
              f"score={best_score:.4f}")

    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(results_list, f, indent=2)
        print(f"Results saved to {output_json}")

    return results_list, best_result


def main():
    parser = argparse.ArgumentParser(
        description='Tune or evaluate PID gains for ADM1 baseline controllers.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--mode', choices=['grid_search', 'eval'],
                        default='grid_search')
    parser.add_argument('--scenario', type=str, default='nominal',
                        help='Scenario key from scenarios.yaml')
    parser.add_argument('--reward_config', type=str, default='conservative',
                        help='Reward config key from training/reward_configs.py')
    parser.add_argument('--steps', type=int, default=1440,
                        help='Episode length (1440 = 15 days at 15-min intervals)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save JSON results')
    # Single-eval gains
    parser.add_argument('--K_p', type=float, default=0.5)
    parser.add_argument('--K_i', type=float, default=0.1)
    parser.add_argument('--K_d', type=float, default=0.05)

    args = parser.parse_args()
    out = Path(args.output) if args.output else None

    if args.mode == 'grid_search':
        grid_search_pid(
            scenario_name=args.scenario,
            reward_config_name=args.reward_config,
            num_steps=args.steps,
            seed=args.seed,
            output_json=out,
        )
    else:
        result = evaluate_pid_params(
            K_p=args.K_p, K_i=args.K_i, K_d=args.K_d,
            scenario_name=args.scenario,
            reward_config_name=args.reward_config,
            num_steps=args.steps,
            seed=args.seed,
            verbose=True,
        )
        print(f"\navg_reward={result['avg_reward']:.4f}  "
              f"avg_ch4={result['avg_ch4']:.1f}  "
              f"violation_rate={result['violation_rate']:.4f}  "
              f"terminated={result['terminated']}")
        if out:
            with open(out, 'w') as f:
                json.dump(result, f, indent=2)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        grid_search_pid(scenario_name='nominal', num_steps=1440, seed=42)
    else:
        main()

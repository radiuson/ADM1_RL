#!/usr/bin/env python3
"""
RL Policy Evaluation
====================

Evaluate a trained SAC (or PPO/TD3) policy on any ADM1 benchmark scenario and
report the metrics used in the paper (methane yield, safety violation rate,
pH statistics).

Usage:
    # Evaluate best model from train_sac.py on the training scenario
    python evaluate_rl_policy.py --model models/sac_nominal_conservative_seed42/best_model/best_model

    # Evaluate on a different scenario (cross-scenario generalisation)
    python evaluate_rl_policy.py \\
        --model models/sac_nominal_conservative_seed42/best_model/best_model \\
        --scenario cold_winter --n-episodes 10 --output results/sac_nominal_on_cold.json

    # Stochastic policy evaluation
    python evaluate_rl_policy.py --model <path> --scenario nominal --stochastic
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor

from env.adm1_gym_env import ADM1Env_v2
from training.reward_configs import REWARD_CONFIGS
from evaluation.metrics_calculator import MetricsCalculator


_ALGO_CLASSES = {'ppo': PPO, 'sac': SAC, 'td3': TD3}

PAPER_SCENARIOS = [
    'nominal', 'high_load', 'low_load',
    'shock_load', 'temperature_drop', 'cold_winter',
]


def load_model(model_path: str, algo: Optional[str] = None):
    """
    Load a trained SB3 model.

    Args:
        model_path: Path to saved model file (with or without .zip extension).
        algo:       Algorithm name ('ppo', 'sac', 'td3').
                    If None, inferred from the directory name.

    Returns:
        Loaded SB3 model.
    """
    model_path = Path(model_path)

    if algo is None:
        path_lower = str(model_path).lower()
        for name in _ALGO_CLASSES:
            if name in path_lower:
                algo = name
                break
        if algo is None:
            raise ValueError(
                f"Cannot infer algorithm from path '{model_path}'. "
                f"Please specify --algo (ppo / sac / td3)."
            )

    AlgoClass = _ALGO_CLASSES[algo]
    print(f"Loading {algo.upper()} model from {model_path} ...")
    return AlgoClass.load(model_path)


def _run_single_episode(
    model,
    env,
    seed: int,
    deterministic: bool,
) -> Dict[str, Any]:
    """
    Run one episode and return a MetricsCalculator result dict.

    Args:
        model:         Trained SB3 model.
        env:           ADM1Env_v2 (may be wrapped in Monitor).
        seed:          Episode seed.
        deterministic: Whether to use deterministic policy.

    Returns:
        Dict from MetricsCalculator.compute_metrics(), augmented with
        'episode_reward' and 'episode_length'.
    """
    obs, _ = env.reset(seed=seed)
    calc = MetricsCalculator()
    episode_reward = 0.0
    step = 0

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        calc.add_step(observation=obs, action=action, reward=reward, info=info)
        episode_reward += reward
        step += 1

        if terminated:
            calc.set_terminated(step)

    metrics = calc.compute_metrics()
    metrics['episode_reward'] = episode_reward
    metrics['episode_length'] = step
    return metrics


def evaluate_policy_on_scenario(
    model,
    scenario: str,
    reward_config: str = 'conservative',
    n_episodes: int = 10,
    deterministic: bool = True,
    seed: int = 42,
    obs_mode: str = 'full',
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a trained policy on a single ADM1 scenario.

    Each episode uses a separate MetricsCalculator; results are then aggregated
    as mean ± std over all episodes (matching run_experiment.py methodology).

    Args:
        model:         Trained SB3 model (SAC, PPO, or TD3).
        scenario:      ADM1 scenario name (see env/scenarios.yaml).
        reward_config: Key into REWARD_CONFIGS (training/reward_configs.py).
        n_episodes:    Number of evaluation episodes.
        deterministic: Use deterministic (greedy) policy if True.
        seed:          Base random seed; episode i uses seed + i.
        obs_mode:      Observation mode — 'full' (13-dim) or 'simple' (5-dim).
        verbose:       Print per-episode progress and final summary.

    Returns:
        Dict with keys:
            'episode_stats'  — mean/std of reward and length over episodes
            'metrics'        — mean of per-episode MetricsCalculator outputs
            'metadata'       — scenario, reward_config, seed, n_episodes
    """
    if reward_config not in REWARD_CONFIGS:
        raise ValueError(
            f"Unknown reward config '{reward_config}'. "
            f"Available: {list(REWARD_CONFIGS.keys())}"
        )

    env = Monitor(ADM1Env_v2(
        scenario_name=scenario,
        reward_config=REWARD_CONFIGS[reward_config],
        obs_mode=obs_mode,
    ))

    if verbose:
        print(f"\nEvaluating on '{scenario}' ({n_episodes} episodes, "
              f"deterministic={deterministic}, obs_mode={obs_mode}) ...")

    episode_results = []
    for ep in range(n_episodes):
        result = _run_single_episode(model, env, seed=seed + ep, deterministic=deterministic)
        episode_results.append(result)

        if verbose and (ep + 1) % max(1, n_episodes // 5) == 0:
            print(f"  ep {ep+1:2d}/{n_episodes}  "
                  f"reward={result['episode_reward']:.2f}  "
                  f"len={result['episode_length']}")

    env.close()

    # Aggregate per-episode metrics
    rewards = np.array([r['episode_reward'] for r in episode_results])
    lengths = np.array([r['episode_length'] for r in episode_results])

    def _mean(key_path: str) -> float:
        vals = []
        for r in episode_results:
            v = r
            for k in key_path.split('.'):
                v = v.get(k, np.nan) if isinstance(v, dict) else np.nan
            vals.append(float(v) if not isinstance(v, dict) else np.nan)
        return float(np.nanmean(vals))

    aggregated = {
        'episode_stats': {
            'n_episodes':          n_episodes,
            'mean_episode_reward': float(np.mean(rewards)),
            'std_episode_reward':  float(np.std(rewards, ddof=1)),
            'mean_episode_length': float(np.mean(lengths)),
            'std_episode_length':  float(np.std(lengths, ddof=1)),
        },
        'metrics': {
            'avg_ch4':        _mean('production.avg_ch4_flow'),
            'violation_rate': _mean('safety.violation_rate'),
            'ph_mean':        _mean('safety.ph_mean'),
            'vfa_max':        _mean('safety.vfa_max'),
            'overall_score':  _mean('summary.overall_score'),
        },
        'metadata': {
            'scenario':      scenario,
            'reward_config': reward_config,
            'obs_mode':      obs_mode,
            'deterministic': deterministic,
            'seed':          seed,
        },
    }

    if verbose:
        ep = aggregated['episode_stats']
        m = aggregated['metrics']
        print(f"\n  Mean reward:     {ep['mean_episode_reward']:.4f} "
              f"± {ep['std_episode_reward']:.4f}")
        print(f"  Overall score:   {m['overall_score']:.4f}")
        print(f"  Violation rate:  {m['violation_rate']*100:.1f}%")
        print(f"  Avg CH4:         {m['avg_ch4']:.1f} m³/d")

    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained RL policy on ADM1 benchmark scenarios.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model '
                             '(e.g. models/sac_nominal_conservative_seed42/best_model/best_model)')
    parser.add_argument('--algo', type=str, choices=['ppo', 'sac', 'td3'], default=None,
                        help='Algorithm (auto-inferred from path if not given)')
    parser.add_argument('--scenario', type=str, default='nominal',
                        choices=PAPER_SCENARIOS,
                        help='Scenario to evaluate on')
    parser.add_argument('--reward-config', type=str, default='conservative',
                        help='Reward config key (training/reward_configs.py)')
    parser.add_argument('--n-episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--obs-mode', type=str, default='full',
                        choices=['full', 'simple'],
                        help='Observation mode: full (13-dim) or simple (5-dim)')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic policy (default: deterministic)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to this JSON file')

    args = parser.parse_args()

    print('=' * 65)
    print('  RL POLICY EVALUATION')
    print('=' * 65)
    print(f'  Model:    {args.model}')
    print(f'  Scenario: {args.scenario}')
    print(f'  Episodes: {args.n_episodes}')
    print(f'  Obs mode: {args.obs_mode}')
    print()

    model = load_model(args.model, args.algo)

    results = evaluate_policy_on_scenario(
        model=model,
        scenario=args.scenario,
        reward_config=args.reward_config,
        n_episodes=args.n_episodes,
        deterministic=not args.stochastic,
        seed=args.seed,
        obs_mode=args.obs_mode,
        verbose=True,
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nResults saved to {out}')

    print('\n' + '=' * 65)
    print('  DONE')
    print('=' * 65)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Custom Controller Benchmark Evaluator
======================================

Evaluate any controller implementing BaseController against all six ADM1
paper scenarios and print a results table matching the paper metrics.

Usage:
    python evaluate_controller.py --controller examples/my_controller.py
    python evaluate_controller.py --controller examples/my_controller.py \\
        --scenarios nominal high_load cold_winter \\
        --episodes 10 --output results/my_controller.json

The script auto-discovers the first BaseController subclass in the given file.
"""

import argparse
import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from baselines.baseline_controllers import BaseController
from env.adm1_gym_env import ADM1Env_v2
from evaluation.metrics_calculator import MetricsCalculator


PAPER_SCENARIOS = [
    'nominal', 'high_load', 'low_load',
    'shock_load', 'temperature_drop', 'cold_winter',
]


def load_controller_from_file(path: str) -> BaseController:
    """
    Import a Python file and return an instance of the first BaseController
    subclass found at module level.

    Args:
        path: Path to a .py file containing a BaseController subclass.

    Returns:
        Instantiated controller.
    """
    fpath = Path(path).resolve()
    if not fpath.exists():
        raise FileNotFoundError(f"Controller file not found: {fpath}")

    spec = importlib.util.spec_from_file_location("_user_controller", fpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, BaseController) and obj is not BaseController:
            print(f"Found controller: {obj.__name__}")
            return obj()

    raise RuntimeError(
        f"No BaseController subclass found in {fpath}.\n"
        f"Make sure your file defines a class that inherits from BaseController."
    )


def evaluate_controller_on_scenario(
    controller: BaseController,
    scenario: str,
    n_episodes: int = 10,
    obs_mode: str = 'full',
    base_seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run `controller` for `n_episodes` on `scenario` and return aggregated metrics.

    Args:
        controller:  Any BaseController instance.
        scenario:    Scenario name (e.g. 'nominal', 'cold_winter').
        n_episodes:  Number of independent episodes.
        obs_mode:    'full' (13-dim) or 'simple' (5-dim).
        base_seed:   Episode i uses seed base_seed + i*100.
        verbose:     Print per-episode progress.

    Returns:
        dict with keys 'overall_score', 'violation_rate', 'avg_ch4',
        'terminated_rate', 'n_episodes', and per-episode lists under 'episodes'.
    """
    env = ADM1Env_v2(scenario_name=scenario, obs_mode=obs_mode)

    scores, viols, ch4s, terminated = [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + ep * 100)
        controller.reset()
        calc = MetricsCalculator()
        done = False
        step = 0

        while not done:
            action = controller.get_action(obs)
            obs, reward, term, trunc, info = env.step(action)
            calc.add_step(obs, action, reward, info)
            step += 1
            if term:
                calc.set_terminated(step)
            done = term or trunc

        m = calc.compute_metrics()
        sc = m['summary']['overall_score']
        vr = m['safety']['violation_rate']
        ch = m['production']['avg_ch4_flow']
        te = m['episode_info']['terminated_early']

        scores.append(sc)
        viols.append(vr)
        ch4s.append(ch)
        terminated.append(float(te))

        if verbose:
            flag = ' [TERMINATED]' if te else ''
            print(f"  ep {ep+1:2d}/{n_episodes}  "
                  f"score={sc:.3f}  viol={vr:.3f}  ch4={ch:.1f}{flag}")

    env.close()

    return {
        'overall_score':   float(np.mean(scores)),
        'score_std':       float(np.std(scores, ddof=1)) if n_episodes > 1 else 0.0,
        'violation_rate':  float(np.mean(viols)),
        'avg_ch4':         float(np.mean(ch4s)),
        'terminated_rate': float(np.mean(terminated)),
        'n_episodes':      n_episodes,
        'episodes': {
            'scores':    scores,
            'viols':     viols,
            'ch4s':      ch4s,
            'terminated': terminated,
        },
    }


def run_full_benchmark(
    controller: BaseController,
    scenarios: list,
    n_episodes: int = 10,
    obs_mode: str = 'full',
    base_seed: int = 42,
) -> dict:
    """
    Evaluate `controller` on each scenario in `scenarios`.

    Returns:
        dict mapping scenario name → result dict from evaluate_controller_on_scenario.
    """
    results = {}
    t0 = time.time()

    for sc in scenarios:
        print(f"\n{'─'*55}")
        print(f"  Scenario: {sc}")
        print(f"{'─'*55}")
        results[sc] = evaluate_controller_on_scenario(
            controller, sc,
            n_episodes=n_episodes,
            obs_mode=obs_mode,
            base_seed=base_seed,
        )

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    return results


def print_summary_table(results: dict) -> None:
    """Print a formatted results table to stdout."""
    header = f"{'Scenario':<18}  {'Score':>7}  {'±':>5}  {'Viol%':>7}  {'CH4':>8}  {'Term%':>6}"
    print(f"\n{'═'*60}")
    print("  RESULTS SUMMARY")
    print(f"{'═'*60}")
    print(f"  {header}")
    print(f"  {'─'*56}")

    scores_all = []
    for scenario, r in results.items():
        scores_all.append(r['overall_score'])
        term_pct = r['terminated_rate'] * 100
        term_str = f"{term_pct:.0f}%" if term_pct > 0 else "  —  "
        print(f"  {scenario:<18}  {r['overall_score']:>7.3f}  "
              f"{r['score_std']:>5.3f}  "
              f"{r['violation_rate']*100:>6.1f}%  "
              f"{r['avg_ch4']:>7.1f}  "
              f"{term_str:>6}")

    print(f"  {'─'*56}")
    print(f"  {'Mean':<18}  {np.mean(scores_all):>7.3f}")
    print(f"{'═'*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a custom controller on the ADM1 paper benchmark.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--controller', required=True,
        help='Path to a .py file containing a BaseController subclass '
             '(e.g. examples/my_controller.py).',
    )
    parser.add_argument(
        '--scenarios', nargs='+', default=PAPER_SCENARIOS,
        choices=PAPER_SCENARIOS, metavar='SCENARIO',
        help='Scenarios to evaluate on.',
    )
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes per scenario.')
    parser.add_argument('--obs-mode', choices=['full', 'simple'], default='full',
                        help='Observation mode: full (13-dim) or simple (5-dim).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed.')
    parser.add_argument('--output', default=None,
                        help='Save full results to this JSON file.')
    args = parser.parse_args()

    print('=' * 60)
    print('  ADM1 CUSTOM CONTROLLER BENCHMARK')
    print('=' * 60)
    print(f'  Controller: {args.controller}')
    print(f'  Scenarios:  {", ".join(args.scenarios)}')
    print(f'  Episodes:   {args.episodes}')
    print(f'  Obs mode:   {args.obs_mode}')
    print()

    controller = load_controller_from_file(args.controller)
    results    = run_full_benchmark(
        controller,
        scenarios=args.scenarios,
        n_episodes=args.episodes,
        obs_mode=args.obs_mode,
        base_seed=args.seed,
    )

    print_summary_table(results)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'controller': Path(args.controller).stem,
            'obs_mode':   args.obs_mode,
            'n_episodes': args.episodes,
            'seed':       args.seed,
            'results':    {
                sc: {k: v for k, v in r.items() if k != 'episodes'}
                for sc, r in results.items()
            },
        }
        with open(out, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f'Results saved to {out}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Full Controller Comparison on Standard ADM1
============================================

Evaluates all controllers on the standard (non-thermal) ADM1 environment
across four load-variation scenarios: nominal, high_load, low_load, shock_load.

Controllers evaluated:
  Baselines  — Constant, PID, CascadedPID
  SAC        — loaded from --sac-dir if provided; skipped otherwise

Baseline controllers output 3-dim actions [q_ad, feed_mult, Q_HEX].
Only the first two dims are used here (Q_HEX is irrelevant in the std env).

Output:
  <output-dir>/
      results_std.json         — full metrics for every (controller, scenario)
      summary_std.csv          — one row per (controller, scenario), key KPIs
      <scenario>_<ctrl>.json   — per-episode JSON (same format as main pipeline)

Usage examples:
    # Baselines only
    python evaluation/evaluate_std_all.py \\
        --output-dir results/std_comparison

    # Baselines + SAC (pre-trained on std env)
    python evaluation/evaluate_std_all.py \\
        --output-dir results/std_comparison \\
        --sac-dir    results/single_scenario

    # Single controller / scenario
    python evaluation/evaluate_std_all.py \\
        --output-dir results/std_comparison \\
        --controller pid --scenario high_load
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env.adm1_gym_env_std import ADM1Env_Std, STD_SCENARIOS
from baselines.baseline_controllers import get_controller
from evaluation.metrics_calculator import MetricsCalculator


# ── Controller catalogue ──────────────────────────────────────────────────────
# Each entry: (display_name, factory_key, constructor_kwargs)
BASELINE_CONTROLLERS: List[tuple] = [
    ('Constant',    'constant',     {}),
    ('PID',         'pid',          {'K_p': 0.5, 'K_i': 0.1, 'K_d': 0.05}),
    ('CascadedPID', 'cascaded_pid', {}),
]


# ── Action adapter ────────────────────────────────────────────────────────────

def _adapt_action(action3d: np.ndarray) -> np.ndarray:
    """Trim a 3-dim baseline action to 2-dim for ADM1Env_Std."""
    return action3d[:2].astype(np.float32)


# ── Single-episode evaluation ─────────────────────────────────────────────────

def evaluate_baseline(
    controller_key: str,
    scenario_name: str,
    controller_params: Optional[Dict] = None,
    reward_config: Optional[Dict] = None,
    seed: int = 42,
    obs_mode: str = 'full',
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run one full episode of a baseline controller on ADM1Env_Std.

    Returns:
        dict — output of MetricsCalculator.compute_metrics() plus
               'controller', 'scenario', 'seed', 'obs_mode'.
    """
    env_kw = dict(scenario_name=scenario_name, obs_mode=obs_mode)
    if reward_config is not None:
        env_kw['reward_config'] = reward_config
    env = ADM1Env_Std(**env_kw)
    obs, _ = env.reset(seed=seed)

    ctrl = get_controller(controller_key, **(controller_params or {}))
    ctrl.reset()
    calc = MetricsCalculator()

    terminated = truncated = False
    while not (terminated or truncated):
        action3d = ctrl.get_action(obs)   # returns 3-dim [q_ad, feed_mult, Q_HEX]
        action2d = _adapt_action(action3d)
        obs, reward, terminated, truncated, info = env.step(action2d)
        calc.add_step(obs, action2d, reward, info)
        if terminated:
            calc.set_terminated(calc.step_count)

    env.close()

    metrics = calc.compute_metrics()
    metrics['controller'] = controller_key
    metrics['scenario']   = scenario_name
    metrics['seed']       = seed
    metrics['obs_mode']   = obs_mode
    return metrics


def evaluate_sac(
    model_path: Path,
    scenario_name: str,
    obs_mode: str = 'full',
    reward_config: Optional[Dict] = None,
    seed: int = 42,
    n_episodes: int = 3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a pre-trained SAC policy on ADM1Env_Std.

    The model must have been trained with the same obs_mode and scenario.
    Results are averaged over ``n_episodes`` independent rollouts.
    """
    try:
        from stable_baselines3 import SAC
    except ImportError:
        raise ImportError("stable-baselines3 not found — install with: pip install stable-baselines3")

    model = SAC.load(str(model_path))

    env_kw = dict(scenario_name=scenario_name, obs_mode=obs_mode)
    if reward_config is not None:
        env_kw['reward_config'] = reward_config

    all_metrics = []
    for ep in range(n_episodes):
        env = ADM1Env_Std(**env_kw)
        obs, _ = env.reset(seed=seed + ep)
        calc = MetricsCalculator()

        terminated = truncated = False
        while not (terminated or truncated):
            action2d, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action2d)
            calc.add_step(obs, action2d, reward, info)
            if terminated:
                calc.set_terminated(calc.step_count)

        env.close()
        all_metrics.append(calc.compute_metrics())

    # Average summary scalars across episodes
    if len(all_metrics) == 1:
        merged = all_metrics[0]
    else:
        merged = _average_metrics(all_metrics)

    merged['controller'] = f'SAC ({model_path.parent.name})'
    merged['scenario']   = scenario_name
    merged['seed']       = seed
    merged['obs_mode']   = obs_mode
    return merged


def _average_metrics(metrics_list: List[Dict]) -> Dict:
    """Average scalar leaves across a list of metrics dicts."""
    def _avg(vals):
        nums = [v for v in vals if isinstance(v, (int, float))]
        return float(np.mean(nums)) if nums else vals[0]

    def _merge(dicts):
        if not dicts:
            return {}
        if not isinstance(dicts[0], dict):
            return _avg(dicts)
        keys = dicts[0].keys()
        return {k: _merge([d[k] for d in dicts if k in d]) for k in keys}

    return _merge(metrics_list)


# ── Summary CSV row ───────────────────────────────────────────────────────────

def _metrics_to_csv_row(ctrl_name: str, scenario: str, m: Dict) -> Dict:
    """Extract flat KPI row for summary CSV."""
    pr = m.get('production', {})
    sf = m.get('safety', {})
    st = m.get('stability', {})
    ec = m.get('economics', {})
    sm = m.get('summary', {})
    ei = m.get('episode_info', {})
    rw = m.get('reward', {})
    return {
        'controller':           ctrl_name,
        'scenario':             scenario,
        'steps':                ei.get('steps', 0),
        'terminated_early':     int(ei.get('terminated_early', False)),
        'avg_reward':           rw.get('mean', 0.0),
        'total_ch4_m3':         pr.get('total_ch4_m3', 0.0),
        'avg_ch4_flow':         pr.get('avg_ch4_flow', 0.0),
        'violation_rate':       sf.get('violation_rate', 0.0),
        'ph_violations':        sf.get('ph_violation_count', 0),
        'vfa_violations':       sf.get('vfa_violation_count', 0),
        'nh3_violations':       sf.get('nh3_violation_count', 0),
        'avg_ph':               np.mean(m.get('_ph_values', [7.0])),
        'ch4_volatility':       st.get('ch4_volatility', 0.0),
        'net_energy_kwh':       ec.get('net_energy_kwh', 0.0),
        'overall_score':        sm.get('overall_score', 0.0),
        'production_score':     sm.get('production_score', 0.0),
        'safety_score':         sm.get('safety_score', 0.0),
    }


# ── Main runner ───────────────────────────────────────────────────────────────

def run_full_comparison(
    output_dir: Path,
    sac_dir: Optional[Path] = None,
    scenarios: Optional[List[str]] = None,
    seeds: List[int] = (42, 123, 456),
    n_sac_episodes: int = 3,
    verbose: bool = True,
) -> Dict:
    """
    Run every (controller, scenario) pair and persist results.

    Args:
        output_dir:      Directory where results are written.
        sac_dir:         Root directory of pre-trained SAC std models (optional).
                         Expected layout: <sac_dir>/sac_<scenario>_safety_first_seed<seed>/
                         or similar produced by run_experiment.py with std config.
        scenarios:       Subset of STD_SCENARIOS to run (default: all four).
        seeds:           Seeds used for averaging baseline runs.
        n_sac_episodes:  Episodes per SAC model for averaging.
        verbose:         Print progress.
    """
    scenarios = scenarios or STD_SCENARIOS
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results  = {}   # {ctrl_name: {scenario: metrics}}
    csv_rows     = []

    # ── Baselines ─────────────────────────────────────────────────────────────
    for display_name, ctrl_key, ctrl_params in BASELINE_CONTROLLERS:
        all_results.setdefault(display_name, {})

        for scenario in scenarios:
            if verbose:
                print(f"  [{display_name}] {scenario} ...", end=' ', flush=True)

            # Average over multiple seeds for robustness
            seed_metrics = []
            for seed in seeds:
                try:
                    m = evaluate_baseline(
                        ctrl_key, scenario,
                        controller_params=ctrl_params,
                        seed=seed, obs_mode='full', verbose=False,
                    )
                    seed_metrics.append(m)
                except Exception as e:
                    print(f"\n    WARNING: {display_name}/{scenario}/seed{seed} failed: {e}")

            if not seed_metrics:
                continue

            merged = _average_metrics(seed_metrics) if len(seed_metrics) > 1 else seed_metrics[0]
            merged['controller'] = display_name
            merged['scenario']   = scenario
            all_results[display_name][scenario] = merged

            # Per-episode JSON
            out_json = output_dir / f"{scenario}_{ctrl_key}.json"
            with open(out_json, 'w') as f:
                json.dump(merged, f, indent=2, default=str)

            vr = merged.get('safety', {}).get('violation_rate', 0.0)
            ch4 = merged.get('production', {}).get('avg_ch4_flow', 0.0)
            if verbose:
                print(f"viol={vr*100:.1f}%  CH4={ch4:.0f} m³/d")

            csv_rows.append(_metrics_to_csv_row(display_name, scenario, merged))

    # ── SAC models ────────────────────────────────────────────────────────────
    if sac_dir is not None and sac_dir.exists():
        if verbose:
            print(f"\n  Loading SAC models from {sac_dir} ...")

        for obs_mode_label, obs_mode in [('SAC-full', 'full'), ('SAC-simple', 'simple')]:
            all_results.setdefault(obs_mode_label, {})

            for scenario in scenarios:
                # Collect all matching model dirs for this scenario + obs_mode
                obs_suffix = '_simple' if obs_mode == 'simple' else ''
                pattern = f'sac_{scenario}_safety_first_seed*{obs_suffix}'
                model_dirs = sorted(sac_dir.glob(pattern))

                if not model_dirs:
                    if verbose:
                        print(f"  [{obs_mode_label}] {scenario}: no models found (pattern: {pattern})")
                    continue

                if verbose:
                    print(f"  [{obs_mode_label}] {scenario}: {len(model_dirs)} model(s) ...", end=' ', flush=True)

                seed_metrics = []
                for model_dir in model_dirs:
                    best_zip  = model_dir / 'best_model' / 'best_model.zip'
                    final_zip = model_dir / 'final_model.zip'
                    model_path = best_zip if best_zip.exists() else (final_zip if final_zip.exists() else None)

                    if model_path is None:
                        continue
                    try:
                        m = evaluate_sac(
                            model_path, scenario,
                            obs_mode=obs_mode,
                            seed=42,
                            n_episodes=n_sac_episodes,
                            verbose=False,
                        )
                        seed_metrics.append(m)
                    except Exception as e:
                        print(f"\n    WARNING: SAC/{scenario}/{model_dir.name} failed: {e}")

                if not seed_metrics:
                    continue

                merged = _average_metrics(seed_metrics) if len(seed_metrics) > 1 else seed_metrics[0]
                merged['controller'] = obs_mode_label
                merged['scenario']   = scenario
                all_results[obs_mode_label][scenario] = merged

                out_json = output_dir / f"{scenario}_sac_{obs_mode}.json"
                with open(out_json, 'w') as f:
                    json.dump(merged, f, indent=2, default=str)

                vr  = merged.get('safety', {}).get('violation_rate', 0.0)
                ch4 = merged.get('production', {}).get('avg_ch4_flow', 0.0)
                if verbose:
                    print(f"viol={vr*100:.1f}%  CH4={ch4:.0f} m³/d")

                csv_rows.append(_metrics_to_csv_row(obs_mode_label, scenario, merged))

    # ── Persist consolidated results ──────────────────────────────────────────
    results_json = output_dir / 'results_std.json'
    with open(results_json, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    if verbose:
        print(f"\n  Results → {results_json}")

    # Summary CSV
    if csv_rows:
        import csv
        csv_path = output_dir / 'summary_std.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        if verbose:
            print(f"  Summary  → {csv_path}")

    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Full controller comparison on standard ADM1',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--output-dir', type=str, default='results/std_comparison',
                        help='Directory for output JSON / CSV files')
    parser.add_argument('--sac-dir', type=str, default=None,
                        help='Root dir containing pre-trained SAC-std models '
                             '(optional; skip if not available)')
    parser.add_argument('--scenario', type=str, default='all',
                        choices=STD_SCENARIOS + ['all'],
                        help='Scenario to evaluate (default: all four)')
    parser.add_argument('--controller', type=str, default='all',
                        choices=['constant', 'pid', 'cascaded_pid', 'sac', 'all'],
                        help='Controller to evaluate (default: all)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                        help='Seeds for baseline averaging')
    parser.add_argument('--n-sac-episodes', type=int, default=3,
                        help='Episodes per SAC model for averaging')
    parser.add_argument('--verbose', action='store_true', default=True)
    args = parser.parse_args()

    scenarios = STD_SCENARIOS if args.scenario == 'all' else [args.scenario]
    output_dir = Path(args.output_dir)
    sac_dir    = Path(args.sac_dir) if args.sac_dir else None

    print("=" * 65)
    print("  Standard ADM1 Full Controller Comparison")
    print(f"  Scenarios:  {scenarios}")
    print(f"  Output:     {output_dir}")
    if sac_dir:
        print(f"  SAC dir:    {sac_dir}")
    print("=" * 65 + "\n")

    # Filter controller list if requested
    if args.controller != 'all':
        global BASELINE_CONTROLLERS
        BASELINE_CONTROLLERS = [
            t for t in BASELINE_CONTROLLERS if t[1] == args.controller
        ]
        if args.controller == 'sac' and sac_dir is None:
            parser.error("--controller sac requires --sac-dir")

    run_full_comparison(
        output_dir      = output_dir,
        sac_dir         = sac_dir,
        scenarios       = scenarios,
        seeds           = args.seeds,
        n_sac_episodes  = args.n_sac_episodes,
        verbose         = args.verbose,
    )

    print("\nDone.")


if __name__ == '__main__':
    main()

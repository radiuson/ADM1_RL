#!/usr/bin/env python3
"""
Full Paper Evaluation — All Controllers × All Scenarios
========================================================

Evaluates every controller on every paper scenario and writes a summary CSV
suitable for direct inclusion in Table II of the paper.

Controllers (in order):
    1. Constant        — open-loop nominal setpoint
    2. RuleBased       — hand-crafted if/else heuristics
    3. PID             — single-loop pH → feed_mult
    4. CascadedPID     — outer pH → inner VFA → feed_mult
    5. NMPC            — Greedy NMPC (N_p=8 steps, ADM1 oracle)
    6. Balanced-RL     — SAC with weak quadratic penalties (no safety shaping)
    7. SF-SAC          — Safety-First SAC (linear + constant penalty)
    8. ST-SAC          — Safety-Target SAC (strong linear, no constant)

Scenarios (paper group):
    nominal, high_load, low_load, shock_load, high_load_real, pre_stressed

Optional: counterfactual run on sep2022_crisis for RL vs. real-operator comparison.

Usage:
    # Baselines only (quick smoke-test)
    python evaluation/full_paper_eval.py \\
        --output-dir results/paper_eval

    # Full run (requires trained RL models in models_std_paper/)
    python evaluation/full_paper_eval.py \\
        --output-dir  results/paper_eval \\
        --models-dir  models_std_paper \\
        --include-nmpc

    # Also run counterfactual crisis scenario
    python evaluation/full_paper_eval.py \\
        --output-dir results/paper_eval \\
        --models-dir models_std_paper \\
        --include-nmpc --include-crisis
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env.adm1_gym_env_std import ADM1Env_Std
from baselines.baseline_controllers import get_controller
from baselines.nmpc_std_controller import NMPCStdController


# ── Constants ─────────────────────────────────────────────────────────────────

PAPER_SCENARIOS = [
    'nominal', 'high_load', 'low_load', 'shock_load',
    'high_load_real', 'pre_stressed',
]

CRISIS_SCENARIO = 'sep2022_crisis'

VFA_SOFT_LIMIT   = 0.30   # kg COD/m³  — primary safety threshold
PH_LOW_LIMIT     = 6.8
PH_HIGH_LIMIT    = 7.8
NH3_LIMIT        = 0.004  # kmol N/m³  — AD inhibition onset (~4 mM free NH3)

# ── Baseline controller specs ─────────────────────────────────────────────────
# (label, factory_key, constructor_kwargs)
BASELINE_SPECS = [
    ('Constant',    'constant',     {}),
    ('RuleBased',   'rule_based',   {}),
    ('PID',         'pid',          {'K_p': 0.5, 'K_i': 0.1, 'K_d': 0.05}),
    ('CascadedPID', 'cascaded_pid', {}),
]


# ── Metrics accumulation ──────────────────────────────────────────────────────

def _run_episode(env, controller, is_nmpc=False) -> Dict[str, Any]:
    """Run one full episode; return per-episode metrics dict."""
    obs, _ = env.reset()
    if is_nmpc:
        controller.set_env(env)

    rewards, ch4_vals, ph_vals, vfa_vals, nh3_vals = [], [], [], [], []
    n_vfa_viol = n_ph_viol = n_nh3_viol = 0
    terminated = truncated = False
    step = 0

    while not (terminated or truncated):
        if is_nmpc:
            action = controller.get_action(obs)
        else:
            raw = controller.get_action(obs)
            action = raw[:2].astype(np.float32)   # trim 3-dim to 2-dim

        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(float(reward))
        ch4_vals.append(float(info.get('q_ch4', 0.0)))
        ph_vals.append(float(info.get('pH', 7.0)))
        vfa = float(info.get('total_vfa', 0.0))
        nh3 = float(info.get('S_nh3', 0.0))
        vfa_vals.append(vfa)
        nh3_vals.append(nh3)

        ph = ph_vals[-1]
        if vfa > VFA_SOFT_LIMIT:              n_vfa_viol += 1
        if ph < PH_LOW_LIMIT or ph > PH_HIGH_LIMIT: n_ph_viol  += 1
        if nh3 > NH3_LIMIT:                   n_nh3_viol += 1
        step += 1

    n = max(step, 1)
    # Primary violation rate: VFA + pH only (both measurable from SCADA/LAB)
    # NH3 tracked separately as secondary metric
    viol_primary = n_vfa_viol + n_ph_viol
    return {
        'steps':            step,
        'terminated_early': bool(terminated and step < env.max_steps),
        'avg_reward':       float(np.mean(rewards)),
        'avg_ch4_flow':     float(np.mean(ch4_vals)),
        'total_ch4_m3':     float(np.sum(ch4_vals) * env.step_size),
        'violation_rate':   viol_primary / n,
        'vfa_viol_rate':    n_vfa_viol / n,
        'ph_viol_rate':     n_ph_viol  / n,
        'nh3_viol_rate':    n_nh3_viol / n,
        'avg_ph':           float(np.mean(ph_vals)),
        'min_ph':           float(np.min(ph_vals)),
        'max_vfa':          float(np.max(vfa_vals)),
        'avg_vfa':          float(np.mean(vfa_vals)),
        'vfa_trajectory':   vfa_vals,      # kept for counterfactual plot
        'ph_trajectory':    ph_vals,
        'ch4_trajectory':   ch4_vals,
    }


def _avg_metrics(metrics_list: List[Dict]) -> Dict:
    """Average scalar leaves over multiple seeds."""
    if len(metrics_list) == 1:
        return metrics_list[0]
    result = {}
    for key in metrics_list[0]:
        vals = [m[key] for m in metrics_list if key in m]
        if isinstance(vals[0], (int, float)):
            result[key] = float(np.mean(vals))
        elif isinstance(vals[0], bool):
            result[key] = any(vals)
        else:
            result[key] = vals[0]   # trajectories: keep first seed only
    return result


# ── Baseline evaluation ───────────────────────────────────────────────────────

def eval_baseline(
    label: str,
    ctrl_key: str,
    ctrl_kwargs: Dict,
    scenario: str,
    seeds: List[int],
    obs_mode: str = 'scada',
) -> Dict:
    ctrl = get_controller(ctrl_key, **ctrl_kwargs)
    seed_results = []
    for seed in seeds:
        env = ADM1Env_Std(scenario_name=scenario, obs_mode=obs_mode)
        env.reset(seed=seed)
        ctrl.reset()
        m = _run_episode(env, ctrl, is_nmpc=False)
        env.close()
        seed_results.append(m)
    metrics = _avg_metrics(seed_results)
    metrics['controller'] = label
    metrics['scenario']   = scenario
    return metrics


# ── NMPC evaluation ───────────────────────────────────────────────────────────

def eval_nmpc(
    scenario: str,
    seeds: List[int],
    N_p: int = 8,
    obs_mode: str = 'scada',
) -> Dict:
    # NMPC uses ADM1 rollouts internally — obs_mode only affects the env's
    # observation vector; NMPC control logic reads env state directly.
    seed_results = []
    for seed in seeds:
        env = ADM1Env_Std(scenario_name=scenario, obs_mode=obs_mode)
        ctrl = NMPCStdController(N_p=N_p)
        m = _run_episode(env, ctrl, is_nmpc=True)
        env.close()
        seed_results.append(m)
    metrics = _avg_metrics(seed_results)
    metrics['controller'] = 'NMPC'
    metrics['scenario']   = scenario
    return metrics


# ── SAC evaluation ────────────────────────────────────────────────────────────

def eval_sac(
    label: str,
    model_paths: List[Path],
    scenario: str,
    obs_mode: str = 'scada',
    n_episodes: int = 3,
) -> Dict:
    try:
        from stable_baselines3 import SAC
    except ImportError:
        raise ImportError("pip install stable-baselines3")

    seed_results = []
    for i, mp in enumerate(model_paths):
        model = SAC.load(str(mp))
        for ep in range(n_episodes):
            env = ADM1Env_Std(scenario_name=scenario, obs_mode=obs_mode)
            obs, _ = env.reset(seed=42 + i * n_episodes + ep)
            rewards, ch4_vals, ph_vals, vfa_vals, nh3_vals = [], [], [], [], []
            n_vfa = n_ph = n_nh3 = 0
            terminated = truncated = False
            step = 0
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(float(reward))
                ch4_vals.append(float(info.get('q_ch4', 0.0)))
                ph_vals.append(float(info.get('pH', 7.0)))
                vfa = float(info.get('total_vfa', 0.0))
                nh3 = float(info.get('S_nh3', 0.0))
                vfa_vals.append(vfa); nh3_vals.append(nh3)
                ph = ph_vals[-1]
                if vfa > VFA_SOFT_LIMIT:                n_vfa  += 1
                if ph < PH_LOW_LIMIT or ph > PH_HIGH_LIMIT: n_ph   += 1
                if nh3 > NH3_LIMIT:                     n_nh3  += 1
                step += 1
            env.close()
            n = max(step, 1)
            viol = n_vfa + n_ph   # VFA+pH only, consistent with _run_episode
            seed_results.append({
                'steps': step, 'terminated_early': bool(terminated and step < env.max_steps),
                'avg_reward': float(np.mean(rewards)),
                'avg_ch4_flow': float(np.mean(ch4_vals)),
                'total_ch4_m3': float(np.sum(ch4_vals) * env.step_size),
                'violation_rate': viol / n, 'vfa_viol_rate': n_vfa / n,
                'ph_viol_rate': n_ph / n, 'nh3_viol_rate': n_nh3 / n,
                'avg_ph': float(np.mean(ph_vals)), 'min_ph': float(np.min(ph_vals)),
                'max_vfa': float(np.max(vfa_vals)), 'avg_vfa': float(np.mean(vfa_vals)),
                'vfa_trajectory': vfa_vals, 'ph_trajectory': ph_vals,
                'ch4_trajectory': ch4_vals,
            })

    metrics = _avg_metrics(seed_results)
    metrics['controller'] = label
    metrics['scenario']   = scenario
    return metrics


# ── Model discovery ───────────────────────────────────────────────────────────

def _find_models(models_dir: Path, reward_config: str, algo: str = 'sac') -> List[Path]:
    """Return list of best_model.zip paths for a given algo and reward config."""
    paths = []
    prefix = f'{algo}_std_cur_{reward_config}_seed*'
    for run_dir in sorted(models_dir.glob(prefix)):
        best = run_dir / 'best_model' / 'best_model.zip'
        final = run_dir / 'final_model.zip'
        p = best if best.exists() else (final if final.exists() else None)
        if p:
            paths.append(p)
    return paths


# ── PPO evaluation ────────────────────────────────────────────────────────────

def eval_ppo(
    label: str,
    model_paths: List[Path],
    scenario: str,
    obs_mode: str = 'scada',
    n_episodes: int = 3,
) -> Dict:
    try:
        from stable_baselines3 import PPO
    except ImportError:
        raise ImportError("pip install stable-baselines3")

    seed_results = []
    for i, mp in enumerate(model_paths):
        model = PPO.load(str(mp))
        for ep in range(n_episodes):
            env = ADM1Env_Std(scenario_name=scenario, obs_mode=obs_mode)
            obs, _ = env.reset(seed=42 + i * n_episodes + ep)
            rewards, ch4_vals, ph_vals, vfa_vals, nh3_vals = [], [], [], [], []
            n_vfa = n_ph = n_nh3 = 0
            terminated = truncated = False
            step = 0
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(float(reward))
                ch4_vals.append(float(info.get('q_ch4', 0.0)))
                ph_vals.append(float(info.get('pH', 7.0)))
                vfa = float(info.get('total_vfa', 0.0))
                nh3 = float(info.get('S_nh3', 0.0))
                vfa_vals.append(vfa); nh3_vals.append(nh3)
                ph = ph_vals[-1]
                if vfa > VFA_SOFT_LIMIT:                    n_vfa  += 1
                if ph < PH_LOW_LIMIT or ph > PH_HIGH_LIMIT: n_ph   += 1
                if nh3 > NH3_LIMIT:                         n_nh3  += 1
                step += 1
            env.close()
            n = max(step, 1)
            viol = n_vfa + n_ph
            seed_results.append({
                'steps': step, 'terminated_early': bool(terminated and step < env.max_steps),
                'avg_reward': float(np.mean(rewards)),
                'avg_ch4_flow': float(np.mean(ch4_vals)),
                'total_ch4_m3': float(np.sum(ch4_vals) * env.step_size),
                'violation_rate': viol / n, 'vfa_viol_rate': n_vfa / n,
                'ph_viol_rate': n_ph / n, 'nh3_viol_rate': n_nh3 / n,
                'avg_ph': float(np.mean(ph_vals)), 'min_ph': float(np.min(ph_vals)),
                'max_vfa': float(np.max(vfa_vals)), 'avg_vfa': float(np.mean(vfa_vals)),
                'vfa_trajectory': vfa_vals, 'ph_trajectory': ph_vals,
                'ch4_trajectory': ch4_vals,
            })

    metrics = _avg_metrics(seed_results)
    metrics['controller'] = label
    metrics['scenario']   = scenario
    return metrics


# ── CSV row helper ────────────────────────────────────────────────────────────

def _to_csv_row(m: Dict) -> Dict:
    return {
        'controller':       m['controller'],
        'scenario':         m['scenario'],
        'avg_ch4_flow':     round(m.get('avg_ch4_flow', 0.0), 1),
        'total_ch4_m3':     round(m.get('total_ch4_m3', 0.0), 0),
        'violation_rate_%': round(m.get('violation_rate', 0.0) * 100, 2),
        'vfa_viol_%':       round(m.get('vfa_viol_rate', 0.0) * 100, 2),
        'ph_viol_%':        round(m.get('ph_viol_rate', 0.0) * 100, 2),
        'avg_ph':           round(m.get('avg_ph', 0.0), 3),
        'min_ph':           round(m.get('min_ph', 0.0), 3),
        'max_vfa':          round(m.get('max_vfa', 0.0), 4),
        'avg_vfa':          round(m.get('avg_vfa', 0.0), 4),
        'avg_reward':       round(m.get('avg_reward', 0.0), 4),
        'terminated_early': int(m.get('terminated_early', False)),
        'steps':            m.get('steps', 0),
    }


# ── Main runner ───────────────────────────────────────────────────────────────

def _load_checkpoint(ckpt_path: Path) -> Dict:
    """Load checkpoint dict keyed by 'controller|scenario'."""
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            return json.load(f)
    return {}


def _save_checkpoint(ckpt_path: Path, done: Dict):
    with open(ckpt_path, 'w') as f:
        json.dump(done, f)


def run_paper_eval(
    output_dir: Path,
    models_dir: Optional[Path] = None,
    scenarios: List[str]       = PAPER_SCENARIOS,
    seeds: List[int]           = (42, 123, 456),
    obs_mode: str              = 'scada',
    include_nmpc: bool         = False,
    include_crisis: bool       = False,
    nmpc_horizon: int          = 8,
    verbose: bool              = True,
    resume: bool               = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / 'checkpoint.json'

    # Load checkpoint (keyed by 'controller|scenario' → result dict)
    done_map = _load_checkpoint(ckpt_path) if resume else {}
    if done_map:
        print(f"  [resume] found {len(done_map)} completed entries in checkpoint")

    all_results  = [v for v in done_map.values()]
    csv_rows     = [_to_csv_row(m) for m in all_results]

    eval_scenarios = list(scenarios)
    if include_crisis:
        eval_scenarios.append(CRISIS_SCENARIO)

    print(f"\n{'='*70}")
    print(f"  Paper Evaluation — {len(eval_scenarios)} scenarios × controllers")
    print(f"  Scenarios: {eval_scenarios}")
    print(f"  Obs mode:  {obs_mode}")
    print(f"{'='*70}")

    def _record(m: Dict):
        key = f"{m['controller']}|{m['scenario']}"
        done_map[key] = {k: v for k, v in m.items()
                         if k not in ('vfa_trajectory', 'ph_trajectory', 'ch4_trajectory')}
        all_results.append(m)
        csv_rows.append(_to_csv_row(m))
        _save_checkpoint(ckpt_path, done_map)

    def _skip(label: str, scenario: str) -> bool:
        key = f"{label}|{scenario}"
        if key in done_map:
            if verbose:
                m = done_map[key]
                print(f"  {label:<15} [skip] viol={m['violation_rate']*100:.1f}%  "
                      f"CH4={m['avg_ch4_flow']:.0f} m³/d")
            return True
        return False

    for scenario in eval_scenarios:
        print(f"\n── Scenario: {scenario} ──────────────────────────────")

        # 1. Baselines
        for label, key, kwargs in BASELINE_SPECS:
            if _skip(label, scenario):
                continue
            if verbose:
                print(f"  {label:<15}", end=' ', flush=True)
            try:
                m = eval_baseline(label, key, kwargs, scenario, list(seeds), obs_mode=obs_mode)
                _record(m)
                if verbose:
                    print(f"viol={m['violation_rate']*100:.1f}%  "
                          f"CH4={m['avg_ch4_flow']:.0f} m³/d  "
                          f"pH={m['avg_ph']:.2f}")
            except Exception as e:
                print(f"  ERROR: {e}")

        # 2. NMPC
        if include_nmpc:
            if not _skip('NMPC', scenario):
                if verbose:
                    print(f"  {'NMPC':<15}", end=' ', flush=True)
                try:
                    m = eval_nmpc(scenario, [seeds[0]], nmpc_horizon, obs_mode=obs_mode)
                    _record(m)
                    if verbose:
                        print(f"viol={m['violation_rate']*100:.1f}%  "
                              f"CH4={m['avg_ch4_flow']:.0f} m³/d  "
                              f"pH={m['avg_ph']:.2f}")
                except Exception as e:
                    print(f"  ERROR: {e}")

        # 3. RL models (SAC)
        if models_dir is not None and models_dir.exists():
            for rl_label, config in [
                ('Balanced-RL', 'balanced'),
                ('SF-SAC',      'safety_first'),
                ('ST-SAC',      'safety_target'),
            ]:
                if _skip(rl_label, scenario):
                    continue
                model_paths = _find_models(models_dir, config, algo='sac')
                if not model_paths:
                    if verbose:
                        print(f"  {rl_label:<15} [no models found in {models_dir}]")
                    continue
                if verbose:
                    print(f"  {rl_label:<15}", end=' ', flush=True)
                try:
                    m = eval_sac(rl_label, model_paths, scenario, obs_mode)
                    _record(m)
                    if verbose:
                        print(f"viol={m['violation_rate']*100:.1f}%  "
                              f"CH4={m['avg_ch4_flow']:.0f} m³/d  "
                              f"pH={m['avg_ph']:.2f}")
                except Exception as e:
                    print(f"  ERROR: {e}")

        # 4. RL models (PPO)
        if models_dir is not None and models_dir.exists():
            for rl_label, config in [
                ('PPO-Balanced', 'balanced'),
                ('PPO-SF',       'safety_first'),
                ('PPO-ST',       'safety_target'),
            ]:
                if _skip(rl_label, scenario):
                    continue
                model_paths = _find_models(models_dir, config, algo='ppo')
                if not model_paths:
                    if verbose:
                        print(f"  {rl_label:<15} [no PPO models — run Phase 1-PPO first]")
                    continue
                if verbose:
                    print(f"  {rl_label:<15}", end=' ', flush=True)
                try:
                    m = eval_ppo(rl_label, model_paths, scenario, obs_mode)
                    _record(m)
                    if verbose:
                        print(f"viol={m['violation_rate']*100:.1f}%  "
                              f"CH4={m['avg_ch4_flow']:.0f} m³/d  "
                              f"pH={m['avg_ph']:.2f}")
                except Exception as e:
                    print(f"  ERROR: {e}")

    # ── Save final outputs ────────────────────────────────────────────────────
    json_path = output_dir / 'paper_eval_full.json'
    clean = [{k: v for k, v in m.items()
              if k not in ('vfa_trajectory', 'ph_trajectory', 'ch4_trajectory')}
             for m in all_results]
    with open(json_path, 'w') as f:
        json.dump(clean, f, indent=2)

    csv_path = output_dir / 'paper_eval_summary.csv'
    if csv_rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
    print(f"\n  Saved summary → {csv_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*80}")
    hdr = f"{'Controller':<15} {'Scenario':<18} {'CH4(m³/d)':>9} {'Viol%':>7} {'AvgpH':>7} {'MaxVFA':>8}"
    print(hdr)
    print('-' * 75)
    for r in csv_rows:
        term = ' *' if r['terminated_early'] else ''
        print(f"{r['controller']:<15} {r['scenario']:<18} "
              f"{r['avg_ch4_flow']:>9.1f} {r['violation_rate_%']:>7.2f} "
              f"{r['avg_ph']:>7.3f} {r['max_vfa']:>8.4f}{term}")

    print(f"\n  JSON  → {json_path}")
    print(f"  CSV   → {csv_path}")
    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Full paper evaluation: all controllers × all scenarios',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--output-dir',    type=str, default='results/paper_eval')
    parser.add_argument('--models-dir',    type=str, default=None,
                        help='Root dir of trained SAC models (models_std_paper/)')
    parser.add_argument('--scenario',      type=str, default='all',
                        help='Single scenario or "all"')
    parser.add_argument('--seeds',         type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--obs-mode',      type=str, default='simple',
                        choices=['scada', 'simple', 'full'])
    parser.add_argument('--include-nmpc',  action='store_true',
                        help='Run NMPC baseline (slow — uses ADM1 rollouts)')
    parser.add_argument('--include-crisis', action='store_true',
                        help='Also evaluate on sep2022_crisis counterfactual')
    parser.add_argument('--nmpc-horizon',  type=int, default=8,
                        help='NMPC prediction horizon steps (default 8 = 2 hours)')
    parser.add_argument('--no-resume',    action='store_true',
                        help='Ignore checkpoint and re-run everything from scratch')
    args = parser.parse_args()

    scenarios = PAPER_SCENARIOS if args.scenario == 'all' else [args.scenario]
    models_dir = Path(args.models_dir) if args.models_dir else None

    run_paper_eval(
        output_dir     = Path(args.output_dir),
        models_dir     = models_dir,
        scenarios      = scenarios,
        seeds          = args.seeds,
        obs_mode       = args.obs_mode,
        include_nmpc   = args.include_nmpc,
        include_crisis = args.include_crisis,
        nmpc_horizon   = args.nmpc_horizon,
        resume         = not args.no_resume,
    )
    print("\nDone.")


if __name__ == '__main__':
    main()

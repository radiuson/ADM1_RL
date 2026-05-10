#!/usr/bin/env python3
"""
ADM1 RL Experiment Runner
=========================

Standalone experiment script for:
  "Safety-First Control of Anaerobic Digestion Under Thermal Stress Using RL
   and a Temperature-Extended ADM1 Model"

Usage:
    # Run with default config
    python run_experiment.py

    # Run with custom config
    python run_experiment.py --config experiment_config.yaml

    # Override mode from command line
    python run_experiment.py --config experiment_config.yaml --mode train
    python run_experiment.py --config experiment_config.yaml --mode eval
    python run_experiment.py --config experiment_config.yaml --mode baseline
    python run_experiment.py --config experiment_config.yaml --mode full

    # Override specific settings
    python run_experiment.py --config experiment_config.yaml --seeds 42 123 456
    python run_experiment.py --config experiment_config.yaml --steps 500000

Modes:
    train        - Train RL agents on configured scenarios and seeds
    eval         - Evaluate existing trained models (cross-scenario)
    baseline     - Run baseline controller comparison only
    train_and_eval - Train then evaluate (default)
    full         - Train + eval + baseline + paper figures

Output:
    results/single_scenario/
    ├── training/
    │   ├── sac_nominal_safety_first_seed42/   (model checkpoints)
    │   ├── sac_nominal_safety_first_seed123/
    │   └── ...
    ├── evaluation/
    │   ├── cross_scenario_results.json
    │   ├── cross_scenario_results.csv
    │   └── per_run/
    ├── baselines/
    │   ├── baseline_results.json
    │   └── baseline_results.csv
    ├── figures/
    │   ├── cross_scenario_heatmap.pdf
    │   ├── safety_comparison.pdf
    │   ├── training_curves.pdf
    │   └── temperature_response.pdf
    └── tables/
        ├── main_results.tex
        └── ablation_results.tex
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings('ignore')

from env.adm1_gym_env import ADM1Env_v2
from training.reward_configs import REWARD_CONFIGS
from evaluation.metrics_calculator import MetricsCalculator
from baselines.baseline_controllers import get_controller


# ============================================================================
# Config Loading
# ============================================================================

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'experiment_config.yaml')


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def merge_cli_overrides(cfg: Dict, args: argparse.Namespace) -> Dict:
    """Override config values with CLI arguments"""
    if args.mode:
        cfg['experiment']['mode'] = args.mode
    if args.seeds:
        cfg['training']['seeds'] = args.seeds
    if args.steps:
        cfg['training']['total_timesteps'] = args.steps
    if args.output_dir:
        cfg['experiment']['output_dir'] = args.output_dir
    if args.reward_config:
        cfg['training']['reward_config'] = args.reward_config
    return cfg


# ============================================================================
# Environment Factory
# ============================================================================

def make_env(scenario: str, reward_config_name: str, seed: int = 42, obs_mode: str = 'full'):
    """Create monitored environment"""

    from stable_baselines3.common.monitor import Monitor

    reward_config = REWARD_CONFIGS[reward_config_name]
    env = ADM1Env_v2(scenario_name=scenario, reward_config=reward_config, obs_mode=obs_mode)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def make_multiscenario_env(scenarios: list, reward_config_name: str, seed: int = 42, obs_mode: str = 'full'):
    """
    Create a multi-scenario environment that randomly samples a scenario at each episode reset.
    This implements domain randomization: the policy sees all scenarios during training.
    """
    import gymnasium as gym
    import numpy as np

    from stable_baselines3.common.monitor import Monitor

    reward_config = REWARD_CONFIGS[reward_config_name]
    rng = np.random.default_rng(seed)

    class _MultiScenarioWrapper(gym.Wrapper):
        def reset(self, **kwargs):
            chosen = rng.choice(scenarios)
            kwargs.setdefault('options', {})['scenario'] = chosen
            return self.env.reset(**kwargs)

    base_env = ADM1Env_v2(scenario_name=scenarios[0], reward_config=reward_config, obs_mode=obs_mode)
    env = _MultiScenarioWrapper(base_env)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


# ============================================================================
# Training
# ============================================================================

def _make_ent_coef_clamp_callback(max_ent_coef: float = 5.0, min_ent_coef: float = 0.01):
    """
    Returns a SB3 BaseCallback that clamps SAC log_ent_coef after every training step.

    Mathematical basis: Projected gradient descent on the dual variable α.
    Equivalent to constrained dual optimisation:
        max E[Q(s,a)] - α·(H_target - E[-log π])
        subject to  α_min ≤ α ≤ α_max

    Strictly stronger than a fixed ent_coef (which disables adaptation entirely)
    while preventing the NaN explosion caused by unconstrained auto-tuning when
    the policy collapses on violation-heavy scenarios (critic_loss→inf→NaN).

    Args:
        max_ent_coef: Upper bound on α (default 5.0, generous for 3-dim actions).
        min_ent_coef: Lower bound on α (default 0.01, prevents α→0 collapse).
    """
    from stable_baselines3.common.callbacks import BaseCallback
    import torch

    _max_log = torch.log(torch.tensor(max_ent_coef, dtype=torch.float32)).item()
    _min_log = torch.log(torch.tensor(min_ent_coef, dtype=torch.float32)).item()

    class _Clamp(BaseCallback):
        def _on_step(self) -> bool:
            if hasattr(self.model, 'log_ent_coef') and self.model.log_ent_coef is not None:
                with torch.no_grad():
                    self.model.log_ent_coef.data.clamp_(min=_min_log, max=_max_log)
            return True

    return _Clamp(verbose=0)


def train_single(
    scenario: str,
    reward_config_name: str,
    seed: int,
    algo: str,
    total_timesteps: int,
    output_dir: Path,
    hyperparams_cfg: Dict,
    eval_freq: int = 10000,
    save_freq: int = 50000,
    n_eval_episodes: int = 5,
    device: str = 'auto',
    verbose: int = 1,
    obs_mode: str = 'full',
) -> Path:
    """
    Train a single RL agent.

    Returns:
        Path to the log directory containing the trained model.
    """
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

    ALGO_CLASSES = {'ppo': PPO, 'sac': SAC, 'td3': TD3}

    # ── Build hyperparameter dict ────────────────────────────────────────────
    hp = _build_hyperparams(algo, hyperparams_cfg)

    # ── Directories ──────────────────────────────────────────────────────────
    obs_suffix = f"_{obs_mode}" if obs_mode != 'full' else ''
    scenario_key = '_'.join(scenario) if isinstance(scenario, list) else scenario
    run_name = f"{algo}_{scenario_key}_{reward_config_name}_seed{seed}{obs_suffix}"
    log_dir = output_dir / 'training' / run_name

    # ── Skip if already complete ──────────────────────────────────────────────
    if (log_dir / 'final_model.zip').exists():
        print(f"\n  [SKIP] {run_name} — final_model.zip already exists")
        return log_dir

    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / 'checkpoints').mkdir(exist_ok=True)
    (log_dir / 'best_model').mkdir(exist_ok=True)
    (log_dir / 'eval').mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  TRAINING: {run_name}")
    print(f"{'='*70}")
    print(f"  Steps:   {total_timesteps:,}")
    print(f"  Seed:    {seed}")
    print(f"  OutDir:  {log_dir}")

    # ── Environments ─────────────────────────────────────────────────────────
    if isinstance(scenario, list):
        # Multi-scenario (domain randomization): random scenario per episode
        train_env = make_multiscenario_env(scenario, reward_config_name, seed,     obs_mode=obs_mode)
        eval_env  = make_env(scenario[0], reward_config_name, seed + 1, obs_mode=obs_mode)
    else:
        train_env = make_env(scenario, reward_config_name, seed,     obs_mode=obs_mode)
        eval_env  = make_env(scenario, reward_config_name, seed + 1, obs_mode=obs_mode)

    # ── Model ────────────────────────────────────────────────────────────────
    AlgoClass = ALGO_CLASSES[algo]
    model = AlgoClass(
        "MlpPolicy",
        train_env,
        verbose=verbose,
        seed=seed,
        device=device,
        tensorboard_log=str(log_dir / 'tensorboard'),
        **hp,
    )

    # ── Callbacks ────────────────────────────────────────────────────────────
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir / 'best_model'),
        log_path=str(log_dir / 'eval'),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(log_dir / 'checkpoints'),
        name_prefix='rl_model',
    )

    # ── Train ────────────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList([eval_cb, ckpt_cb, _make_ent_coef_clamp_callback(
                max_ent_coef=hp.get('ent_coef_max', 5.0) if isinstance(hp.get('ent_coef'), str) else 1e9,
            )]),
            progress_bar=(verbose >= 1),
        )
    except KeyboardInterrupt:
        print("\n  [Training interrupted]")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed/60:.1f} min")

    # ── Save ─────────────────────────────────────────────────────────────────
    model.save(str(log_dir / 'final_model'))

    # Save run metadata
    meta = {
        'algo': algo,
        'scenario': scenario,
        'reward_config': reward_config_name,
        'seed': seed,
        'total_timesteps': total_timesteps,
        'elapsed_seconds': elapsed,
        'hyperparams': {k: str(v) if isinstance(v, dict) else v for k, v in hp.items()},
    }
    with open(log_dir / 'run_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    train_env.close()
    eval_env.close()

    print(f"  Model saved → {log_dir}")
    return log_dir


def run_training(cfg: Dict, output_dir: Path) -> List[Path]:
    """Train all (obs_mode × scenario × seed) combinations. Returns list of log dirs."""
    tcfg = cfg['training']
    algo            = tcfg['algorithm']
    reward_config   = tcfg['reward_config']
    total_timesteps = tcfg['total_timesteps']
    seeds           = tcfg['seeds']
    train_scenarios = tcfg['train_scenarios']
    eval_freq       = tcfg.get('eval_freq', 10000)
    save_freq       = tcfg.get('save_freq', 50000)
    n_eval_episodes = tcfg.get('n_eval_episodes', 5)
    device          = tcfg.get('device', 'auto')
    verbose         = tcfg.get('verbose', 1)

    # Support obs_modes (list) or obs_mode (scalar, backward compat)
    obs_modes = tcfg.get('obs_modes') or [tcfg.get('obs_mode', 'full')]

    hp_key = f'{algo}_hyperparams'
    hp_cfg = tcfg.get(hp_key, {})

    total_runs = len(obs_modes) * len(train_scenarios) * len(seeds)
    print(f"\n{'#'*70}")
    print(f"  TRAINING PHASE")
    print(f"  Obs modes:  {obs_modes}")
    print(f"  Scenarios:  {train_scenarios}")
    print(f"  Seeds:      {seeds}")
    print(f"  Total runs: {total_runs}")
    print(f"{'#'*70}")

    log_dirs = []
    for obs_mode in obs_modes:
        for scenario in train_scenarios:
            # Support "multi:[s1,s2,...]" string or plain list from YAML
            if isinstance(scenario, str) and scenario.startswith('multi:'):
                scenario = [s.strip() for s in scenario[6:].split(',')]
            for seed in seeds:
                log_dir = train_single(
                    scenario=scenario,
                    reward_config_name=reward_config,
                    seed=seed,
                    algo=algo,
                    total_timesteps=total_timesteps,
                    output_dir=output_dir,
                    hyperparams_cfg=hp_cfg,
                    eval_freq=eval_freq,
                    save_freq=save_freq,
                    n_eval_episodes=n_eval_episodes,
                    device=device,
                    verbose=verbose,
                    obs_mode=obs_mode,
                )
                log_dirs.append(log_dir)

    print(f"\n{'#'*70}")
    print(f"  TRAINING COMPLETE — {len(log_dirs)} runs finished")
    print(f"{'#'*70}")
    return log_dirs


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model_on_scenario(
    model_path: str,
    test_scenario: str,
    reward_config_name: str,
    num_steps: int = 2880,
    n_eval_episodes: int = 10,
    seed: int = 42,
    deterministic: bool = True,
    algo: str = 'sac',
    obs_mode: str = 'full',
) -> Dict:
    """
    Evaluate a trained model on a single test scenario.

    Returns a metrics dict with mean/std over n_eval_episodes.
    """

    from stable_baselines3 import PPO, SAC, TD3
    ALGO_CLASSES = {'ppo': PPO, 'sac': SAC, 'td3': TD3}

    AlgoClass = ALGO_CLASSES[algo]

    # Load model (add .zip if missing)
    mp = model_path if model_path.endswith('.zip') else model_path + '.zip'
    if not os.path.exists(mp):
        print(f"  [WARN] Model not found: {mp}")
        return {}

    model = AlgoClass.load(mp)

    # Run n_eval_episodes
    episode_metrics = []
    for ep in range(n_eval_episodes):
        ep_seed = seed + ep * 100

        reward_config = REWARD_CONFIGS[reward_config_name]
        env = ADM1Env_v2(scenario_name=test_scenario, reward_config=reward_config, obs_mode=obs_mode)
        obs, info = env.reset(seed=ep_seed)

        calc = MetricsCalculator()
        for step in range(num_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            calc.add_step(obs, action, reward, info)

            if terminated:
                calc.set_terminated(step)
                break
            if truncated:
                break

        metrics = calc.compute_metrics()
        episode_metrics.append(metrics)
        env.close()

    # Aggregate over episodes
    aggregated = _aggregate_episode_metrics(episode_metrics)
    return aggregated


def run_evaluation(cfg: Dict, output_dir: Path, log_dirs: Optional[List[Path]] = None):
    """
    Cross-scenario evaluation of all trained models.

    Iterates over obs_modes × trained_on × seeds × test_on.
    If log_dirs is None, scans output_dir/training/ for existing models.
    """
    ecfg   = cfg['evaluation']
    tcfg   = cfg['training']
    algo   = tcfg['algorithm']
    rc     = tcfg['reward_config']
    seeds  = tcfg['seeds']

    # Support obs_modes (list) or obs_mode (scalar, backward compat)
    obs_modes = tcfg.get('obs_modes') or [tcfg.get('obs_mode', 'full')]

    trained_on  = ecfg.get('trained_on', tcfg['train_scenarios'])
    test_on     = ecfg.get('test_on', tcfg['train_scenarios'])
    n_ep        = ecfg.get('n_eval_episodes', 10)
    num_steps   = ecfg.get('num_steps', 2880)
    seed        = ecfg.get('seed', 42)
    use_best    = ecfg.get('use_best_model', True)
    verbose     = ecfg.get('verbose', True)

    eval_dir = output_dir / 'evaluation' / 'per_run'
    eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"  EVALUATION PHASE (cross-scenario)")
    print(f"  Obs modes:  {obs_modes}")
    print(f"  Trained on: {trained_on}")
    print(f"  Test on:    {test_on}")
    print(f"  Seeds:      {seeds}")
    print(f"{'#'*70}")

    all_records = []

    for obs_mode in obs_modes:
        obs_suffix = f"_{obs_mode}" if obs_mode != 'full' else ''
        for train_scenario in trained_on:
            # Resolve "multi:s1,s2,..." to list (matches directory naming in train_single)
            if isinstance(train_scenario, str) and train_scenario.startswith('multi:'):
                train_scenario = [s.strip() for s in train_scenario[6:].split(',')]
            for seed_val in seeds:
                run_name = f"{algo}_{train_scenario}_{rc}_seed{seed_val}{obs_suffix}"
                model_base = output_dir / 'training' / run_name

                # Choose model file
                if use_best:
                    mp = str(model_base / 'best_model' / 'best_model')
                else:
                    mp = str(model_base / 'final_model')

                if not os.path.exists(mp + '.zip'):
                    if verbose:
                        print(f"  [SKIP] No model at: {mp}.zip")
                    continue

                for test_scenario in test_on:
                    per_run_path = eval_dir / f"{run_name}_on_{test_scenario}.json"
                    if per_run_path.exists():
                        with open(per_run_path) as _f:
                            all_records.append(json.load(_f)['record'])
                        continue

                    if verbose:
                        print(f"  [{obs_mode}] Eval: {run_name}  →  {test_scenario}")

                    metrics = evaluate_model_on_scenario(
                        model_path=mp,
                        test_scenario=test_scenario,
                        reward_config_name=rc,
                        num_steps=num_steps,
                        n_eval_episodes=n_ep,
                        seed=seed,
                        algo=algo,
                        obs_mode=obs_mode,
                    )

                    if not metrics:
                        continue

                    record = {
                        'obs_mode':           obs_mode,
                        'train_scenario':     train_scenario,
                        'test_scenario':      test_scenario,
                        'algo':               algo,
                        'reward_config':      rc,
                        'seed':               seed_val,
                        # Core metrics
                        'reward_mean':        metrics.get('reward_mean', np.nan),
                        'reward_std':         metrics.get('reward_std', np.nan),
                        'ch4_avg':            metrics.get('ch4_avg', np.nan),
                        'ch4_std':            metrics.get('ch4_std', np.nan),
                        'violation_rate':     metrics.get('violation_rate', np.nan),
                        'violation_rate_std': metrics.get('violation_rate_std', np.nan),
                        'overall_score':      metrics.get('overall_score', np.nan),
                        'terminated_rate':    metrics.get('terminated_rate', np.nan),
                        'ph_mean':            metrics.get('ph_mean', np.nan),
                        'vfa_max_mean':       metrics.get('vfa_max_mean', np.nan),
                        # Risk metrics
                        'vfa_cvar95':         metrics.get('vfa_cvar95', np.nan),
                        'vfa_worst_episode':  metrics.get('vfa_worst_episode', np.nan),
                        'score_worst':        metrics.get('score_worst', np.nan),
                        'score_cvar95':       metrics.get('score_cvar95', np.nan),
                    }
                    all_records.append(record)

                    # Save per-run JSON
                    with open(per_run_path, 'w') as f:
                        json.dump({'record': record, 'raw': metrics}, f, indent=2)

    if not all_records:
        print("  [WARN] No evaluation results produced.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # Save aggregated results
    df.to_csv(output_dir / 'evaluation' / 'cross_scenario_results.csv', index=False)
    df.to_json(output_dir / 'evaluation' / 'cross_scenario_results.json', orient='records', indent=2)

    _print_eval_summary(df)

    print(f"\n  Results saved → {output_dir / 'evaluation'}")
    return df


# ============================================================================
# Baseline Evaluation
# ============================================================================

_THERMAL_SCENARIOS = {'temperature_drop', 'cold_winter'}

# Thermal controller specs added automatically for thermal scenarios.
# Q_HEX_bias is scenario-specific (cold_winter needs ~2400 W to hold 35°C at 5°C ambient).
_THERMAL_CTRL_SPECS = [
    {'name': 'ConstantThermal', 'type': 'constant_thermal'},
    {'name': 'ThermalPID',      'type': 'thermal_pid'},
    {'name': 'FullPID',         'type': 'full_pid'},
]


def run_baseline(cfg: Dict, output_dir: Path) -> pd.DataFrame:
    """Evaluate all baseline controllers across configured scenarios.

    For thermal scenarios (temperature_drop, cold_winter) thermal controllers
    (ConstantThermal, ThermalPID, FullPID) are automatically appended with
    scenario-appropriate Q_HEX_bias values.
    """

    bcfg        = cfg.get('baseline', {})
    tcfg        = cfg['training']
    rc_name     = tcfg['reward_config']
    scenarios   = bcfg.get('scenarios', tcfg['train_scenarios'])
    controllers = bcfg.get('controllers', [])
    n_ep        = bcfg.get('n_eval_episodes', 5)
    num_steps   = bcfg.get('num_steps', 2880)
    seed        = bcfg.get('seed', 42)

    base_dir = output_dir / 'baselines'
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"  BASELINE EVALUATION")
    print(f"  Scenarios:   {scenarios}")
    print(f"  Controllers: {[c['name'] for c in controllers]} + thermal (for thermal scenarios)")
    print(f"{'#'*70}")

    all_records = []

    for scenario in scenarios:
        # Build controller list for this scenario
        ctrl_list = list(controllers)
        if scenario in _THERMAL_SCENARIOS:
            Q_HEX_bias = 2400.0 if scenario == 'cold_winter' else 500.0
            for spec in _THERMAL_CTRL_SPECS:
                ctrl_list.append({
                    'name':   spec['name'],
                    'type':   spec['type'],
                    'params': {'Q_HEX_bias': Q_HEX_bias},
                })

        for ctrl_cfg in ctrl_list:
            ctrl_name = ctrl_cfg['name']
            ctrl_type = ctrl_cfg['type']
            ctrl_params = ctrl_cfg.get('params', {})

            print(f"  {ctrl_name} on {scenario} ...")

            episode_metrics = []
            for ep in range(n_ep):
                ep_seed = seed + ep * 100
                reward_config = REWARD_CONFIGS[rc_name]
                env = ADM1Env_v2(scenario_name=scenario, reward_config=reward_config)
                obs, info = env.reset(seed=ep_seed)

                controller = get_controller(ctrl_type, **ctrl_params)
                controller.reset()

                # Detect env action dimension once per episode
                action_dim = env.action_space.shape[0]

                calc = MetricsCalculator()
                for step in range(num_steps):
                    action = controller.get_action(obs)
                    # Baselines output 2-dim [q_ad, feed_mult].
                    # If env expects 3-dim (with Q_HEX), pad with 0 (no heating).
                    if len(action) < action_dim:
                        action = np.append(action, [0.0] * (action_dim - len(action)))
                    obs, reward, terminated, truncated, info = env.step(action)
                    calc.add_step(obs, action, reward, info)
                    if terminated:
                        calc.set_terminated(step)
                        break
                    if truncated:
                        break

                metrics = calc.compute_metrics()
                episode_metrics.append(metrics)
                env.close()

            agg = _aggregate_episode_metrics(episode_metrics)

            record = {
                'controller':    ctrl_name,
                'scenario':      scenario,
                'reward_config': rc_name,
                'reward_mean':   agg.get('reward_mean', np.nan),
                'reward_std':    agg.get('reward_std', np.nan),
                'ch4_avg':       agg.get('ch4_avg', np.nan),
                'ch4_std':       agg.get('ch4_std', np.nan),
                'violation_rate': agg.get('violation_rate', np.nan),
                'overall_score': agg.get('overall_score', np.nan),
                'terminated_rate': agg.get('terminated_rate', np.nan),
                'ph_mean':       agg.get('ph_mean', np.nan),
                'vfa_max_mean':  agg.get('vfa_max_mean', np.nan),
            }
            all_records.append(record)

    df = pd.DataFrame(all_records)
    df.to_csv(base_dir / 'baseline_results.csv', index=False)
    df.to_json(base_dir / 'baseline_results.json', orient='records', indent=2)

    _print_baseline_summary(df)
    print(f"\n  Results saved → {base_dir}")
    return df


# ============================================================================
# Figure Generation
# ============================================================================

def generate_figures(cfg: Dict, output_dir: Path, eval_df: Optional[pd.DataFrame] = None,
                     baseline_df: Optional[pd.DataFrame] = None):
    """Generate paper-ready figures."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  [WARN] matplotlib/seaborn not available — skipping figures")
        return

    ocfg   = cfg.get('output', {})
    dpi    = ocfg.get('figure_dpi', 300)
    fmt    = ocfg.get('figure_format', 'pdf')
    style  = ocfg.get('figure_style', 'seaborn-v0_8-paper')

    try:
        plt.style.use(style)
    except Exception:
        plt.style.use('seaborn-v0_8-whitegrid')

    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── (1) Cross-scenario heatmap ────────────────────────────────────────────
    if eval_df is not None and not eval_df.empty:
        _plot_cross_scenario_heatmap(eval_df, fig_dir, fmt, dpi)
        _plot_training_seed_comparison(eval_df, fig_dir, fmt, dpi)

    # ── (2) RL vs Baseline comparison ────────────────────────────────────────
    if eval_df is not None and baseline_df is not None and not eval_df.empty and not baseline_df.empty:
        _plot_rl_vs_baseline(eval_df, baseline_df, fig_dir, fmt, dpi)

    # ── (3) Safety comparison ────────────────────────────────────────────────
    if eval_df is not None and not eval_df.empty:
        _plot_safety_comparison(eval_df, fig_dir, fmt, dpi)

    print(f"\n  Figures saved → {fig_dir}")


def _plot_cross_scenario_heatmap(df: pd.DataFrame, fig_dir: Path, fmt: str, dpi: int):
    """Overall score heatmap: rows=train_scenario, cols=test_scenario.
    Generates one heatmap per obs_mode if the column is present."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        obs_modes = df['obs_mode'].unique().tolist() if 'obs_mode' in df.columns else ['full']
        ncols = len(obs_modes)
        fig, axes = plt.subplots(1, ncols, figsize=(10 * ncols, 5), squeeze=False)

        for col_idx, obs_mode in enumerate(obs_modes):
            ax = axes[0][col_idx]
            sub = df[df['obs_mode'] == obs_mode] if 'obs_mode' in df.columns else df
            pivot = sub.groupby(['train_scenario', 'test_scenario'])['overall_score'].mean().unstack()
            sns.heatmap(
                pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=1, ax=ax,
                linewidths=0.5, linecolor='gray'
            )
            ax.set_title(f'Cross-Scenario Generalization\n({obs_mode} obs)', fontsize=12)
            ax.set_xlabel('Test Scenario')
            ax.set_ylabel('Train Scenario')

        plt.tight_layout()
        fig.savefig(fig_dir / f'cross_scenario_heatmap.{fmt}', dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: cross_scenario_heatmap.{fmt}")
    except Exception as e:
        print(f"  [WARN] Heatmap failed: {e}")


def _plot_training_seed_comparison(df: pd.DataFrame, fig_dir: Path, fmt: str, dpi: int):
    """Bar chart showing mean±std across seeds for same-scenario evaluation.
    If obs_mode column present, generates one row of subplots per obs_mode."""
    try:
        import matplotlib.pyplot as plt

        obs_modes = df['obs_mode'].unique().tolist() if 'obs_mode' in df.columns else ['full']
        nrows = len(obs_modes)
        metrics = ['overall_score', 'violation_rate']

        fig, axes = plt.subplots(nrows, 2, figsize=(12, 5 * nrows), squeeze=False)

        for row_idx, obs_mode in enumerate(obs_modes):
            sub = df[df['obs_mode'] == obs_mode] if 'obs_mode' in df.columns else df
            same = sub[sub['train_scenario'] == sub['test_scenario']].copy()

            for col_idx, metric in enumerate(metrics):
                ax = axes[row_idx][col_idx]
                if same.empty:
                    ax.set_visible(False)
                    continue
                grouped = same.groupby('train_scenario')[metric].agg(['mean', 'std'])
                x = range(len(grouped))
                ax.bar(x, grouped['mean'], yerr=grouped['std'], capsize=5,
                       color='steelblue', alpha=0.8, label=f'SAC ({obs_mode})')
                ax.set_xticks(list(x))
                ax.set_xticklabels(grouped.index, rotation=20, ha='right')
                ax.set_title(f"[{obs_mode}] {metric.replace('_', ' ').title()}")
                ax.set_ylabel(metric)
                ax.legend()

        plt.suptitle('Performance across Seeds (mean ± std)', fontsize=12)
        plt.tight_layout()
        fig.savefig(fig_dir / f'seed_comparison.{fmt}', dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: seed_comparison.{fmt}")
    except Exception as e:
        print(f"  [WARN] Seed comparison plot failed: {e}")


def _plot_rl_vs_baseline(eval_df: pd.DataFrame, baseline_df: pd.DataFrame,
                          fig_dir: Path, fmt: str, dpi: int):
    """Grouped bar chart: RL vs baselines per scenario.
    If obs_mode column present, generates one figure per obs_mode."""
    try:
        import matplotlib.pyplot as plt

        obs_modes = eval_df['obs_mode'].unique().tolist() if 'obs_mode' in eval_df.columns else ['full']
        base_agg = baseline_df.groupby(['scenario', 'controller'])[
            ['overall_score', 'violation_rate']
        ].mean().reset_index()
        base_agg = base_agg.rename(columns={'scenario': 'test_scenario'})
        all_ctrl_names = ['SAC (ours)'] + sorted(base_agg['controller'].unique().tolist())
        colors = plt.cm.Set2.colors  # type: ignore

        for obs_mode in obs_modes:
            sub = eval_df[eval_df['obs_mode'] == obs_mode] if 'obs_mode' in eval_df.columns else eval_df

            rl_agg = sub[sub['train_scenario'] == sub['test_scenario']].copy()
            rl_agg = rl_agg.groupby('test_scenario')[['overall_score', 'violation_rate']].mean()

            common_scenarios = sorted(set(rl_agg.index) & set(base_agg['test_scenario']))
            if not common_scenarios:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            width = 0.8 / len(all_ctrl_names)
            x = np.arange(len(common_scenarios))

            for metric, ax in zip(['overall_score', 'violation_rate'], axes):
                for i, ctrl in enumerate(all_ctrl_names):
                    if ctrl == 'SAC (ours)':
                        vals = [rl_agg.loc[s, metric] if s in rl_agg.index else np.nan
                                for s in common_scenarios]
                    else:
                        vals = [base_agg[(base_agg['test_scenario'] == s) &
                                         (base_agg['controller'] == ctrl)][metric].mean()
                                for s in common_scenarios]
                    ax.bar(x + i * width, vals, width, label=ctrl,
                           color=colors[i % len(colors)], alpha=0.85)

                ax.set_xticks(x + width * (len(all_ctrl_names) - 1) / 2)
                ax.set_xticklabels(common_scenarios, rotation=20, ha='right')
                ax.set_title(metric.replace('_', ' ').title())
                ax.set_ylabel(metric)
                ax.legend(fontsize=7)

            plt.suptitle(f'SAC [{obs_mode}] vs Baseline Controllers', fontsize=13)
            plt.tight_layout()
            fname = f'rl_vs_baseline_{obs_mode}.{fmt}'
            fig.savefig(fig_dir / fname, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {fname}")
    except Exception as e:
        print(f"  [WARN] RL vs baseline plot failed: {e}")


def _plot_safety_comparison(df: pd.DataFrame, fig_dir: Path, fmt: str, dpi: int):
    """Scatter: CH4 production vs violation rate (efficiency-safety trade-off)"""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))

        for train_s, grp in df.groupby('train_scenario'):
            ax.scatter(
                grp['violation_rate'] * 100,
                grp['ch4_avg'],
                label=f'Train={train_s}',
                alpha=0.7, s=60,
            )

        ax.set_xlabel('Violation Rate (%)', fontsize=11)
        ax.set_ylabel('Avg CH4 Flow (m³/day)', fontsize=11)
        ax.set_title('Safety–Production Trade-off', fontsize=12)
        ax.legend()
        plt.tight_layout()
        fig.savefig(fig_dir / f'safety_production_tradeoff.{fmt}', dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: safety_production_tradeoff.{fmt}")
    except Exception as e:
        print(f"  [WARN] Safety comparison plot failed: {e}")


# ============================================================================
# LaTeX Table Generation
# ============================================================================

def generate_latex_tables(cfg: Dict, output_dir: Path,
                           eval_df: Optional[pd.DataFrame] = None,
                           baseline_df: Optional[pd.DataFrame] = None):
    """Export results as LaTeX tables for paper."""
    dp = cfg.get('output', {}).get('decimal_places', 3)
    tables_dir = output_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)

    # ── Main results table ───────────────────────────────────────────────────
    if eval_df is not None and baseline_df is not None and not eval_df.empty and not baseline_df.empty:
        _write_main_results_table(eval_df, baseline_df, tables_dir, dp)

    # ── Cross-scenario generalization table ──────────────────────────────────
    if eval_df is not None and not eval_df.empty:
        _write_generalization_table(eval_df, tables_dir, dp)

    print(f"\n  LaTeX tables saved → {tables_dir}")


def _write_main_results_table(eval_df: pd.DataFrame, baseline_df: pd.DataFrame,
                               tables_dir: Path, dp: int):
    """Table: Controller × Scenario → Score / Violation Rate / CH4.
    If obs_mode column present, generates one table per obs_mode."""
    fmt = f'.{dp}f'

    obs_modes = eval_df['obs_mode'].unique().tolist() if 'obs_mode' in eval_df.columns else ['full']

    for obs_mode in obs_modes:
        sub_eval = eval_df[eval_df['obs_mode'] == obs_mode] if 'obs_mode' in eval_df.columns else eval_df

        # RL rows (mean ± std across seeds, same-scenario)
        rl_rows = []
        same = sub_eval[sub_eval['train_scenario'] == sub_eval['test_scenario']]
        for (_, test_s), grp in same.groupby(['train_scenario', 'test_scenario']):
            rl_rows.append({
                'Controller': f'SAC [{obs_mode}]',
                'Scenario': test_s,
                'Score': f"{grp['overall_score'].mean():{fmt}} $\\pm$ {grp['overall_score'].std():{fmt}}",
                'Violation%': f"{grp['violation_rate'].mean()*100:{fmt}}",
                'CH4': f"{grp['ch4_avg'].mean():{fmt}}",
            })

        # Baseline rows (same for all obs_modes)
        base_rows = []
        for (ctrl, scen), grp in baseline_df.groupby(['controller', 'scenario']):
            base_rows.append({
                'Controller': ctrl,
                'Scenario': scen,
                'Score': f"{grp['overall_score'].mean():{fmt}}",
                'Violation%': f"{grp['violation_rate'].mean()*100:{fmt}}",
                'CH4': f"{grp['ch4_avg'].mean():{fmt}}",
            })

        all_rows = rl_rows + base_rows
        if not all_rows:
            continue

        df = pd.DataFrame(all_rows).sort_values(['Scenario', 'Controller'])

        lines = []
        lines.append(r'\begin{table}[htbp]')
        lines.append(r'\centering')
        lines.append(rf'\caption{{Main Evaluation Results: SAC [{obs_mode}] vs Baseline Controllers}}')
        lines.append(rf'\label{{tab:main_results_{obs_mode}}}')
        lines.append(r'\begin{tabular}{llccc}')
        lines.append(r'\toprule')
        lines.append(r'Controller & Scenario & Score & Violation Rate (\%) & CH4 (m³/d) \\')
        lines.append(r'\midrule')
        for _, row in df.iterrows():
            lines.append(f"{row['Controller']} & {row['Scenario']} & {row['Score']} & {row['Violation%']} & {row['CH4']} \\\\")
        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        lines.append(r'\end{table}')

        fname = f'main_results_{obs_mode}.tex'
        with open(tables_dir / fname, 'w') as f:
            f.write('\n'.join(lines))
        print(f"  Saved: {fname}")


def _write_generalization_table(eval_df: pd.DataFrame, tables_dir: Path, dp: int):
    """Heatmap table: train×test overall_score. One table per obs_mode."""
    fmt = f'.{dp}f'
    obs_modes = eval_df['obs_mode'].unique().tolist() if 'obs_mode' in eval_df.columns else ['full']

    for obs_mode in obs_modes:
        sub = eval_df[eval_df['obs_mode'] == obs_mode] if 'obs_mode' in eval_df.columns else eval_df
        pivot = sub.groupby(['train_scenario', 'test_scenario'])['overall_score'].mean().unstack()

        test_scenarios = list(pivot.columns)
        lines = []
        lines.append(r'\begin{table}[htbp]')
        lines.append(r'\centering')
        lines.append(rf'\caption{{Cross-Scenario Generalization: Overall Score [{obs_mode} obs]}}')
        lines.append(rf'\label{{tab:generalization_{obs_mode}}}')
        lines.append(r'\begin{tabular}{l' + 'c' * len(test_scenarios) + r'}')
        lines.append(r'\toprule')
        lines.append(r'Train $\backslash$ Test & ' + ' & '.join(test_scenarios) + r' \\')
        lines.append(r'\midrule')
        for train_s, row in pivot.iterrows():
            vals = ' & '.join(f"{v:{fmt}}" if not np.isnan(v) else '—' for v in row.values)
            lines.append(f"{train_s} & {vals} \\\\")
        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        lines.append(r'\end{table}')

        fname = f'generalization_table_{obs_mode}.tex'
        with open(tables_dir / fname, 'w') as f:
            f.write('\n'.join(lines))
        print(f"  Saved: {fname}")


# ============================================================================
# Helpers
# ============================================================================

def _build_hyperparams(algo: str, hp_cfg: Dict) -> Dict:
    """Merge algorithm defaults with config overrides."""
    DEFAULTS = {
        'ppo': {
            'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 64,
            'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95,
            'clip_range': 0.2, 'ent_coef': 0.0, 'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'policy_kwargs': dict(net_arch=[256, 256]),
        },
        'sac': {
            'learning_rate': 3e-4, 'buffer_size': 1_000_000,
            'batch_size': 256, 'tau': 0.005, 'gamma': 0.99,
            'learning_starts': 10_000, 'train_freq': 1,
            'gradient_steps': 1, 'ent_coef': 'auto',
            'policy_kwargs': dict(net_arch=[256, 256]),
        },
        'td3': {
            'learning_rate': 1e-3, 'buffer_size': 1_000_000,
            'batch_size': 100, 'tau': 0.005, 'gamma': 0.99,
            'learning_starts': 10_000, 'train_freq': 1,
            'gradient_steps': 1, 'policy_delay': 2,
            'target_policy_noise': 0.2, 'target_noise_clip': 0.5,
            'policy_kwargs': dict(net_arch=[256, 256]),
        },
    }

    hp = DEFAULTS.get(algo, {}).copy()

    # Apply overrides from config
    for k, v in hp_cfg.items():
        if k == 'net_arch':
            hp['policy_kwargs'] = dict(net_arch=v)
        else:
            hp[k] = v

    return hp


def _aggregate_episode_metrics(episode_metrics: List[Dict]) -> Dict:
    """Aggregate a list of per-episode metric dicts into mean/std."""
    if not episode_metrics:
        return {}

    def extract(key_path: str):
        """Navigate nested dict with dot-separated key path."""
        vals = []
        for m in episode_metrics:
            v = m
            for k in key_path.split('.'):
                v = v.get(k, {}) if isinstance(v, dict) else np.nan
            vals.append(float(v) if not isinstance(v, dict) else np.nan)
        return np.array(vals, dtype=float)

    terminated = extract('episode_info.terminated_early')
    vfa_rates  = extract('safety.violation_rate')
    scores     = extract('summary.overall_score')
    rewards    = extract('reward.mean')

    # CVaR-95: conditional value at risk at 95% level
    # = mean of the worst 5% of episodes (or at least 1 episode)
    def _cvar(arr, alpha=0.05):
        arr_clean = arr[~np.isnan(arr)]
        if len(arr_clean) == 0:
            return np.nan
        k = max(1, int(np.ceil(alpha * len(arr_clean))))
        return float(np.mean(np.sort(arr_clean)[:k]))

    # Worst-episode violation duration: max violation_rate across episodes
    vfa_worst  = float(np.nanmax(vfa_rates)) if len(vfa_rates) > 0 else np.nan
    score_worst = float(np.nanmin(scores)) if len(scores) > 0 else np.nan

    return {
        'reward_mean':        float(np.nanmean(rewards)),
        'reward_std':         float(np.nanstd(rewards, ddof=1)),
        'ch4_avg':            float(np.nanmean(extract('production.avg_ch4_flow'))),
        'ch4_std':            float(np.nanstd(extract('production.avg_ch4_flow'), ddof=1)),
        'violation_rate':     float(np.nanmean(vfa_rates)),
        'violation_rate_std': float(np.nanstd(vfa_rates, ddof=1)),
        'vfa_cvar95':         _cvar(vfa_rates, alpha=0.05),   # worst-5% violation rate
        'vfa_worst_episode':  vfa_worst,
        'overall_score':      float(np.nanmean(scores)),
        'score_worst':        score_worst,
        'score_cvar95':       _cvar(-scores, alpha=0.05),     # worst-5% score (negated → worst)
        'terminated_rate':    float(np.nanmean(terminated)),
        'ph_mean':            float(np.nanmean(extract('safety.ph_mean'))),
        'vfa_max_mean':       float(np.nanmean(extract('safety.vfa_max'))),
        'n_episodes':         len(episode_metrics),
    }


def _print_eval_summary(df: pd.DataFrame):
    """Print a readable evaluation summary table."""
    print(f"\n{'='*80}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*80}")
    group_cols = ['obs_mode', 'train_scenario', 'test_scenario'] if 'obs_mode' in df.columns \
        else ['train_scenario', 'test_scenario']
    summary = df.groupby(group_cols)[['overall_score', 'violation_rate', 'ch4_avg']].mean()
    print(summary.to_string(float_format='{:.3f}'.format))


def _print_baseline_summary(df: pd.DataFrame):
    """Print a readable baseline summary table."""
    print(f"\n{'='*80}")
    print("  BASELINE SUMMARY")
    print(f"{'='*80}")
    summary = df.groupby(['controller', 'scenario'])[
        ['overall_score', 'violation_rate', 'ch4_avg']
    ].mean()
    print(summary.to_string(float_format='{:.3f}'.format))


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='ADM1 RL Experiment Runner for Direction A paper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--config', type=str, default=DEFAULT_CONFIG_PATH,
        help=f'Path to YAML config file (default: {DEFAULT_CONFIG_PATH})'
    )
    parser.add_argument(
        '--mode', type=str,
        choices=['train', 'eval', 'baseline', 'train_and_eval', 'full'],
        help='Override experiment mode from config'
    )
    parser.add_argument(
        '--seeds', type=int, nargs='+',
        help='Override seeds (e.g. --seeds 42 123 456)'
    )
    parser.add_argument(
        '--steps', type=int,
        help='Override total_timesteps'
    )
    parser.add_argument(
        '--output-dir', type=str,
        help='Override output directory'
    )
    parser.add_argument(
        '--reward-config', type=str,
        help='Override reward config name'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print config and exit without running'
    )
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    # ── Load & merge config ──────────────────────────────────────────────────
    print(f"Loading config: {args.config}")
    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, args)

    exp_cfg    = cfg['experiment']
    mode       = exp_cfg['mode']
    output_dir = Path(exp_cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config for reproducibility
    with open(output_dir / 'resolved_config.yaml', 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    if args.dry_run:
        print("\n[DRY RUN] Resolved config:")
        print(yaml.dump(cfg, default_flow_style=False))
        return

    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT: {exp_cfg['name']}")
    print(f"  Mode:       {mode}")
    print(f"  Output:     {output_dir}")
    print(f"{'#'*70}\n")

    t_start = time.time()

    eval_df     = None
    baseline_df = None
    log_dirs    = None

    # ── Run phases ───────────────────────────────────────────────────────────
    if mode in ('train', 'train_and_eval', 'full'):
        log_dirs = run_training(cfg, output_dir)

    if mode in ('eval', 'train_and_eval', 'full'):
        eval_df = run_evaluation(cfg, output_dir, log_dirs)

    if mode in ('baseline', 'full') or cfg.get('baseline', {}).get('enabled', False):
        if mode in ('baseline', 'full'):
            baseline_df = run_baseline(cfg, output_dir)

    # ── Figures & tables ─────────────────────────────────────────────────────
    ocfg = cfg.get('output', {})

    if mode == 'full' or ocfg.get('generate_plots', False):
        generate_figures(cfg, output_dir, eval_df, baseline_df)

    if mode == 'full' or ocfg.get('latex_tables', False):
        generate_latex_tables(cfg, output_dir, eval_df, baseline_df)

    # ── Done ─────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"  Total time: {elapsed/60:.1f} min")
    print(f"  Results in: {output_dir}")
    print(f"{'#'*70}")


if __name__ == '__main__':
    main()

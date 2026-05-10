"""
Per-Step Reward Time Series — All Controllers × Non-thermal Scenarios
=====================================================================

Re-runs one episode per (controller, scenario) pair and plots the
cumulative reward and per-step reward over simulation time.

Controllers: Constant, PID, CascadedPID, SAC-Full, SAC-Compact
Scenarios:   nominal, high_load, low_load, shock_load
             (MPC/NMPC excluded — too slow for interactive use)

Subplots: 2 rows (scenarios pairs) × 2 cols, 4 panels total.
Each panel: per-step reward smoothed with a rolling window.

Usage:
    python analysis/plot_reward_timeseries.py

    python analysis/plot_reward_timeseries.py \\
        --training-dir /path/to/sac_single_scenario/training \\
        --output-dir   /path/to/figures \\
        --seed 42 --smooth 50
"""

import argparse
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Constants ─────────────────────────────────────────────────────────────────

SCENARIOS = ['nominal', 'high_load', 'low_load', 'shock_load']
SCENARIO_LABELS = ['Nominal', 'High Load', 'Low Load', 'Shock Load']

REWARD_CONFIG = 'safety_first'
SEEDS_SAC     = [42, 123, 456]   # average over seeds for SAC

# Controller styles: (key, label, color, linestyle, lw)
CTRL_STYLES = [
    ('constant',     'Constant',    '#999999', '-',   1.0),
    ('pid',          'PID',         '#BBBBBB', '--',  1.0),
    ('cascaded_pid', 'Cas.-PID',    '#CCCCCC', ':',   1.0),
    ('sac_full',     'SAC-Full',    '#003F88', '-',   1.8),
    ('sac_simple',   'SAC-Compact', '#C00000', '--',  1.8),
]

DEFAULT_TRAINING_DIR = (
    _ROOT / 'results' / 'sac_single_scenario' / 'training'
)
DEFAULT_OUTPUT_DIR = _ROOT / 'results' / 'figures'


# ── Episode runner ────────────────────────────────────────────────────────────

def run_baseline_episode(ctrl_key: str, scenario: str, seed: int = 42):
    """Run one episode with a baseline controller; return per-step rewards."""
    from env.adm1_gym_env import ADM1Env_v2
    from baselines.baseline_controllers import get_controller
    from training.reward_configs import REWARD_CONFIGS

    env = ADM1Env_v2(
        scenario_name=scenario,
        reward_config=REWARD_CONFIGS[REWARD_CONFIG],
        obs_mode='full',
    )
    obs, _ = env.reset(seed=seed)
    ctrl = get_controller(ctrl_key)
    ctrl.reset()

    rewards = []
    done = False
    while not done:
        action = ctrl.get_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
    env.close()
    return np.array(rewards)


def run_sac_episode(model_path: pathlib.Path, scenario: str,
                    obs_mode: str, seed: int = 42):
    """Run one episode with a SAC policy; return per-step rewards."""
    from stable_baselines3 import SAC
    from env.adm1_gym_env import ADM1Env_v2
    from training.reward_configs import REWARD_CONFIGS

    model = SAC.load(str(model_path))
    env = ADM1Env_v2(
        scenario_name=scenario,
        reward_config=REWARD_CONFIGS[REWARD_CONFIG],
        obs_mode=obs_mode,
    )
    obs, _ = env.reset(seed=seed)

    rewards = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
    env.close()
    return np.array(rewards)


def _find_best_model(training_dir: pathlib.Path,
                     scenario: str, obs_mode: str, seed: int) -> pathlib.Path | None:
    suffix   = '_simple' if obs_mode == 'simple' else ''
    run_name = f'sac_{scenario}_{REWARD_CONFIG}_seed{seed}{suffix}'
    run_dir  = training_dir / run_name
    best     = run_dir / 'best_model' / 'best_model.zip'
    final    = run_dir / 'final_model.zip'
    if best.exists():  return best
    if final.exists(): return final
    return None


def collect_all(training_dir: pathlib.Path, seed: int = 42):
    """
    Returns {scenario: {ctrl_key: rewards_array}}.
    SAC models: average over SEEDS_SAC seeds.
    """
    results = {sc: {} for sc in SCENARIOS}

    for scenario in SCENARIOS:
        print(f'  {scenario}:')

        # Baselines
        for ctrl_key, label, *_ in CTRL_STYLES:
            if ctrl_key.startswith('sac'):
                continue
            print(f'    {label} ...', end=' ', flush=True)
            try:
                r = run_baseline_episode(ctrl_key, scenario, seed=seed)
                results[scenario][ctrl_key] = r
                print(f'OK ({len(r)} steps)')
            except Exception as e:
                print(f'FAIL: {e}')

        # SAC (average across seeds)
        for obs_mode, ctrl_key in [('full', 'sac_full'), ('simple', 'sac_simple')]:
            label = 'SAC-Full' if obs_mode == 'full' else 'SAC-Compact'
            print(f'    {label} ...', end=' ', flush=True)
            seed_rewards = []
            for s in SEEDS_SAC:
                mp = _find_best_model(training_dir, scenario, obs_mode, s)
                if mp is None:
                    continue
                try:
                    r = run_sac_episode(mp, scenario, obs_mode, seed=seed)
                    seed_rewards.append(r)
                except Exception as e:
                    print(f'seed{s} FAIL: {e}', end=' ')
            if seed_rewards:
                min_len = min(len(r) for r in seed_rewards)
                mean_r  = np.mean([r[:min_len] for r in seed_rewards], axis=0)
                results[scenario][ctrl_key] = mean_r
                print(f'OK ({len(mean_r)} steps, {len(seed_rewards)} seeds)')
            else:
                print('no models found')

    return results


# ── Smoothing ─────────────────────────────────────────────────────────────────

def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')


# ── Figure ────────────────────────────────────────────────────────────────────

def build_figure(
    results: dict,
    output_dir: pathlib.Path,
    smooth: int = 50,
    dpi: int    = 300,
) -> None:

    plt.rcParams.update({
        'font.family':      'sans-serif',
        'font.sans-serif':  ['Arial', 'DejaVu Sans'],
        'font.size':        7,
        'axes.linewidth':   0.7,
        'xtick.direction':  'in',
        'ytick.direction':  'in',
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
    })

    n_sc  = len(SCENARIOS)
    n_row, n_col = 2, 2
    fig, axes = plt.subplots(n_row, n_col, figsize=(6.5, 4.2), sharex=False)

    handles_legend = []

    for idx, (scenario, sc_label) in enumerate(zip(SCENARIOS, SCENARIO_LABELS)):
        row, col = divmod(idx, n_col)
        ax = axes[row, col]

        ax.axhline(0, color='#DDDDDD', lw=0.7, ls='--', zorder=0)
        sc_data = results.get(scenario, {})

        for ctrl_key, label, color, ls, lw in CTRL_STYLES:
            r = sc_data.get(ctrl_key)
            if r is None:
                continue
            steps_d = np.arange(len(r)) / 24.0   # steps → days (1 step = 1 hour)
            r_smooth = _smooth(r, smooth)
            line, = ax.plot(steps_d, r_smooth,
                            color=color, lw=lw, ls=ls,
                            label=label, zorder=3 if 'sac' in ctrl_key else 2)
            if idx == 0:
                handles_legend.append(line)

        ax.set_title(sc_label, fontsize=7.5, pad=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_major_locator(mticker.AutoLocator())

        if row == n_row - 1:
            ax.set_xlabel('Simulation time (days)', fontsize=7)
        if col == 0:
            ax.set_ylabel('Per-step reward (smoothed)', fontsize=7)

    # Legend in last subplot
    if handles_legend:
        axes[-1, -1].legend(
            handles=handles_legend,
            labels=[s[1] for s in CTRL_STYLES],
            loc='lower right',
            ncol=1,
            fontsize=6.0,
            framealpha=0.93,
            edgecolor='#CCCCCC',
            handlelength=2.0,
            handletextpad=0.3,
            borderpad=0.5,
        )

    fig.tight_layout(pad=0.6, h_pad=0.8, w_pad=0.8)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = 'fig_reward_timeseries'
    fig.savefig(output_dir / f'{stem}.pdf', bbox_inches='tight', dpi=dpi)
    fig.savefig(output_dir / f'{stem}.png', bbox_inches='tight', dpi=200)
    print(f'\nSaved: {output_dir}/{stem}.pdf / .png')
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Per-step reward time series for all controllers.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--training-dir', type=pathlib.Path,
                        default=DEFAULT_TRAINING_DIR)
    parser.add_argument('--output-dir',   type=pathlib.Path,
                        default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--seed',   type=int, default=42,
                        help='Seed for baseline episodes')
    parser.add_argument('--smooth', type=int, default=50,
                        help='Rolling-window size for reward smoothing (steps)')
    parser.add_argument('--dpi',    type=int, default=300)
    args = parser.parse_args()

    print('Running episodes ...')
    results = collect_all(args.training_dir, seed=args.seed)

    print('\nBuilding figure ...')
    build_figure(results, args.output_dir, smooth=args.smooth, dpi=args.dpi)


if __name__ == '__main__':
    main()

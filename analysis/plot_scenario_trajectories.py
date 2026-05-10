"""
Scenario Trajectory Plot
========================

Runs one episode per scenario using the SAC-full seed42 policy and plots
key state/action variables over time. Each line = one scenario.

Subplots (rows):
  1. Effective feed rate  q_ad × feed_mult  [m³/d]
  2. Heating power        Q_HEX             [W]
  3. Reactor temperature  T_L               [°C]
  4. pH
  5. Total VFA            [g COD / L]
  6. Methane flow         q_CH4             [m³ STP/d]

Usage:
    python analysis/plot_scenario_trajectories.py \\
        --training-dir /path/to/sac_single_scenario/training \\
        --output-dir   /path/to/figures \\
        [--seed 42]
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
sys.path.insert(0, str(_ROOT))

from env.adm1_gym_env import ADM1Env_v2
from training.reward_configs import REWARD_CONFIGS
from stable_baselines3 import SAC


# ── Config ────────────────────────────────────────────────────────────────────

SCENARIOS = ['nominal', 'high_load', 'shock_load', 'low_load',
             'temperature_drop', 'cold_winter']
SCENARIO_LABELS = ['Nominal', 'High Load', 'Shock Load', 'Low Load',
                   'Temp. Drop', 'Cold Winter']

COLORS = ['#2166AC', '#D6604D', '#F4A582', '#4DAC26', '#762A83', '#E66101']

REWARD_KEY = 'safety_first'
NUM_STEPS  = 2880
EVAL_SEED  = 42

# T_L obs index and normalization
T_L_OBS_IDX = 12
T_L_REF     = 308.15   # K

DEFAULT_TRAINING_DIR = (
    _ROOT / 'results' / 'sac_single_scenario' / 'training'
)
DEFAULT_OUTPUT_DIR = _ROOT / 'results' / 'figures'


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(model, scenario: str, seed: int) -> dict:
    env = ADM1Env_v2(
        scenario_name=scenario,
        reward_config=REWARD_CONFIGS[REWARD_KEY],
        obs_mode='full',
    )
    obs, _ = env.reset(seed=seed)

    t_days, q_feed, q_hex, T_L, pH, vfa, q_ch4, feed_mult_arr = \
        [], [], [], [], [], [], [], []

    for _ in range(NUM_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)

        t_days.append(float(info['time_days']))
        q_feed.append(float(info['q_ad']))
        feed_mult_arr.append(float(info['feed_multiplier']))
        q_hex.append(float(action[2]))
        # T_L from obs[12]: T_L_norm = (T_L - 308.15) / 10
        T_L.append(float(obs[T_L_OBS_IDX]) * 10.0 + T_L_REF - 273.15)  # → °C
        pH.append(float(info['pH']))
        vfa.append(float(info['total_vfa']))
        q_ch4.append(float(info['q_ch4']))

        if terminated or truncated:
            break

    env.close()

    t = np.array(t_days)
    return {
        't':            t,
        'q_eff':        np.array(q_feed) * np.array(feed_mult_arr),
        'q_hex':        np.array(q_hex),
        'T_L':          np.array(T_L),
        'pH':           np.array(pH),
        'vfa':          np.array(vfa),
        'q_ch4':        np.array(q_ch4),
        'terminated':   terminated,
    }


# ── Figure ────────────────────────────────────────────────────────────────────

PANELS = [
    ('q_eff',  'Feed rate\n[m³/d]',          None,        None),
    ('q_hex',  'Q_HEX\n[W]',                 None,        None),
    ('T_L',    'Reactor temp.\n[°C]',         None,        None),
    ('pH',     'pH',                          (6.5, 8.0),  [(6.8, 7.8, '#FFE0B2')]),
    ('vfa',    'Total VFA\n[g COD/L]',        (0.0, 0.35), [(None, 0.2, '#FFE0B2')]),
    ('q_ch4',  'CH₄ flow\n[m³ STP/d]',       (0, None),   None),
]


def build_figure(trajectories: dict, output_dir: pathlib.Path) -> None:
    plt.rcParams.update({
        'font.family':      'sans-serif',
        'font.sans-serif':  ['Arial', 'DejaVu Sans'],
        'font.size':        7.5,
        'axes.linewidth':   0.7,
        'xtick.direction':  'in',
        'ytick.direction':  'in',
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
    })

    n_panels = len(PANELS)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(8.0, 1.55 * n_panels),
        sharex=True,
    )

    for ax, (key, ylabel, ylim, bands) in zip(axes, PANELS):
        # Safety bands
        if bands:
            for lo, hi, fc in bands:
                lo_ = lo if lo is not None else ax.get_ylim()[0]
                hi_ = hi if hi is not None else ax.get_ylim()[1]
                ax.axhspan(lo_, hi_, color=fc, alpha=0.35, zorder=0, linewidth=0)

        for sc, label, color in zip(SCENARIOS, SCENARIO_LABELS, COLORS):
            tr = trajectories[sc]
            t  = tr['t']
            y  = tr[key]
            lw = 1.4
            ax.plot(t, y, color=color, lw=lw, label=label, alpha=0.88)

            # Mark termination
            if tr['terminated'] and len(t) < NUM_STEPS:
                ax.axvline(t[-1], color=color, lw=0.8, ls=':', alpha=0.5)

        ax.set_ylabel(ylabel, fontsize=7, labelpad=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if ylim:
            lo, hi = ylim
            cur_lo, cur_hi = ax.get_ylim()
            ax.set_ylim(lo if lo is not None else cur_lo,
                        hi if hi is not None else cur_hi)

        # Soft constraint reference lines
        if key == 'pH':
            ax.axhline(6.8, color='#E65100', lw=0.7, ls='--', alpha=0.6)
            ax.axhline(7.8, color='#E65100', lw=0.7, ls='--', alpha=0.6)
        if key == 'vfa':
            ax.axhline(0.2, color='#E65100', lw=0.7, ls='--', alpha=0.6)

    # Legend on first panel
    axes[0].legend(
        ncol=3, fontsize=6.5,
        loc='upper right',
        framealpha=0.92, edgecolor='#CCCCCC',
        handlelength=1.8, handletextpad=0.4,
        columnspacing=0.8, borderpad=0.5,
    )

    axes[-1].set_xlabel('Time [days]', fontsize=8)
    axes[-1].set_xlim(0, 30)
    axes[-1].xaxis.set_major_locator(mticker.MultipleLocator(5))

    fig.tight_layout(pad=0.5, h_pad=0.3)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = 'fig_scenario_trajectories'
    fig.savefig(output_dir / f'{stem}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(output_dir / f'{stem}.png', bbox_inches='tight', dpi=200)
    print(f'Saved: {output_dir}/{stem}.pdf / .png')
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Scenario trajectory plot (SAC-Full seed42 policy).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--training-dir', type=pathlib.Path,
                        default=DEFAULT_TRAINING_DIR)
    parser.add_argument('--output-dir',   type=pathlib.Path,
                        default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    trajectories = {}
    for sc in SCENARIOS:
        obs_suffix = ''
        run_name   = f'sac_{sc}_{REWARD_KEY}_seed{args.seed}{obs_suffix}'
        model_path = args.training_dir / run_name / 'best_model' / 'best_model.zip'
        if not model_path.exists():
            print(f'  [SKIP] model not found: {model_path}')
            continue
        print(f'  Running {sc} (seed{args.seed}) ...', end='', flush=True)
        model = SAC.load(str(model_path))
        tr = run_episode(model, sc, seed=EVAL_SEED)
        trajectories[sc] = tr
        n = len(tr['t'])
        print(f' {n} steps  score_proxy={tr["q_ch4"].mean():.0f} m³/d CH4')

    build_figure(trajectories, args.output_dir)


if __name__ == '__main__':
    main()

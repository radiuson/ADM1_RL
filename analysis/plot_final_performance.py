"""
Final Training Performance Bar Chart (Stage 1)
===============================================

Grouped bar chart showing the final eval reward (last checkpoint of the best
eval run) for each scenario × obs-mode pair, averaged over seeds ± 1 std.

Layout: single horizontal bar group per scenario (6 groups × 2 bars).
Colour: deep blue = full obs, deep red = simple obs.

Data source: TensorBoard eval/mean_reward from the last SAC_* event file in
    <training-dir>/sac_<scenario>_safety_first_seed<N>[_simple]/tensorboard/

Usage:
    python analysis/plot_final_performance.py

    python analysis/plot_final_performance.py \\
        --training-dir /path/to/results/sac_single_scenario/training \\
        --output-dir   /path/to/figures \\
        --seeds 42 123 456
"""

import argparse
import pathlib
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches


# ── Constants ─────────────────────────────────────────────────────────────────

SCENARIOS = [
    'nominal', 'high_load', 'low_load',
    'shock_load', 'temperature_drop', 'cold_winter',
]
SCENARIO_LABELS = [
    'Nominal', 'High\nLoad', 'Low\nLoad',
    'Shock\nLoad', 'Temp.\nDrop', 'Cold\nWinter',
]

DEFAULT_SEEDS   = [42, 123, 456]
REWARD_CONFIG   = 'safety_first'

COLOR_FULL      = '#003F88'
COLOR_SIMPLE    = '#C00000'
ALPHA_BAR       = 0.82

DEFAULT_TRAINING_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / 'results' / 'sac_single_scenario' / 'training'
)
DEFAULT_OUTPUT_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / 'results' / 'figures'
)


# ── Data loading ──────────────────────────────────────────────────────────────

def _last_eval_reward(run_dir: pathlib.Path) -> float | None:
    """Return the *final* eval/mean_reward value from the last SAC_* folder."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        raise ImportError("pip install tensorboard")

    tb_root = run_dir / 'tensorboard'
    if not tb_root.exists():
        return None
    sac_dirs = sorted(tb_root.glob('SAC_*'))
    if not sac_dirs:
        return None

    ea = EventAccumulator(str(sac_dirs[-1]))
    ea.Reload()
    if 'eval/mean_reward' not in ea.Tags().get('scalars', []):
        return None

    events = ea.Scalars('eval/mean_reward')
    return events[-1].value


def _peak_eval_reward(run_dir: pathlib.Path) -> float | None:
    """Return the *best* eval/mean_reward across all steps in the last SAC_* folder."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        raise ImportError("pip install tensorboard")

    tb_root = run_dir / 'tensorboard'
    if not tb_root.exists():
        return None
    sac_dirs = sorted(tb_root.glob('SAC_*'))
    if not sac_dirs:
        return None

    ea = EventAccumulator(str(sac_dirs[-1]))
    ea.Reload()
    if 'eval/mean_reward' not in ea.Tags().get('scalars', []):
        return None

    return max(e.value for e in ea.Scalars('eval/mean_reward'))


def load_final_stats(
    training_dir: pathlib.Path,
    seeds: list[int],
    use_peak: bool = False,
) -> dict[str, dict[str, tuple[float, float]]]:
    """
    Returns {scenario: {obs_mode: (mean, std)}} for all scenarios.
    obs_mode in {'full', 'simple'}.
    use_peak: if True use best reward in curve instead of final reward.
    """
    loader = _peak_eval_reward if use_peak else _last_eval_reward
    result = {}
    for scenario in SCENARIOS:
        result[scenario] = {}
        for obs_mode in ('full', 'simple'):
            suffix = '_simple' if obs_mode == 'simple' else ''
            vals = []
            for seed in seeds:
                run_name = f'sac_{scenario}_{REWARD_CONFIG}_seed{seed}{suffix}'
                v = loader(training_dir / run_name)
                if v is not None:
                    vals.append(v)
                else:
                    warnings.warn(f'Missing: {run_name}', stacklevel=2)
            if vals:
                result[scenario][obs_mode] = (
                    float(np.mean(vals)),
                    float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
                )
            else:
                result[scenario][obs_mode] = (float('nan'), 0.0)
    return result


# ── Figure ────────────────────────────────────────────────────────────────────

def build_figure(
    training_dir: pathlib.Path,
    output_dir:   pathlib.Path,
    seeds:        list[int] = DEFAULT_SEEDS,
    use_peak:     bool      = False,
    dpi:          int       = 300,
) -> None:
    """Build and save the final-performance grouped bar chart."""

    plt.rcParams.update({
        'font.family':      'sans-serif',
        'font.sans-serif':  ['Arial', 'DejaVu Sans'],
        'font.size':        7,
        'axes.linewidth':   0.7,
        'xtick.direction':  'in',
        'ytick.direction':  'in',
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'xtick.labelsize':  6.5,
        'ytick.labelsize':  6.5,
    })

    print('Loading final-reward statistics ...')
    data = load_final_stats(training_dir, seeds, use_peak=use_peak)

    n_sc  = len(SCENARIOS)
    x     = np.arange(n_sc)
    BAR_W = 0.32
    offsets = [-BAR_W / 2, BAR_W / 2]

    fig, ax = plt.subplots(figsize=(5.5, 2.8))

    # Zero line
    ax.axhline(0, color='#BBBBBB', lw=0.8, ls='--', zorder=0)

    for oi, (obs_mode, color, label_obs) in enumerate([
        ('full',   COLOR_FULL,   'Full obs (13-dim)'),
        ('simple', COLOR_SIMPLE, 'Compact obs (5-dim)'),
    ]):
        means = np.array([data[sc][obs_mode][0] for sc in SCENARIOS])
        stds  = np.array([data[sc][obs_mode][1] for sc in SCENARIOS])

        bx = x + offsets[oi]
        ax.bar(
            bx, means,
            width=BAR_W,
            color=color, alpha=ALPHA_BAR,
            edgecolor=color, linewidth=0.6,
            label=label_obs,
            zorder=3,
        )
        # Error bars only where std > 0
        valid = (stds > 0) & ~np.isnan(means)
        if valid.any():
            ax.errorbar(
                bx[valid], means[valid],
                yerr=stds[valid],
                fmt='none', ecolor=color,
                elinewidth=1.0, capsize=2.5, capthick=0.8,
                alpha=0.9, zorder=5,
            )

        # Annotate value on top of bar (skip NaN)
        for xi, (m, s) in zip(bx, zip(means, stds)):
            if np.isnan(m):
                continue
            va   = 'bottom' if m >= 0 else 'top'
            ypos = m + (s + 30 if m >= 0 else -(s + 30))
            ax.text(xi, ypos, f'{m:.0f}',
                    ha='center', va=va, fontsize=4.8, color=color,
                    rotation=90 if abs(m) > 1500 else 0)

    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIO_LABELS, fontsize=7)
    ylabel = ('Peak eval reward' if use_peak else 'Final eval reward') + ' (mean ± 1 std)'
    ax.set_ylabel(ylabel, fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_locator(mticker.AutoLocator())

    ax.legend(
        loc='upper right',
        fontsize=6.5,
        framealpha=0.92,
        edgecolor='#CCCCCC',
        handlelength=1.6,
        handletextpad=0.4,
        borderpad=0.5,
    )

    # Annotate seed count
    ax.text(
        0.01, 0.99, f'n={len(seeds)} seeds per bar',
        transform=ax.transAxes,
        fontsize=5.5, va='top', ha='left', color='#666666',
    )

    fig.tight_layout(pad=0.6)

    output_dir.mkdir(parents=True, exist_ok=True)
    tag  = 'peak' if use_peak else 'final'
    stem = f'fig_performance_{tag}'
    fig.savefig(output_dir / f'{stem}.pdf', bbox_inches='tight', dpi=dpi)
    fig.savefig(output_dir / f'{stem}.png', bbox_inches='tight', dpi=200)
    print(f'Saved: {output_dir}/{stem}.pdf / .png')
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Final/peak training performance bar chart (Stage 1).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--training-dir', type=pathlib.Path,
                        default=DEFAULT_TRAINING_DIR)
    parser.add_argument('--output-dir',   type=pathlib.Path,
                        default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    parser.add_argument('--peak', action='store_true',
                        help='Use best (peak) reward instead of final reward')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    build_figure(
        training_dir = args.training_dir,
        output_dir   = args.output_dir,
        seeds        = args.seeds,
        use_peak     = args.peak,
        dpi          = args.dpi,
    )


if __name__ == '__main__':
    main()

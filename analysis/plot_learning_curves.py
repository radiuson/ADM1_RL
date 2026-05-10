"""
SAC Training Learning Curves (Stage 1)
=======================================

Plots eval/mean_reward vs training steps for all six ADM1 scenarios,
comparing full-obs (13-dim) and simple-obs (5-dim compact) observation modes.

Layout: 2 × 3 subplots (double-column width, ~7 in wide).
Each subplot shows:
  - Solid line   : full obs,   mean ± 1 std across seeds
  - Dashed line  : simple obs, mean ± 1 std across seeds
  - Shaded band  : ± 1 std (same color, lower alpha)

Data source: TensorBoard event files in
    <training-dir>/sac_<scenario>_safety_first_seed<N>[_simple]/tensorboard/SAC_*/

Usage:
    # Default paths (paper reproduction layout)
    python analysis/plot_learning_curves.py

    # Custom paths
    python analysis/plot_learning_curves.py \\
        --training-dir /path/to/results/single_scenario/training \\
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


# ── Constants ─────────────────────────────────────────────────────────────────

SCENARIOS = [
    'nominal', 'high_load', 'low_load',
    'shock_load', 'temperature_drop', 'cold_winter',
]
SCENARIO_LABELS = [
    'Nominal', 'High Load', 'Low Load',
    'Shock Load', 'Temp. Drop', 'Cold Winter',
]

DEFAULT_SEEDS        = []   # empty → auto-discover per scenario
REWARD_CONFIG        = 'safety_first'
N_STEPS_TARGET       = 300_000
EVAL_FREQ            = 10_000

# Colour palette (consistent with plot_combo.py colour language)
COLOR_FULL   = '#003F88'   # deep blue  — full obs
COLOR_SIMPLE = '#C00000'   # deep red   — simple obs

ALPHA_BAND   = 0.15
ALPHA_LINE   = 0.95

STEPS_K_TICKS = [0, 50, 100, 150, 200, 250, 300]   # ×1000

DEFAULT_TRAINING_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / 'results' / 'single_scenario' / 'training'
)
DEFAULT_OUTPUT_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / 'results' / 'figures'
)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_eval_curve(run_dir: pathlib.Path) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load eval/mean_reward from the *last* SAC_* TensorBoard folder in run_dir.

    Returns (steps_array, rewards_array) or None if no data found.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        raise ImportError(
            "tensorboard not found — install with: pip install tensorboard"
        )

    tb_root = run_dir / 'tensorboard'
    if not tb_root.exists():
        return None

    sac_dirs = sorted(tb_root.glob('SAC_*'))
    if not sac_dirs:
        return None

    # Use the last SAC_* folder (completed / most recent run)
    ea = EventAccumulator(str(sac_dirs[-1]))
    ea.Reload()

    if 'eval/mean_reward' not in ea.Tags().get('scalars', []):
        return None

    events  = ea.Scalars('eval/mean_reward')
    steps   = np.array([e.step   for e in events], dtype=float)
    rewards = np.array([e.value  for e in events], dtype=float)
    return steps, rewards


def discover_seeds(training_dir: pathlib.Path, scenario: str) -> list[int]:
    """Scan training_dir for all available seeds for this scenario (full obs)."""
    import re
    found = set()
    for d in training_dir.glob(f'sac_{scenario}_{REWARD_CONFIG}_seed*'):
        if 'simple' in d.name:
            continue
        m = re.search(r'seed(\d+)', d.name)
        if m:
            found.add(int(m.group(1)))
    return sorted(found)


def load_scenario_curves(
    training_dir: pathlib.Path,
    scenario: str,
    seeds: list[int],
    obs_mode: str,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Aggregate learning curves for one (scenario, obs_mode) pair over *seeds*.

    If seeds is empty, auto-discovers all available seeds for this scenario.
    Returns (steps, mean_rewards, std_rewards) aligned to the shortest
    available curve, or (None, None, None) if no seeds loaded.
    """
    if not seeds:
        seeds = discover_seeds(training_dir, scenario)

    suffix = '_simple' if obs_mode == 'simple' else ''
    curves = []

    for seed in seeds:
        run_name = f'sac_{scenario}_{REWARD_CONFIG}_seed{seed}{suffix}'
        run_dir  = training_dir / run_name
        result   = _load_eval_curve(run_dir)
        if result is None:
            continue
        curves.append(result)

    if not curves:
        return None, None, None

    min_len  = min(len(c[1]) for c in curves)
    steps    = curves[0][0][:min_len]
    val_mat  = np.stack([c[1][:min_len] for c in curves])   # (n_seeds, T)

    mean = np.mean(val_mat, axis=0)
    std  = np.std(val_mat,  axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
    return steps, mean, std


# ── Figure ────────────────────────────────────────────────────────────────────

def build_figure(
    training_dir: pathlib.Path,
    output_dir:   pathlib.Path,
    seeds:        list[int] = DEFAULT_SEEDS,
    clip_bottom:  float     = -4000.0,   # clip Y-axis floor to suppress early crashes
    dpi:          int       = 300,
) -> None:
    """Build and save the 2×3 learning-curve figure."""

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
        'legend.fontsize':  6.5,
    })

    n_rows, n_cols = 3, 2
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.0, 5.5),
        sharex=True,
    )

    handles_legend = []   # collect once from first subplot

    for idx, (scenario, label) in enumerate(zip(SCENARIOS, SCENARIO_LABELS)):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        ax.axhline(0, color='#CCCCCC', lw=0.7, ls='--', zorder=0)

        # Auto-discover actual seeds for this scenario
        actual_seeds = discover_seeds(training_dir, scenario) if not seeds else seeds

        any_data = False
        for obs_mode, color, ls, label_obs in [
            ('full',   COLOR_FULL,   '-',  'Full obs (13-dim)'),
            ('simple', COLOR_SIMPLE, '--', 'Compact obs (5-dim)'),
        ]:
            steps, mean, std = load_scenario_curves(
                training_dir, scenario, actual_seeds, obs_mode
            )
            if steps is None:
                continue
            any_data = True

            steps_k = steps / 1_000   # display in thousands

            ax.fill_between(
                steps_k, mean - std, mean + std,
                color=color, alpha=ALPHA_BAND, linewidth=0,
            )
            line, = ax.plot(
                steps_k, mean,
                color=color, lw=1.5, ls=ls, alpha=ALPHA_LINE,
                label=label_obs,
            )
            if idx == 0:
                handles_legend.append(line)

        # Subplot title and annotations
        ax.set_title(label, fontsize=7.5, pad=3)

        # Seed count annotation (top-left corner) — use actual discovered count
        n_seeds_shown = len(actual_seeds)
        ax.text(
            0.04, 0.96, f'n={n_seeds_shown} seeds',
            transform=ax.transAxes,
            fontsize=5.5, va='top', ha='left', color='#666666',
        )

        ax.set_xlim(0, N_STEPS_TARGET / 1_000)
        ax.xaxis.set_major_locator(mticker.FixedLocator(STEPS_K_TICKS))

        # Y-axis: clip floor to suppress very negative early-episode rewards
        ymin_cur, _ = ax.get_ylim()
        ax.set_ylim(bottom=max(ymin_cur, clip_bottom))
        ax.yaxis.set_major_locator(mticker.AutoLocator())
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if not any_data:
            ax.text(0.5, 0.5, 'No data',
                    transform=ax.transAxes,
                    ha='center', va='center', fontsize=7, color='#999999')

    # Shared axis labels
    for ax in axes[-1, :]:
        ax.set_xlabel('Training steps (×10³)', fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel('Eval reward (mean)', fontsize=7)

    # Legend — anchored inside the last subplot (Cold Winter), lower-left corner
    if handles_legend:
        axes[-1, -1].legend(
            handles=handles_legend,
            labels=['Full obs (13-dim)', 'Compact obs (5-dim)'],
            loc='lower left',
            ncol=1,
            framealpha=0.92,
            edgecolor='#CCCCCC',
            handlelength=2.5,
            handletextpad=0.4,
            borderpad=0.5,
            fontsize=6.5,
        )

    fig.tight_layout(pad=0.6, h_pad=0.8, w_pad=0.6)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = 'fig_learning_curves'
    fig.savefig(output_dir / f'{stem}.pdf', bbox_inches='tight', dpi=dpi)
    fig.savefig(output_dir / f'{stem}.png', bbox_inches='tight', dpi=200)
    print(f'Saved: {output_dir}/{stem}.pdf / .png')
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Plot SAC training learning curves (Stage 1).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--training-dir', type=pathlib.Path,
        default=DEFAULT_TRAINING_DIR,
        help='Root directory of Stage 1 training runs '
             '(contains sac_<scenario>_... subdirs).',
    )
    parser.add_argument(
        '--output-dir', type=pathlib.Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Directory to write figures.',
    )
    parser.add_argument(
        '--seeds', type=int, nargs='*',
        default=DEFAULT_SEEDS,
        help='Seeds to aggregate over (omit or pass empty to auto-discover per scenario).',
    )
    parser.add_argument(
        '--clip-bottom', type=float, default=-4000.0,
        help='Y-axis floor to suppress very negative early-episode rewards.',
    )
    parser.add_argument(
        '--dpi', type=int, default=300,
        help='PDF/PNG resolution for raster output.',
    )
    args = parser.parse_args()

    print('Loading TensorBoard eval curves ...')
    print(f'  training-dir : {args.training_dir}')
    print(f'  seeds        : {args.seeds}')

    build_figure(
        training_dir = args.training_dir,
        output_dir   = args.output_dir,
        seeds        = args.seeds,
        clip_bottom  = args.clip_bottom,
        dpi          = args.dpi,
    )


if __name__ == '__main__':
    main()

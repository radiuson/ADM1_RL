"""
Entropy Coefficient Trajectory (Stage 1)
=========================================

Two-panel figure illustrating auto-entropy tuning behaviour:

  Left  — Stable annealing: ent_coef vs training steps for a well-converged
           run (nominal / seed 42).  Shows the explore → exploit decay,
           with horizontal dashed lines marking the clamp bounds [0.01, 5.0].

  Right — Divergent training (without clamp): ent_coef for the high_load
           scenario during the unclamped attempt (SAC_2 folder), plotted on a
           log Y-scale to show the explosive growth from ~1.6 to ~5×10²⁸.
           A red shaded region highlights the divergence zone.

Data source: TensorBoard train/ent_coef from
    <training-dir>/sac_<scenario>_safety_first_seed<N>/tensorboard/SAC_*/

Usage:
    python analysis/plot_entropy_coef.py

    python analysis/plot_entropy_coef.py \\
        --training-dir /path/to/results/single_scenario/training \\
        --output-dir   /path/to/figures
"""

import argparse
import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Constants ─────────────────────────────────────────────────────────────────

ENT_COEF_MIN = 0.01
ENT_COEF_MAX = 5.0

COLOR_STABLE    = '#003F88'   # deep blue
COLOR_DIVERGE   = '#C00000'   # deep red
COLOR_CLAMP     = '#888888'   # grey dashed

DEFAULT_TRAINING_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / 'results' / 'single_scenario' / 'training'
)
DEFAULT_OUTPUT_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / 'results' / 'figures'
)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_ent_coef(
    run_dir: pathlib.Path,
    sac_subfolder: str | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load train/ent_coef from TensorBoard.

    Args:
        run_dir:       Path to the training run directory.
        sac_subfolder: e.g. 'SAC_2' to pick a specific subfolder.
                       If None, picks the last SAC_* folder.
    Returns:
        (steps, ent_coef_values) or None.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        raise ImportError("pip install tensorboard")

    tb_root = run_dir / 'tensorboard'
    if not tb_root.exists():
        return None

    if sac_subfolder:
        target = tb_root / sac_subfolder
        if not target.exists():
            return None
        sac_dir = target
    else:
        sac_dirs = sorted(tb_root.glob('SAC_*'))
        if not sac_dirs:
            return None
        sac_dir = sac_dirs[-1]

    ea = EventAccumulator(str(sac_dir))
    ea.Reload()
    if 'train/ent_coef' not in ea.Tags().get('scalars', []):
        return None

    events  = ea.Scalars('train/ent_coef')
    steps   = np.array([e.step  for e in events], dtype=float)
    values  = np.array([e.value for e in events], dtype=float)
    return steps, values


# ── Figure ────────────────────────────────────────────────────────────────────

def build_figure(
    training_dir: pathlib.Path,
    output_dir:   pathlib.Path,
    stable_scenario: str = 'nominal',
    stable_seed:     int = 42,
    diverge_scenario: str = 'high_load',
    diverge_seed:     int = 42,
    diverge_sac_sub:  str = 'SAC_2',   # unclamped training attempt
    dpi: int = 300,
) -> None:
    """Build and save the two-panel entropy-coefficient figure."""

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

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.4))

    # ── Left: stable annealing ─────────────────────────────────────────────
    ax = axes[0]
    run_dir = training_dir / f'sac_{stable_scenario}_safety_first_seed{stable_seed}'
    result  = _load_ent_coef(run_dir)

    if result is not None:
        steps, vals = result
        steps_k = steps / 1_000
        ax.plot(steps_k, vals, color=COLOR_STABLE, lw=1.5, zorder=3,
                label='Entropy coef. $\\alpha$')

        # Clamp bounds
        ax.axhline(ENT_COEF_MIN, color=COLOR_CLAMP, lw=0.9, ls='--', zorder=2,
                   label=f'Clamp min = {ENT_COEF_MIN}')
        ax.axhline(ENT_COEF_MAX, color='#E07B39',   lw=0.9, ls='--', zorder=2,
                   label=f'Clamp max = {ENT_COEF_MAX}')

        # Annotate final value
        ax.annotate(
            f'  final: {vals[-1]:.3f}',
            xy=(steps_k[-1], vals[-1]),
            fontsize=5.5, color=COLOR_STABLE, va='bottom',
        )
    else:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', color='#999999')

    ax.set_xlabel('Training steps (×10³)', fontsize=7)
    ax.set_ylabel('Entropy coefficient $\\alpha$', fontsize=7)
    ax.set_title(
        f'(a) Stable annealing — {stable_scenario.replace("_", " ").title()} '
        f'(seed {stable_seed})',
        fontsize=7, pad=4,
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', fontsize=5.5, framealpha=0.9,
              edgecolor='#CCCCCC', handlelength=2.0)

    # ── Right: divergent training ──────────────────────────────────────────
    ax2 = axes[1]
    run_dir2 = training_dir / f'sac_{diverge_scenario}_safety_first_seed{diverge_seed}'
    result2  = _load_ent_coef(run_dir2, sac_subfolder=diverge_sac_sub)

    if result2 is not None:
        steps2, vals2 = result2
        steps_k2 = steps2 / 1_000

        # Log-scale — clamp tiny/zero values
        vals2_log = np.clip(vals2, 1e-3, None)
        ax2.semilogy(steps_k2, vals2_log,
                     color=COLOR_DIVERGE, lw=1.5, zorder=3,
                     label='Entropy coef. $\\alpha$ (no clamp)')

        # Shaded "danger zone" above ENT_COEF_MAX
        ymax_display = vals2_log.max() * 5
        ax2.axhspan(ENT_COEF_MAX, ymax_display,
                    color=COLOR_DIVERGE, alpha=0.08, zorder=0)
        ax2.axhline(ENT_COEF_MAX, color='#E07B39', lw=0.9, ls='--', zorder=2,
                    label=f'Clamp max = {ENT_COEF_MAX}')

        # Annotate explosion start
        cross_idx = np.argmax(vals2 > ENT_COEF_MAX)
        if vals2[cross_idx] > ENT_COEF_MAX:
            ax2.annotate(
                f'diverges at\n{steps_k2[cross_idx]:.0f}k steps',
                xy=(steps_k2[cross_idx], vals2_log[cross_idx]),
                xytext=(steps_k2[cross_idx] + 20, vals2_log[cross_idx] * 10),
                fontsize=5.5, color=COLOR_DIVERGE,
                arrowprops=dict(arrowstyle='->', color=COLOR_DIVERGE, lw=0.7),
            )
    else:
        ax2.text(0.5, 0.5, f'No data\n(looking for {diverge_sac_sub})',
                 transform=ax2.transAxes,
                 ha='center', va='center', color='#999999', fontsize=6)

    ax2.set_xlabel('Training steps (×10³)', fontsize=7)
    ax2.set_ylabel('Entropy coefficient $\\alpha$  (log scale)', fontsize=7)
    ax2.set_title(
        f'(b) Divergent training — {diverge_scenario.replace("_", " ").title()} '
        f'/ {diverge_sac_sub} (no clamp)',
        fontsize=7, pad=4,
    )
    ax2.set_xlim(left=0)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(loc='upper left', fontsize=5.5, framealpha=0.9,
               edgecolor='#CCCCCC', handlelength=2.0)

    fig.tight_layout(pad=0.6, w_pad=1.2)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = 'fig_entropy_coef'
    fig.savefig(output_dir / f'{stem}.pdf', bbox_inches='tight', dpi=dpi)
    fig.savefig(output_dir / f'{stem}.png', bbox_inches='tight', dpi=200)
    print(f'Saved: {output_dir}/{stem}.pdf / .png')
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Entropy-coefficient trajectory figure (Stage 1).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--training-dir', type=pathlib.Path,
                        default=DEFAULT_TRAINING_DIR)
    parser.add_argument('--output-dir',   type=pathlib.Path,
                        default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--stable-scenario',  default='nominal')
    parser.add_argument('--stable-seed',      type=int, default=42)
    parser.add_argument('--diverge-scenario', default='high_load')
    parser.add_argument('--diverge-seed',     type=int, default=42)
    parser.add_argument('--diverge-sac-sub',  default='SAC_2',
                        help='SAC_N subfolder containing the unclamped run')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    build_figure(
        training_dir     = args.training_dir,
        output_dir       = args.output_dir,
        stable_scenario  = args.stable_scenario,
        stable_seed      = args.stable_seed,
        diverge_scenario = args.diverge_scenario,
        diverge_seed     = args.diverge_seed,
        diverge_sac_sub  = args.diverge_sac_sub,
        dpi              = args.dpi,
    )


if __name__ == '__main__':
    main()

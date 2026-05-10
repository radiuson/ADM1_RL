"""
Cross-scenario generalisation figure (v4 style).

Loads overall_score from per-run JSON result files and aggregates over seeds.
No hardcoded score values.

Directory layout expected under --results-dir:
    sac_single_scenario/evaluation/per_run/
        sac_<train>_safety_first_seed<N>_on_<test>.json          (full obs.)
        sac_<train>_safety_first_seed<N>_simple_on_<test>.json   (compact obs.)

Usage:
    python analysis/plot_generalization.py \\
        --results-dir /path/to/results \\
        [--output-dir /path/to/figures]
"""

import argparse
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Scenario ordering and labels ──────────────────────────────────────────────

SCENARIOS = [
    'cold_winter', 'high_load', 'low_load',
    'nominal',     'shock_load', 'temperature_drop',
]
TRAIN_LABELS = [
    'Cold Winter', 'High Load', 'Low Load',
    'Nominal',     'Shock Load', 'Temp. Drop',
]
TEST_LABELS_SHORT = [
    'Cold W.', 'High L.', 'Low L.',
    'Nominal', 'Shock L.', 'Temp. D.',
]

# Okabe-Ito colorblind-safe palette (Okabe & Ito 2008; Nature Methods 2011)
SCEN_COLORS = [
    '#0072B2',  # Cold Winter  — blue
    '#D55E00',  # High Load    — vermillion
    '#009E73',  # Low Load     — bluish green
    '#CC79A7',  # Nominal      — reddish purple
    '#56B4E9',  # Shock Load   — sky blue
    '#E69F00',  # Temp. Drop   — orange
]

CLIP_LO = -0.20
X_LO    = -0.25
X_HI    =  1.05


# ── Data loading ──────────────────────────────────────────────────────────────

def load_generalization_matrix(per_run_dir: pathlib.Path,
                                obs_mode: str) -> np.ndarray:
    """Return (n_train x n_test) mean overall_score matrix.

    Aggregates over all seeds found in *per_run_dir*.  Missing cells remain NaN.
    """
    n      = len(SCENARIOS)
    matrix = np.full((n, n), np.nan)

    for ri, train_sc in enumerate(SCENARIOS):
        for ci, test_sc in enumerate(SCENARIOS):
            scores = []
            # Match both 'seed42_on_' (full) and 'seed42_simple_on_' (compact)
            pattern = f'sac_{train_sc}_safety_first_seed*_*on_{test_sc}.json'
            for fpath in per_run_dir.glob(pattern):
                try:
                    rec = json.loads(fpath.read_text()).get('record', {})
                    if (rec.get('obs_mode') == obs_mode
                            and rec.get('reward_config') == 'safety_first'):
                        scores.append(rec['overall_score'])
                except Exception:
                    continue
            if scores:
                matrix[ri, ci] = float(np.mean(scores))

    return matrix


# ── Drawing helpers ────────────────────────────────────────────────────────────

def _draw_break_marks(ax, x, y_center,
                      half_h: float = 0.26,
                      dx: float = 0.022,
                      color: str = '#555555') -> None:
    for yo in (-0.10, 0.10):
        yc = y_center + yo
        ax.plot([x - dx, x + dx], [yc - half_h, yc + half_h],
                color=color, lw=0.9, clip_on=True, zorder=8,
                solid_capstyle='round')


# ── Figure ────────────────────────────────────────────────────────────────────

def build_figure(full: np.ndarray,
                 simple: np.ndarray,
                 output_dir: pathlib.Path) -> None:
    N_TRAIN = len(SCENARIOS)
    N_TEST  = len(SCENARIOS)

    plt.rcParams.update({
        'font.family':     'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size':       6,
        'axes.linewidth':  0.6,
        'svg.fonttype':    'none',  # editable text in SVG / PowerPoint
    })

    fig, axes = plt.subplots(
        N_TRAIN, 2,
        figsize=(3.5, 8.0),
        gridspec_kw={
            'hspace': 0.50, 'wspace': 0.52,
            'top': 0.97, 'bottom': 0.04,
            'left': 0.20, 'right': 0.98,
        },
    )

    variants = [
        (full,   'SAC (full obs.)'),
        (simple, 'SAC (compact obs.)'),
    ]

    for col_idx, (data, _col_title) in enumerate(variants):
        for row_idx in range(N_TRAIN):
            ax     = axes[row_idx, col_idx]
            scores = data[row_idx]

            for ti in range(N_TEST):
                raw = scores[ti]
                if np.isnan(raw):
                    continue
                clipped = raw < CLIP_LO
                disp    = max(raw, CLIP_LO)
                col     = SCEN_COLORS[ti]

                is_diag = (ti == row_idx)
                lw = 1.0 if is_diag else 0.4
                ec = '#111111' if is_diag else col

                ax.barh(ti, disp, height=0.70,
                        color=col, edgecolor=ec, linewidth=lw,
                        alpha=0.90, zorder=3)

                if clipped:
                    _draw_break_marks(ax, CLIP_LO, ti)
                    ax.text(CLIP_LO + 0.01, ti, f'{raw:.1f}',
                            ha='left', va='center', fontsize=4.8,
                            color='#222222', fontweight='bold', clip_on=True)

            # Zero reference
            ax.axvline(0, color='#888888', lw=0.6, ls='--', zorder=2)

            ax.set_xlim(X_LO, X_HI)
            ax.set_ylim(-0.5, N_TEST - 0.5)
            ax.set_yticks(range(N_TEST))
            ax.set_yticklabels(
                [f'{lbl} -' for lbl in TEST_LABELS_SHORT],
                fontsize=4.2, fontweight='bold',
            )
            for lbl in ax.get_yticklabels():
                lbl.set_ha('right')
            ax.tick_params(axis='y', length=0, pad=0)
            ax.tick_params(axis='x', labelsize=5.0, direction='in', length=1.5)
            ax.xaxis.set_major_locator(
                mticker.FixedLocator([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
            ax.tick_params(axis='x', which='major', length=3.0, direction='in')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='x', ls=':', lw=0.30, color='#D8D8D8', zorder=0)

            if col_idx == 0:
                letter = chr(ord('a') + row_idx)
                ax.text(
                    -0.38, 1.15,
                    f'({letter}) {TRAIN_LABELS[row_idx]}',
                    transform=ax.transAxes,
                    fontsize=6.5, fontweight='bold', va='top',
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'fig_generalization_v4.pdf',
                bbox_inches='tight', dpi=300)
    fig.savefig(output_dir / 'fig_generalization_v4.png',
                bbox_inches='tight', dpi=600)
    print(f'Saved: {output_dir}/fig_generalization_v4.pdf / .png (600 dpi)')
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Cross-scenario generalisation figure (reads from JSON).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--results-dir', required=True,
        help='Root results directory containing '
             'sac_single_scenario/evaluation/per_run/.',
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='Directory to write figures (default: <results-dir>/figures).',
    )
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir).resolve()
    per_run     = results_dir / 'sac_single_scenario' / 'evaluation' / 'per_run'
    if not per_run.is_dir():
        raise FileNotFoundError(f'per_run directory not found: {per_run}')

    output_dir = (
        pathlib.Path(args.output_dir).resolve()
        if args.output_dir
        else results_dir / 'figures'
    )

    print('Loading generalisation data ...')
    full   = load_generalization_matrix(per_run, 'full')
    simple = load_generalization_matrix(per_run, 'simple')

    n = len(SCENARIOS)
    print(f'  Full obs.:    {int(np.sum(~np.isnan(full)))}/{n * n} cells loaded')
    print(f'  Compact obs.: {int(np.sum(~np.isnan(simple)))}/{n * n} cells loaded')

    build_figure(full, simple, output_dir)


if __name__ == '__main__':
    main()

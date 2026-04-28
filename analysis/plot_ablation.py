"""
Reward ablation figure (v1 style).

Compares Safety-First vs. linear-penalty-only vs. constant-penalty-only on
three in-distribution scenarios (nominal, high_load, cold_winter).
Score bars (left axis) + violation-rate lines (right axis, inverted).
Single-column width (3.5 in).

Loads data from per-run JSON result files; no hardcoded score values.

Directory layout expected under --results-dir:
    paper_direction_a/evaluation/per_run/
        sac_<sc>_safety_first_seed<N>_on_<sc>.json
    paper_direction_a_ablation_rerun/evaluation/per_run/
        sac_<sc>_sf_linear_only_seed<N>_on_<sc>.json
    paper_direction_a_ablation_const_rerun/evaluation/per_run/
        sac_<sc>_sf_constant_only_seed<N>_on_<sc>.json

Usage:
    python analysis/plot_ablation.py \\
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
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.legend_handler import HandlerTuple


# ── Scenarios ─────────────────────────────────────────────────────────────────

ABLATION_SCENARIOS = ['nominal', 'high_load', 'cold_winter']
SCENARIO_LABELS    = ['Nominal', 'High\nLoad', 'Cold\nWinter']

# ── Method styles ─────────────────────────────────────────────────────────────

STYLES = {
    'safety_first': dict(
        label='Safety-First\n(linear + const.)',
        color='#003F88', hatch=None,  lw=2.0, ls='-',  mk='*', ms=9,
    ),
    'sf_constant_only': dict(
        label='Constant-penalty\nonly',
        color='#E07B39', hatch='///', lw=1.5, ls='--', mk='D', ms=6,
    ),
    'sf_linear_only': dict(
        label='Linear-penalty\nonly',
        color='#C00000', hatch='xxx', lw=1.5, ls=':',  mk='s', ms=6,
    ),
}
REWARD_ORDER = ['safety_first', 'sf_constant_only', 'sf_linear_only']


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_diagonal(per_run_dir: pathlib.Path,
                   reward_config: str,
                   obs_mode: str = 'full') -> dict:
    """Return {'means', 'stds', 'viols'} lists for ABLATION_SCENARIOS.

    Only considers in-distribution files (train_scenario == test_scenario).
    Aggregates overall_score and violation_rate over all available seeds.
    """
    means, stds, viols = [], [], []
    for sc in ABLATION_SCENARIOS:
        scores, vrates = [], []
        pattern = f'sac_{sc}_{reward_config}_seed*_*on_{sc}.json'
        for fpath in per_run_dir.glob(pattern):
            try:
                rec = json.loads(fpath.read_text()).get('record', {})
                if (rec.get('obs_mode') == obs_mode
                        and rec.get('reward_config') == reward_config):
                    scores.append(rec['overall_score'])
                    vrates.append(rec['violation_rate'])
            except Exception:
                continue
        if scores:
            means.append(float(np.mean(scores)))
            stds.append(float(np.std(scores, ddof=1) if len(scores) > 1 else 0.0))
            viols.append(float(np.mean(vrates)))
        else:
            means.append(float('nan'))
            stds.append(0.0)
            viols.append(float('nan'))

    return {'means': means, 'stds': stds, 'viols': viols}


def load_all(results_dir: pathlib.Path) -> dict:
    """Load ablation statistics for all three reward configurations."""
    dirs = {
        'safety_first':    results_dir / 'paper_direction_a' / 'evaluation' / 'per_run',
        'sf_linear_only':  results_dir / 'paper_direction_a_ablation_rerun'
                           / 'evaluation' / 'per_run',
        'sf_constant_only': results_dir / 'paper_direction_a_ablation_const_rerun'
                            / 'evaluation' / 'per_run',
    }

    data = {}
    for rc, per_run_dir in dirs.items():
        if not per_run_dir.is_dir():
            print(f'  [WARN] directory not found, skipping {rc}: {per_run_dir}')
            data[rc] = {'means': [float('nan')] * 3,
                        'stds':  [0.0] * 3,
                        'viols': [float('nan')] * 3}
        else:
            data[rc] = _load_diagonal(per_run_dir, rc)
    return data


# ── Figure ────────────────────────────────────────────────────────────────────

def build_figure(data: dict, output_dir: pathlib.Path) -> None:
    x   = np.arange(len(ABLATION_SCENARIOS))
    N_M = len(REWARD_ORDER)

    S_LO, S_HI   = -0.20,  1.25
    V_BOT, V_TOP =  1.875, -0.09
    CLIP_LO      = -0.18

    BAR_W   = 0.22
    offsets = np.linspace(
        -(N_M - 1) / 2 * BAR_W, (N_M - 1) / 2 * BAR_W, N_M)

    plt.rcParams.update({
        'font.family':     'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size':       7,
        'axes.linewidth':  0.7,
    })

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax2 = ax.twinx()

    # Background
    ax.axhspan(S_LO, 0, color='#FDECEA', alpha=0.4, zorder=0)
    ax.axhline(0,   color='#BBBBBB', lw=1.2, ls=(0, (4, 3)), zorder=1)
    for yv in [0.5, 1.0]:
        ax.axhline(yv, color='#E8E8E8', lw=0.8, ls=(0, (4, 3)), zorder=0)

    bar_handles = []
    for mi, rc in enumerate(REWARD_ORDER):
        st   = STYLES[rc]
        col  = st['color']
        mean = np.array(data[rc]['means'])
        std  = np.array(data[rc]['stds'])

        for si in range(len(ABLATION_SCENARIOS)):
            m  = mean[si]
            if np.isnan(m):
                continue
            bx = x[si] + offsets[mi]
            dm = max(m, CLIP_LO)
            ax.bar(bx, dm, width=BAR_W,
                   color=col, edgecolor=col, linewidth=0.7,
                   hatch=st['hatch'], alpha=0.80, zorder=3)
            if std[si] > 0:
                ax.errorbar(
                    bx, dm,
                    yerr=[[min(std[si], dm - S_LO)],
                          [min(std[si], S_HI - dm)]],
                    fmt='none', ecolor=col, elinewidth=1.0,
                    capsize=2.0, capthick=0.8, alpha=0.8, zorder=5,
                )
            if m < CLIP_LO:
                ax.text(bx, CLIP_LO - 0.01, f'{m:.2f}',
                        ha='center', va='top', fontsize=4.8,
                        color=col, rotation=90)

        bar_handles.append(mpatches.Patch(
            facecolor=col, edgecolor=col, linewidth=0.7,
            hatch=st['hatch'],
            label=st['label'].replace('\n', ' '),
        ))

    # Violation-rate lines
    line_handles = []
    for mi, rc in enumerate(REWARD_ORDER):
        st   = STYLES[rc]
        col  = st['color']
        viol = np.array(data[rc]['viols'])
        valid = ~np.isnan(viol)
        ax2.plot(x[valid], viol[valid],
                 color=col, lw=st['lw'], ls=st['ls'], alpha=0.9, zorder=6)
        ax2.scatter(x[valid], viol[valid],
                    marker=st['mk'], s=st['ms'] ** 2,
                    facecolors=col, edgecolors='white',
                    linewidths=0.6, zorder=7)
        line_handles.append(
            mlines.Line2D([], [], color=col, lw=st['lw'], ls=st['ls'],
                          marker=st['mk'], markersize=st['ms'] * 0.65,
                          markerfacecolor=col,
                          markeredgecolor='white', markeredgewidth=0.5)
        )

    # Axis formatting
    ax.set_xlim(-0.5, len(ABLATION_SCENARIOS) - 0.5)
    ax.set_ylim(S_LO, S_HI)
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIO_LABELS, fontsize=7)
    ax.set_ylabel('Overall Score  \u2191', fontsize=7)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.tick_params(direction='in', length=2.5, labelsize=6.5)
    ax.spines['top'].set_visible(False)
    ax.grid(False)

    ax2.set_ylim(V_BOT, V_TOP)
    ax2.set_ylabel('Violation Rate  \u2191', fontsize=7)
    ax2.yaxis.set_major_locator(
        mticker.FixedLocator([0, 0.375, 0.75, 1.125, 1.5]))
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2g'))
    ax2.tick_params(direction='in', length=2.5, labelsize=6.5)
    ax2.spines['top'].set_visible(False)

    # Combined legend (bar patch + violation line per method)
    combined      = [(bar_handles[i], line_handles[i]) for i in range(N_M)]
    method_labels = [STYLES[rc]['label'].replace('\n', ' ')
                     for rc in REWARD_ORDER]
    ax.legend(
        handles=combined,
        labels=method_labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
        loc='lower left',
        bbox_to_anchor=(0.0, 0.0),
        ncol=1,
        fontsize=5.8,
        framealpha=0.92,
        edgecolor='#CCCCCC',
        handlelength=3.2,
        handletextpad=0.4,
        borderpad=0.5,
    )

    fig.tight_layout(pad=0.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'fig_ablation_v1.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(output_dir / 'fig_ablation_v1.png', bbox_inches='tight', dpi=200)
    print(f'Saved: {output_dir}/fig_ablation_v1.pdf / .png')
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Reward ablation figure (reads from JSON).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--results-dir', required=True,
        help='Root results directory containing the ablation per-run '
             'subdirectories.',
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='Directory to write figures (default: <results-dir>/figures).',
    )
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir).resolve()
    output_dir  = (
        pathlib.Path(args.output_dir).resolve()
        if args.output_dir
        else results_dir / 'figures'
    )

    print('Loading ablation data ...')
    data = load_all(results_dir)

    for rc in REWARD_ORDER:
        n_loaded = sum(1 for v in data[rc]['means'] if not np.isnan(v))
        print(f'  {rc}: {n_loaded}/{len(ABLATION_SCENARIOS)} scenarios loaded')

    build_figure(data, output_dir)


if __name__ == '__main__':
    main()

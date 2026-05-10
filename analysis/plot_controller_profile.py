"""
Controller Profile Plot — Non-thermal Scenarios
================================================

Line plot (profile / parallel-coordinates style) showing overall_score and
violation_rate for all controllers across the four non-thermal ADM1 scenarios:
nominal, high_load, low_load, shock_load.

X-axis  : scenario
Y-axis  : overall_score (top panel) and violation_rate (bottom panel)
Lines   : one per controller, with markers and ± 1 std shading where n > 1

Data source: evaluation/baselines/ and evaluation/per_run/ JSONs.
SAC: in-distribution evaluation only (train_scenario == test_scenario).

Usage:
    python analysis/plot_controller_profile.py

    python analysis/plot_controller_profile.py \\
        --eval-dir /path/to/results/sac_single_scenario/evaluation \\
        --output-dir /path/to/figures
"""

import argparse
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Constants ─────────────────────────────────────────────────────────────────

SCENARIOS = ['nominal', 'high_load', 'low_load', 'shock_load']
SCENARIO_LABELS = ['Nominal', 'High\nLoad', 'Low\nLoad', 'Shock\nLoad']

SEEDS = [42, 123, 456]

# Controller display config: (key, label, color, linestyle, marker)
CONTROLLERS = [
    ('constant',     'Constant',    '#888888', '-',   'o',  False),
    ('pid',          'PID',         '#AAAAAA', '--',  's',  False),
    ('cascaded_pid', 'Cas.-PID',    '#BBBBBB', ':',   'D',  False),
    ('mpc',          'MPC',         '#E07B39', '-.',  '^',  False),
    ('nmpc_oracle',  'NMPC$^*$',    '#C04000', '--',  'v',  False),
    ('sac_full',     'SAC-Full',    '#003F88', '-',   '*',  True),
    ('sac_simple',   'SAC-Compact', '#C00000', '--',  'P',  True),
]

DEFAULT_EVAL_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / 'results' / 'sac_single_scenario' / 'evaluation'
)
DEFAULT_OUTPUT_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / 'results' / 'figures'
)


# ── Data loading ──────────────────────────────────────────────────────────────

def _agg(vals):
    clean = [v for v in vals if v is not None]
    if not clean:
        return float('nan'), 0.0
    return float(np.mean(clean)), float(np.std(clean, ddof=1) if len(clean) > 1 else 0.0)


def load_data(eval_dir: pathlib.Path) -> dict:
    """
    Returns {ctrl_key: {'scores': [mean per scenario],
                         'score_stds': [...],
                         'viols':  [mean per scenario],
                         'viol_stds':  [...]}}
    """
    per_run   = eval_dir / 'per_run'
    baselines = eval_dir / 'baselines'
    data = {}

    for ctrl_key, label, *_ in CONTROLLERS:
        scores, score_stds, viols, viol_stds = [], [], [], []

        for sc in SCENARIOS:
            if ctrl_key in ('constant', 'pid', 'cascaded_pid'):
                f = baselines / f'{ctrl_key}_on_{sc}.json'
                if f.exists():
                    r = json.loads(f.read_text()).get('record',
                        json.loads(f.read_text()))
                    sm = float(r.get('overall_score', float('nan')))
                    vm = float(r.get('violation_rate', float('nan')))
                    ss = vs = 0.0
                else:
                    sm = ss = vm = vs = float('nan')

            elif ctrl_key in ('mpc', 'nmpc_oracle'):
                sv, vv = [], []
                for seed in SEEDS:
                    f = per_run / f'{ctrl_key}_{sc}_seed{seed}_on_{sc}.json'
                    if not f.exists():
                        continue
                    d = json.loads(f.read_text())
                    s = d.get('overall_score')
                    v = d.get('violation_rate')
                    if s is not None: sv.append(float(s))
                    if v is not None: vv.append(float(v))
                sm, ss = _agg(sv)
                vm, vs = _agg(vv)

            else:  # SAC full / simple
                suffix = '_simple' if ctrl_key == 'sac_simple' else ''
                sv, vv = [], []
                # In-distribution: look across all train scenarios
                for train_sc in ['nominal', 'high_load', 'low_load', 'shock_load',
                                  'temperature_drop', 'cold_winter']:
                    for seed in SEEDS:
                        fname = (f'sac_{train_sc}_safety_first'
                                 f'_seed{seed}{suffix}_on_{sc}.json')
                        f = per_run / fname
                        if not f.exists():
                            continue
                        d = json.loads(f.read_text())
                        r = d.get('record', d)
                        if r.get('train_scenario', train_sc) != sc:
                            continue
                        s = r.get('overall_score')
                        v = r.get('violation_rate')
                        if s is not None: sv.append(float(s))
                        if v is not None: vv.append(float(v))
                sm, ss = _agg(sv)
                vm, vs = _agg(vv)

            scores.append(sm);      score_stds.append(ss)
            viols.append(vm);       viol_stds.append(vs)

        data[ctrl_key] = {
            'scores':     np.array(scores),
            'score_stds': np.array(score_stds),
            'viols':      np.array(viols),
            'viol_stds':  np.array(viol_stds),
        }

    return data


# ── Figure ────────────────────────────────────────────────────────────────────

def build_figure(
    data: dict,
    output_dir: pathlib.Path,
    dpi: int = 300,
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
        'xtick.labelsize':  7,
        'ytick.labelsize':  6.5,
    })

    x = np.arange(len(SCENARIOS))
    fig, (ax_s, ax_v) = plt.subplots(
        2, 1, figsize=(5.0, 4.8),
        sharex=True,
        gridspec_kw={'height_ratios': [1.15, 1]},
    )

    for ctrl_key, label, color, ls, mk, is_sac in CONTROLLERS:
        d      = data[ctrl_key]
        scores = d['scores']
        sstds  = d['score_stds']
        viols  = d['viols']
        vstds  = d['viol_stds']

        lw  = 1.8 if is_sac else 1.1
        ms  = 7   if mk == '*' else 5
        zord = 4  if is_sac else 2
        alpha_band = 0.15

        # ── Score panel ───────────────────────────────────────────────────
        valid = ~np.isnan(scores)
        ax_s.plot(x[valid], scores[valid],
                  color=color, lw=lw, ls=ls,
                  marker=mk, markersize=ms,
                  label=label, zorder=zord)
        mask_std = valid & (sstds > 0.001)
        if mask_std.any():
            ax_s.fill_between(
                x[mask_std],
                scores[mask_std] - sstds[mask_std],
                scores[mask_std] + sstds[mask_std],
                color=color, alpha=alpha_band, linewidth=0,
            )

        # ── Violation panel ───────────────────────────────────────────────
        valid_v = ~np.isnan(viols)
        ax_v.plot(x[valid_v], viols[valid_v] * 100,
                  color=color, lw=lw, ls=ls,
                  marker=mk, markersize=ms,
                  zorder=zord)
        mask_vstd = valid_v & (vstds > 0.001)
        if mask_vstd.any():
            ax_v.fill_between(
                x[mask_vstd],
                (viols[mask_vstd] - vstds[mask_vstd]) * 100,
                (viols[mask_vstd] + vstds[mask_vstd]) * 100,
                color=color, alpha=alpha_band, linewidth=0,
            )

    # ── Score axis formatting ─────────────────────────────────────────────
    ax_s.set_ylabel('Overall Score  $\\uparrow$', fontsize=7)
    ax_s.set_ylim(-0.05, 1.08)
    ax_s.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax_s.axhline(1.0, color='#DDDDDD', lw=0.6, ls='--', zorder=0)
    ax_s.spines['top'].set_visible(False)
    ax_s.spines['right'].set_visible(False)

    # Legend inside score panel
    ax_s.legend(
        loc='lower left',
        ncol=2,
        fontsize=6.0,
        framealpha=0.93,
        edgecolor='#CCCCCC',
        handlelength=2.2,
        handletextpad=0.3,
        columnspacing=0.8,
        borderpad=0.5,
    )

    # ── Violation axis formatting ─────────────────────────────────────────
    ax_v.set_ylabel('Violation Rate  $\\downarrow$ (%)', fontsize=7)
    ax_v.set_ylim(-2, 105)
    ax_v.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax_v.axhline(0, color='#DDDDDD', lw=0.6, ls='--', zorder=0)
    ax_v.spines['top'].set_visible(False)
    ax_v.spines['right'].set_visible(False)

    # ── X axis ───────────────────────────────────────────────────────────
    ax_v.set_xticks(x)
    ax_v.set_xticklabels(SCENARIO_LABELS, fontsize=7)
    ax_v.set_xlim(-0.3, len(SCENARIOS) - 0.7)

    # Panel labels
    ax_s.text(0.01, 0.98, '(a)', transform=ax_s.transAxes,
              fontsize=7, va='top', fontweight='bold')
    ax_v.text(0.01, 0.98, '(b)', transform=ax_v.transAxes,
              fontsize=7, va='top', fontweight='bold')

    fig.tight_layout(pad=0.6, h_pad=0.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = 'fig_controller_profile'
    fig.savefig(output_dir / f'{stem}.pdf', bbox_inches='tight', dpi=dpi)
    fig.savefig(output_dir / f'{stem}.png', bbox_inches='tight', dpi=200)
    print(f'Saved: {output_dir}/{stem}.pdf / .png')
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Controller profile line plot across non-thermal scenarios.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--eval-dir',   type=pathlib.Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument('--output-dir', type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    print('Loading evaluation data ...')
    data = load_data(args.eval_dir)
    build_figure(data, args.output_dir, dpi=args.dpi)


if __name__ == '__main__':
    main()

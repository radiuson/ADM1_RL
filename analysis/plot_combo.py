"""
Combined comparison figure (v8 style — simplified).

Shows overall score (bar, left axis) and safety violation rate (line, right
axis) for six controllers across six evaluation scenarios:
  Cascaded PID, PID, Constant, MPC, SAC (compact obs.), SAC (full obs.)

NMPC and multi-scenario SAC variants removed from this version.

Usage:
    python analysis/plot_combo.py \\
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


# ── Scenario ordering ─────────────────────────────────────────────────────────

SCENARIOS = [
    'nominal', 'high_load', 'shock_load',
    'low_load', 'temperature_drop', 'cold_winter',
]
SCENARIO_LABELS = [
    'Nominal', 'High\nLoad', 'Shock\nLoad',
    'Low\nLoad', 'Temp.\nDrop', 'Cold\nWinter',
]

NAN = float('nan')


# ── Data loading ──────────────────────────────────────────────────────────────

def _aggregate(files: list[pathlib.Path],
               obs_mode: str | None,
               reward_config: str | None) -> tuple[list[float], list[float], list[float]]:
    """Return (scores, viols, ch4s) over matching files.

    ch4 field name differs across controller types:
      SAC per-run → ch4_avg
      engineering baselines → avg_ch4
      MPC per-run → avg_ch4_flow
    """
    scores, viols, ch4s = [], [], []
    for fpath in files:
        try:
            raw = json.loads(fpath.read_text())
            rec = raw.get('record', raw)
            if obs_mode is not None and rec.get('obs_mode') != obs_mode:
                continue
            if reward_config is not None and rec.get('reward_config') != reward_config:
                continue
            scores.append(rec['overall_score'])
            viols.append(rec['violation_rate'])
            ch4 = NAN
            for key in ('ch4_avg', 'avg_ch4', 'avg_ch4_flow'):
                if key in rec and rec[key] is not None:
                    ch4 = float(rec[key])
                    break
            ch4s.append(ch4)
        except Exception:
            continue
    return scores, viols, ch4s


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return NAN, 0.0
    arr = np.array(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0)


def load_naive_multi(multi_per_run_dir: pathlib.Path,
                     obs_mode: str) -> tuple[list, list, list]:
    """Load naive multi-scenario SAC (all-6 simultaneous, no curriculum); returns (ch4_means, ch4_stds, viols)."""
    ch4_means, ch4_stds, viols = [], [], []
    for sc in SCENARIOS:
        pattern = f'sac_*_safety_first_seed*_*on_{sc}.json'
        files   = list(multi_per_run_dir.glob(pattern))
        _, vr, ch4s = _aggregate(files, obs_mode, 'safety_first')
        valid_ch4 = [c for c in ch4s if not np.isnan(c)]
        c, cs = _mean_std(valid_ch4)
        ch4_means.append(c)
        ch4_stds.append(cs)
        viols.append(float(np.mean(vr)) if vr else NAN)
    return ch4_means, ch4_stds, viols


def load_baseline(per_run_dir: pathlib.Path,
                  prefix: str) -> tuple[list, list]:
    """Load MPC-style baseline; returns (ch4_means, viols)."""
    ch4_means, viols = [], []
    for sc in SCENARIOS:
        pattern = f'{prefix}_{sc}_seed*_on_{sc}.json'
        files   = list(per_run_dir.glob(pattern))
        _, vr, ch4s = _aggregate(files, None, None)
        valid_ch4 = [c for c in ch4s if not np.isnan(c)]
        ch4_means.append(float(np.mean(valid_ch4)) if valid_ch4 else NAN)
        viols.append(float(np.mean(vr)) if vr else NAN)
    return ch4_means, viols


def _load_eng_baseline(baselines_dir: pathlib.Path,
                       ctrl_file: str, thermal_file: str) -> tuple[list, list]:
    """Load engineering baseline CH4 production and violation rate across all 6 scenarios."""
    THERMAL = {'temperature_drop', 'cold_winter'}
    ch4_means, viols = [], []
    for sc in SCENARIOS:
        fname = thermal_file if sc in THERMAL else ctrl_file
        fname_sc = fname.replace('<SC>', sc)
        fpath = baselines_dir / fname_sc
        if fpath.exists():
            rec = json.loads(fpath.read_text()).get('record', {})
            viols.append(float(rec.get('violation_rate', NAN)))
            ch4 = NAN
            for key in ('ch4_avg', 'avg_ch4', 'avg_ch4_flow'):
                if key in rec and rec[key] is not None:
                    ch4 = float(rec[key])
                    break
            ch4_means.append(ch4)
        else:
            ch4_means.append(NAN)
            viols.append(NAN)
    return ch4_means, viols


def load_sdc_sac(results_dir: pathlib.Path,
                 stages_name: str = 'fast') -> tuple[list, list, list]:
    """Load SDC-SAC per-scenario results (all 6 scenarios, single policy).

    Returns (ch4_means, ch4_stds, viols) in SCENARIOS order.
    """
    fpath = results_dir / 'evaluation_60d' / 'scenario_cur_results.json'
    data  = json.loads(fpath.read_text())
    records = [r for r in data if r.get('stages_name') == stages_name]

    from collections import defaultdict
    by_sc: dict[str, list] = defaultdict(list)
    for r in records:
        by_sc[r['test_scenario']].append(r)

    ch4_means, ch4_stds, viols = [], [], []
    for sc in SCENARIOS:
        recs = by_sc.get(sc, [])
        ch4_vals = [r['ch4_avg']      for r in recs]
        vr_vals  = [r['violation_rate'] for r in recs]
        c, cs = _mean_std(ch4_vals)
        ch4_means.append(c)
        ch4_stds.append(cs)
        viols.append(float(np.mean(vr_vals)) if vr_vals else NAN)
    return ch4_means, ch4_stds, viols


def load_all(results_dir: pathlib.Path) -> dict:
    per_run       = results_dir / 'sac_single_scenario' / 'evaluation' / 'per_run'
    multi_per_run = results_dir / 'sac_multi_scenario'  / 'evaluation' / 'per_run'
    baselines_dir = results_dir / 'sac_single_scenario' / 'evaluation' / 'baselines'

    if not per_run.is_dir():
        raise FileNotFoundError(f'Directory not found: {per_run}')

    naive_ch4, naive_ch4_std, naive_viol = load_naive_multi(multi_per_run, 'full')
    sdc_ch4,   sdc_ch4_std,   sdc_viol   = load_sdc_sac(results_dir, 'fast')
    mpc_ch4,   mpc_viol                  = load_baseline(per_run, 'mpc')

    const_ch4, const_viol = _load_eng_baseline(
        baselines_dir,
        ctrl_file    = 'constant_on_<SC>.json',
        thermal_file = 'constant_thermal_on_<SC>.json',
    )
    pid_ch4, pid_viol = _load_eng_baseline(
        baselines_dir,
        ctrl_file    = 'pid_on_<SC>.json',
        thermal_file = 'full_pid_on_<SC>.json',
    )
    cpid_ch4, cpid_viol = _load_eng_baseline(
        baselines_dir,
        ctrl_file    = 'cascaded_pid_on_<SC>.json',
        thermal_file = 'cascaded_pid_thermal_on_<SC>.json',
    )

    loaded = {
        'Constant':     (const_ch4,  const_viol),
        'PID':          (pid_ch4,    pid_viol),
        'Cascaded PID': (cpid_ch4,   cpid_viol),
    }
    for name, (ch4s, _) in loaded.items():
        n_ok = sum(1 for v in ch4s if not np.isnan(v))
        print(f'  {name}: {n_ok}/6 scenarios loaded')

    return {
        'SDC-SAC': dict(
            ch4=sdc_ch4,   ch4_std=sdc_ch4_std,   viol=sdc_viol,
            color='#007B6E', lw=2.2, ls='-',   mk='*', ms=11, hatch=None,  sac=True,
        ),
        'SAC (no curriculum)': dict(
            ch4=naive_ch4, ch4_std=naive_ch4_std, viol=naive_viol,
            color='#4472C4', lw=1.8, ls='--',  mk='D', ms=7,  hatch=None,  sac=True,
        ),
        'MPC': dict(
            ch4=mpc_ch4,   ch4_std=None,          viol=mpc_viol,
            color='#70AD47', lw=1.4, ls=':',   mk='D', ms=5,  hatch='xx',  sac=False,
        ),
        'Constant': dict(
            ch4=const_ch4, ch4_std=None,          viol=const_viol,
            color='#7F7F7F', lw=1.3, ls='--',  mk='o', ms=6,  hatch='//',  sac=False,
        ),
        'PID': dict(
            ch4=pid_ch4,   ch4_std=None,          viol=pid_viol,
            color='#ED7D31', lw=1.3, ls='--',  mk='s', ms=6,  hatch='\\\\', sac=False,
        ),
        'Cascaded PID': dict(
            ch4=cpid_ch4,  ch4_std=None,          viol=cpid_viol,
            color='#C00000', lw=1.3, ls='--',  mk='^', ms=6,  hatch='||',  sac=False,
        ),
    }


# ── Figure ────────────────────────────────────────────────────────────────────

METHODS_ORDER = [
    'Cascaded PID', 'PID', 'Constant', 'MPC',
    'SAC (no curriculum)', 'SDC-SAC',
]


def build_figure(data: dict, output_dir: pathlib.Path) -> None:
    x   = np.arange(len(SCENARIOS))
    N_M = len(METHODS_ORDER)

    S_LO, S_HI   =    0,  2800      # CH4 production axis (m³/d)
    # V_BOT / V_TOP chosen so that:
    #   VR = 0.0  aligns with CH4 ≈ 2400 m³/d (88 % of axis height)
    #   VR = 1.0  aligns with CH4 ≈  500 m³/d (18 % of axis height)
    V_BOT, V_TOP =  1.26,  -0.21
    CLIP_LO      =    0

    BAR_W   = 0.060
    offsets = np.linspace(
        -(N_M - 1) / 2 * BAR_W, (N_M - 1) / 2 * BAR_W, N_M)

    plt.rcParams.update({
        'font.family':  'DejaVu Sans',
        'font.size':    9,
        'axes.linewidth': 0.8,
    })

    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax2 = ax.twinx()

    # Background
    ax.axvspan(3.5, 5.5, color='#F0F4FF', alpha=0.50, zorder=0)
    for yval in range(0, 2801, 500):
        ax.axhline(yval, color='#DDDDDD', lw=1.2, ls=(0, (4, 3)), zorder=0)

    legend_handles = []
    for mi, name in enumerate(METHODS_ORDER):
        d       = data[name]
        col     = d['color']
        ch4     = np.array(d['ch4'],     dtype=float)
        ch4_std = np.array(d['ch4_std'], dtype=float) if d['ch4_std'] is not None else None

        for si in range(len(SCENARIOS)):
            v = ch4[si]
            if np.isnan(v):
                continue
            bx = x[si] + offsets[mi]
            ax.bar(bx, v, width=BAR_W,
                   color=col if d['sac'] else 'none',
                   edgecolor=col, linewidth=0.9,
                   hatch=d['hatch'],
                   alpha=0.75 if d['sac'] else 1.0,
                   zorder=3)
            if d['sac'] and ch4_std is not None and ch4_std[si] > 0:
                ax.errorbar(
                    bx, v,
                    yerr=[[min(ch4_std[si], v - S_LO)],
                          [min(ch4_std[si], S_HI - v)]],
                    fmt='none', ecolor=col, elinewidth=1.1,
                    capsize=2.5, capthick=1.0, alpha=0.75, zorder=5,
                )

        bar_proxy  = mpatches.Patch(
            facecolor=col if d['sac'] else 'none',
            edgecolor=col, linewidth=0.9, hatch=d['hatch'],
        )
        line_proxy = mlines.Line2D(
            [], [], color=col, ls=d['ls'], lw=d['lw'],
            marker=d['mk'], markersize=d['ms'] * 0.6,
            markerfacecolor=col if d['sac'] else 'none',
            markeredgecolor=col,
        )
        legend_handles.append((bar_proxy, line_proxy))

    # Violation-rate lines
    for name in METHODS_ORDER:
        d    = data[name]
        col  = d['color']
        viol = np.array(d['viol'], dtype=float)
        valid = ~np.isnan(viol)
        ax2.plot(x[valid], viol[valid],
                 color=col, lw=d['lw'], ls=d['ls'], alpha=0.95, zorder=6)
        for si in range(len(SCENARIOS)):
            if not valid[si]:
                continue
            ax2.scatter(si, viol[si],
                        marker=d['mk'], s=d['ms'] ** 2,
                        facecolors=col if d['sac'] else 'none',
                        edgecolors='white' if d['sac'] else col,
                        linewidths=0.8 if d['sac'] else 1.0,
                        zorder=7)

    # Axis formatting
    ax.set_xlim(-0.60, len(SCENARIOS) - 0.40)
    ax.set_ylim(S_LO, S_HI)
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIO_LABELS, fontsize=10)
    ax.set_xlabel('Evaluation Scenario', fontsize=11, labelpad=5)
    ax.set_ylabel('CH\u2084 Production (m\u00b3/d)', fontsize=11)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(500))
    ax.tick_params(direction='in', length=3)
    ax.spines['top'].set_visible(False)
    ax.grid(False)

    ax2.set_ylim(V_BOT, V_TOP)
    ax2.set_ylabel('Safety Violation Rate', fontsize=11)
    ax2.yaxis.set_major_locator(
        mticker.FixedLocator([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    ax2.tick_params(direction='in', length=3)
    ax2.spines['top'].set_visible(False)

    # 5 evenly-spaced reference dashed lines across the VR [0, 1] range.
    # With the inverted axis (V_BOT=4, V_TOP=-0.16), VR values in [0,1] appear
    # in the upper portion of the plot, aligned with left-axis 1.0–1.5.
    for vr_ref in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        ax2.axhline(vr_ref, color='#BBBBBB', lw=0.7,
                    ls=(0, (4, 3)), zorder=0)

    # Legend — row-major reorder for ncol=3 (6 methods → 2×3 grid)
    _ncol  = 3
    _n     = len(METHODS_ORDER)
    _nrows = int(np.ceil(_n / _ncol))
    _reorder = [None] * _n
    for _di in range(_n):
        _r, _c = _di // _ncol, _di % _ncol
        _si = _r + _c * _nrows
        if _si < _n:
            _reorder[_si] = _di
    _reorder = [i for i in _reorder if i is not None]

    ax.legend(
        handles=[legend_handles[i] for i in _reorder],
        labels=[METHODS_ORDER[i] for i in _reorder],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)},
        loc='lower left',
        bbox_to_anchor=(0.01, 0.01),
        ncol=3,
        fontsize=8.0,
        framealpha=0.92,
        edgecolor='#CCCCCC',
        handlelength=3.0,
        handletextpad=0.5,
        columnspacing=1.0,
        borderpad=0.6,
    )

    # ── Top graphical indicators ───────────────────────────────────────────────
    # Left: small bar icon  →  "Bars: CH₄ production"
    bar_icon = mpatches.Rectangle(
        (0.010, 1.024), 0.016, 0.060,
        transform=ax.transAxes,
        facecolor='#444444', edgecolor='none',
        clip_on=False, zorder=12,
    )
    ax.add_patch(bar_icon)
    ax.text(0.033, 1.054,
            'Bars — CH₄ production (m³/d)',
            transform=ax.transAxes, fontsize=9,
            va='center', ha='left', color='#222222',
            clip_on=False)

    # Right: small line + marker icon  →  "Lines: safety violation rate"
    lx = [0.956, 0.969, 0.982]
    ly = [1.054, 1.054, 1.054]
    ax.plot(lx, ly, '-', color='#444444', lw=2.0,
            transform=ax.transAxes, clip_on=False, zorder=12)
    ax.plot([lx[1]], [ly[1]], 'o', ms=5, color='#444444',
            transform=ax.transAxes, clip_on=False, zorder=12)
    ax.text(0.950, 1.054,
            'Lines — safety violation rate',
            transform=ax.transAxes, fontsize=9,
            va='center', ha='right', color='#222222',
            clip_on=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'fig_combo_v8.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(output_dir / 'fig_combo_v8.png', bbox_inches='tight', dpi=200)
    print(f'Saved: {output_dir}/fig_combo_v8.pdf / .png')
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Combined comparison figure (reads SAC / MPC / NMPC from JSON).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--results-dir', required=True,
        help='Root results directory containing sac_single_scenario/ and '
             'sac_multi_scenario/ subdirectories.',
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

    print('Loading result data ...')
    data = load_all(results_dir)

    for name in METHODS_ORDER:
        n_ok = sum(1 for v in data[name]['ch4'] if not np.isnan(v))
        print(f'  {name}: {n_ok}/{len(SCENARIOS)} scenarios loaded')

    build_figure(data, output_dir)


if __name__ == '__main__':
    main()

"""
Combined comparison figure (v7 style).

Shows overall score (bar, left axis) and safety violation rate (line, right
axis) for all methods across six evaluation scenarios.

SAC single-scenario, SAC multi-scenario, MPC, and NMPC values are loaded from
per-run JSON result files.  PID, Constant, and Cascaded-PID values remain
hardcoded because no per-run JSON files exist for those controllers; re-run
full_evaluation.py with those controllers to regenerate their JSON files.

Directory layout expected under --results-dir:
    single_scenario/evaluation/per_run/
        sac_<sc>_safety_first_seed<N>_on_<sc>.json
        sac_<sc>_safety_first_seed<N>_simple_on_<sc>.json
        mpc_<sc>_seed<N>_on_<sc>.json
        nmpc_oracle_<sc>_seed<N>_on_<sc>.json
    sac_multi_scenario/evaluation/per_run/
        sac_*_safety_first_seed<N>_on_<sc>.json
        sac_*_safety_first_seed<N>_simple_on_<sc>.json

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
               reward_config: str | None) -> tuple[list[float], list[float]]:
    """Return (scores, viols) over matching files.

    Filters by obs_mode and reward_config when provided.
    For baseline JSON files the top-level dict is used directly (no 'record'
    wrapper).
    """
    scores, viols = [], []
    for fpath in files:
        try:
            raw = json.loads(fpath.read_text())
            rec = raw.get('record', raw)  # baseline files have no 'record' key
            if obs_mode is not None and rec.get('obs_mode') != obs_mode:
                continue
            if reward_config is not None and rec.get('reward_config') != reward_config:
                continue
            scores.append(rec['overall_score'])
            viols.append(rec['violation_rate'])
        except Exception:
            continue
    return scores, viols


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return NAN, 0.0
    arr = np.array(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0)


def load_sac_single(per_run_dir: pathlib.Path,
                    obs_mode: str) -> tuple[list, list, list]:
    """Load SAC single-scenario diagonal (train == test) over all seeds."""
    means, stds, viols = [], [], []
    for sc in SCENARIOS:
        pattern = f'sac_{sc}_safety_first_seed*_*on_{sc}.json'
        files   = list(per_run_dir.glob(pattern))
        scores, vr = _aggregate(files, obs_mode, 'safety_first')
        m, s = _mean_std(scores)
        means.append(m)
        stds.append(s)
        viols.append(float(np.mean(vr)) if vr else NAN)
    return means, stds, viols


def load_sac_multi(multi_per_run_dir: pathlib.Path,
                   obs_mode: str) -> tuple[list, list, list]:
    """Load SAC multi-scenario results (tested individually per scenario)."""
    means, stds, viols = [], [], []
    for sc in SCENARIOS:
        # Multi files use a list-based train scenario name; match via glob + filter
        pattern = f'sac_*_safety_first_seed*_*on_{sc}.json'
        files   = list(multi_per_run_dir.glob(pattern))
        scores, vr = _aggregate(files, obs_mode, 'safety_first')
        m, s = _mean_std(scores)
        means.append(m)
        stds.append(s)
        viols.append(float(np.mean(vr)) if vr else NAN)
    return means, stds, viols


def load_baseline(per_run_dir: pathlib.Path,
                  prefix: str) -> tuple[list, list]:
    """Load MPC-style baseline (single episode per seed; no obs_mode filter)."""
    means, viols = [], []
    for sc in SCENARIOS:
        pattern = f'{prefix}_{sc}_seed*_on_{sc}.json'
        files   = list(per_run_dir.glob(pattern))
        scores, vr = _aggregate(files, None, None)
        means.append(float(np.mean(scores)) if scores else NAN)
        viols.append(float(np.mean(vr)) if vr else NAN)
    return means, viols


def _load_eng_baseline(baselines_dir: pathlib.Path,
                       ctrl_file: str, thermal_file: str) -> tuple[list, list]:
    """
    Load engineering baseline scores/violations across all 6 scenarios.

    For thermal scenarios (temperature_drop, cold_winter) use the thermal
    controller variant (with Q_HEX set); for the other 4 use the standard file.
    Scenario order matches SCENARIOS list.
    """
    THERMAL = {'temperature_drop', 'cold_winter'}
    means, viols = [], []
    for sc in SCENARIOS:
        fname = thermal_file if sc in THERMAL else ctrl_file
        fname_sc = fname.replace('<SC>', sc)
        fpath = baselines_dir / fname_sc
        if fpath.exists():
            rec = json.loads(fpath.read_text()).get('record', {})
            means.append(float(rec.get('overall_score', NAN)))
            viols.append(float(rec.get('violation_rate', NAN)))
        else:
            means.append(NAN)
            viols.append(NAN)
    return means, viols


def load_all(results_dir: pathlib.Path) -> dict:
    per_run       = results_dir / 'single_scenario' / 'evaluation' / 'per_run'
    multi_per_run = results_dir / 'sac_multi_scenario' / 'evaluation' / 'per_run'
    baselines_dir = results_dir / 'single_scenario' / 'evaluation' / 'baselines'

    for d in (per_run, multi_per_run):
        if not d.is_dir():
            raise FileNotFoundError(f'Directory not found: {d}')

    sac_full_mean, sac_full_std, sac_full_viol = load_sac_single(per_run, 'full')
    sac_sim_mean,  sac_sim_std,  sac_sim_viol  = load_sac_single(per_run, 'simple')
    msc_full_mean, msc_full_std, msc_full_viol = load_sac_multi(multi_per_run, 'full')
    msc_sim_mean,  msc_sim_std,  msc_sim_viol  = load_sac_multi(multi_per_run, 'simple')
    mpc_mean,  mpc_viol  = load_baseline(per_run, 'mpc')
    nmpc_mean, nmpc_viol = load_baseline(per_run, 'nmpc_oracle')

    # Engineering baselines: non-thermal scenarios use Q_HEX=0 controllers;
    # thermal scenarios (temperature_drop, cold_winter) use thermal variants
    # with scenario-appropriate Q_HEX bias for a fair comparison with SAC.
    const_mean, const_viol = _load_eng_baseline(
        baselines_dir,
        ctrl_file    = 'constant_on_<SC>.json',
        thermal_file = 'constant_thermal_on_<SC>.json',
    )
    pid_mean, pid_viol = _load_eng_baseline(
        baselines_dir,
        ctrl_file    = 'pid_on_<SC>.json',
        thermal_file = 'full_pid_on_<SC>.json',
    )
    cpid_mean, cpid_viol = _load_eng_baseline(
        baselines_dir,
        ctrl_file    = 'cascaded_pid_on_<SC>.json',
        thermal_file = 'cascaded_pid_thermal_on_<SC>.json',
    )

    loaded = {
        'Constant':     (const_mean, const_viol),
        'PID':          (pid_mean,   pid_viol),
        'Cascaded PID': (cpid_mean,  cpid_viol),
    }
    for name, (means, viols) in loaded.items():
        n_ok = sum(1 for v in means if not np.isnan(v))
        print(f'  {name}: {n_ok}/6 scenarios loaded')

    return {
        'SAC (full obs.)': dict(
            mean=sac_full_mean, std=sac_full_std, viol=sac_full_viol,
            color='#003F88', lw=2.0, ls='-',   mk='*', ms=10, hatch=None,  sac=True,
        ),
        'SAC multi (full obs.)': dict(
            mean=msc_full_mean, std=msc_full_std, viol=msc_full_viol,
            color='#003F88', lw=1.6, ls='--',  mk='D', ms=7,  hatch='///', sac=True,
        ),
        'SAC (compact obs.)': dict(
            mean=sac_sim_mean, std=sac_sim_std, viol=sac_sim_viol,
            color='#4472C4', lw=1.7, ls='-',   mk='*', ms=8,  hatch=None,  sac=True,
        ),
        'SAC multi (compact obs.)': dict(
            mean=msc_sim_mean, std=msc_sim_std, viol=msc_sim_viol,
            color='#4472C4', lw=1.4, ls='--',  mk='D', ms=6,  hatch='///', sac=True,
        ),
        'NMPC (oracle)': dict(
            mean=nmpc_mean, std=None, viol=nmpc_viol,
            color='#1F7A4D', lw=1.4, ls='-.',  mk='P', ms=6,  hatch='--',  sac=False,
        ),
        'MPC': dict(
            mean=mpc_mean, std=None, viol=mpc_viol,
            color='#70AD47', lw=1.4, ls=':',   mk='D', ms=5,  hatch='xx',  sac=False,
        ),
        'Constant': dict(
            mean=const_mean, std=None, viol=const_viol,
            color='#7F7F7F', lw=1.3, ls='--',  mk='o', ms=6,  hatch='//',  sac=False,
        ),
        'PID': dict(
            mean=pid_mean, std=None, viol=pid_viol,
            color='#ED7D31', lw=1.3, ls='--',  mk='s', ms=6,  hatch='\\\\', sac=False,
        ),
        'Cascaded PID': dict(
            mean=cpid_mean, std=None, viol=cpid_viol,
            color='#C00000', lw=1.3, ls='--',  mk='^', ms=6,  hatch='||',  sac=False,
        ),
    }


# ── Figure ────────────────────────────────────────────────────────────────────

METHODS_ORDER = [
    'Cascaded PID', 'PID', 'Constant', 'MPC', 'NMPC (oracle)',
    'SAC (compact obs.)', 'SAC multi (compact obs.)',
    'SAC (full obs.)',    'SAC multi (full obs.)',
]


def build_figure(data: dict, output_dir: pathlib.Path) -> None:
    x   = np.arange(len(SCENARIOS))
    N_M = len(METHODS_ORDER)

    S_LO, S_HI   = -0.50,  1.58
    # V_BOT / V_TOP are chosen so that:
    #   VR = 0.0  aligns with left-axis value 1.5
    #   VR = 1.0  aligns with left-axis value 1.0
    # Solved from two alignment equations given S_LO=-0.5, S_HI=1.58:
    #   V_BOT = 4.0,  V_TOP = -0.16
    V_BOT, V_TOP =  4.00,  -0.16
    CLIP_LO      = -0.44

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
    ax.axhspan(S_LO, 0,  color='#FDECEA', alpha=0.35, zorder=0)
    ax.axhline(0, color='#AAAAAA', lw=1.8, ls=(0, (4, 3)), zorder=1)
    for yval in np.arange(-0.5, 1.6, 0.5):
        if yval == 0:
            continue
        ax.axhline(yval, color='#DDDDDD', lw=1.2, ls=(0, (4, 3)), zorder=0)

    legend_handles = []
    for mi, name in enumerate(METHODS_ORDER):
        d    = data[name]
        col  = d['color']
        mean = np.array(d['mean'], dtype=float)
        std  = np.array(d['std'],  dtype=float) if d['std'] is not None else None

        for si in range(len(SCENARIOS)):
            m = mean[si]
            if np.isnan(m):
                continue
            bx      = x[si] + offsets[mi]
            disp_m  = max(m, CLIP_LO)
            ax.bar(bx, disp_m, width=BAR_W,
                   color=col if d['sac'] else 'none',
                   edgecolor=col, linewidth=0.9,
                   hatch=d['hatch'],
                   alpha=0.75 if d['sac'] else 1.0,
                   zorder=3)
            if d['sac'] and std is not None and std[si] > 0:
                ax.errorbar(
                    bx, disp_m,
                    yerr=[[min(std[si], disp_m - S_LO)],
                          [min(std[si], S_HI - disp_m)]],
                    fmt='none', ecolor=col, elinewidth=1.1,
                    capsize=2.5, capthick=1.0, alpha=0.75, zorder=5,
                )
            if m < CLIP_LO:
                ax.text(bx, CLIP_LO - 0.01, f'{m:.2f}',
                        ha='center', va='top', fontsize=5.5,
                        color=col, rotation=90)

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
    ax.set_ylabel('Overall Score  \u2191', fontsize=11)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.tick_params(direction='in', length=3)
    ax.spines['top'].set_visible(False)
    ax.grid(False)

    ax2.set_ylim(V_BOT, V_TOP)
    ax2.set_ylabel('Safety Violation Rate  \u2191', fontsize=11)
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

    # Legend — row-major reorder for ncol=5
    _ncol  = 5
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
        ncol=5,
        fontsize=8.0,
        framealpha=0.92,
        edgecolor='#CCCCCC',
        handlelength=3.0,
        handletextpad=0.5,
        columnspacing=1.0,
        borderpad=0.6,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'fig_combo_v7.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(output_dir / 'fig_combo_v7.png', bbox_inches='tight', dpi=200)
    print(f'Saved: {output_dir}/fig_combo_v7.pdf / .png')
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Combined comparison figure (reads SAC / MPC / NMPC from JSON).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--results-dir', required=True,
        help='Root results directory containing single_scenario/ and '
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
        n_ok = sum(1 for v in data[name]['mean'] if not np.isnan(v))
        print(f'  {name}: {n_ok}/{len(SCENARIOS)} scenarios loaded')

    build_figure(data, output_dir)


if __name__ == '__main__':
    main()

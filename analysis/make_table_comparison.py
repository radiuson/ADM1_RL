"""
LaTeX Comparison Table — All Controllers × All Scenarios
=========================================================

Generates a publication-ready booktabs table summarising overall_score and
violation_rate for every (controller, scenario) pair.

Controllers covered:
  Constant, PID, Cascaded PID           — from evaluation/baselines/
  MPC, NMPC-Oracle                      — from evaluation/per_run/ (top-level fields)
  SAC-Full (13-dim), SAC-Compact (5-dim) — from evaluation/per_run/ (record wrapper)
    → in-distribution only: train_scenario == test_scenario

Output:  two complementary LaTeX tables
  table_comparison_score.tex   — overall_score  (mean ± std where n>1)
  table_comparison_viol.tex    — violation_rate (mean ± std where n>1)
  table_comparison_combined.tex — score / viol% in each cell (compact single table)

Usage:
    python analysis/make_table_comparison.py

    python analysis/make_table_comparison.py \\
        --eval-dir /path/to/results/single_scenario/evaluation \\
        --output-dir /path/to/tables
"""

import argparse
import json
import pathlib
from typing import Optional

import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

SCENARIOS = [
    'nominal', 'high_load', 'low_load',
    'shock_load', 'temperature_drop', 'cold_winter',
]
SCENARIO_LABELS = {
    'nominal':          'Nominal',
    'high_load':        'High Load',
    'low_load':         'Low Load',
    'shock_load':       'Shock Load',
    'temperature_drop': 'Temp.\ Drop',
    'cold_winter':      'Cold Winter',
}

# Controller display order and labels
CONTROLLERS = [
    ('constant',     'Constant'),
    ('pid',          'PID'),
    ('cascaded_pid', 'Cas.-PID'),
    ('mpc',          'MPC'),
    ('nmpc_oracle',  'NMPC$^*$'),
    ('sac_full',     'SAC-F'),
    ('sac_simple',   'SAC-C'),
]
CTRL_KEYS  = [c[0] for c in CONTROLLERS]
CTRL_NAMES = {c[0]: c[1] for c in CONTROLLERS}

SEEDS = [42, 123, 456]

DEFAULT_EVAL_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / 'results' / 'single_scenario' / 'evaluation'
)
DEFAULT_OUTPUT_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / 'results' / 'tables'
)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_json(path: pathlib.Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _extract(d: dict, key: str):
    """Try record wrapper first, then top-level."""
    v = d.get('record', {}).get(key)
    if v is None:
        v = d.get(key)
    return v


def _agg(vals: list) -> tuple[Optional[float], Optional[float]]:
    clean = [v for v in vals if v is not None and not np.isnan(float(v))]
    if not clean:
        return None, None
    n = len(clean)
    return float(np.mean(clean)), (float(np.std(clean, ddof=1)) if n > 1 else 0.0)


def load_all(eval_dir: pathlib.Path) -> dict:
    """
    Returns {ctrl_key: {scenario: {'score_mean', 'score_std',
                                    'viol_mean',  'viol_std', 'n'}}}
    """
    per_run   = eval_dir / 'per_run'
    baselines = eval_dir / 'baselines'
    data: dict = {k: {} for k in CTRL_KEYS}

    # ── Baseline controllers (single file per scenario) ───────────────────────
    for ctrl in ('constant', 'pid', 'cascaded_pid'):
        for sc in SCENARIOS:
            f = baselines / f'{ctrl}_on_{sc}.json'
            d = _load_json(f)
            if d is None:
                continue
            score = _extract(d, 'overall_score')
            viol  = _extract(d, 'violation_rate')
            if score is not None:
                data[ctrl][sc] = {
                    'score_mean': float(score),
                    'score_std':  0.0,
                    'viol_mean':  float(viol) if viol is not None else float('nan'),
                    'viol_std':   0.0,
                    'n': 1,
                }

    # ── MPC / NMPC (multiple seeds, top-level fields) ─────────────────────────
    for ctrl in ('mpc', 'nmpc_oracle'):
        for sc in SCENARIOS:
            scores, viols = [], []
            for seed in SEEDS:
                f = per_run / f'{ctrl}_{sc}_seed{seed}_on_{sc}.json'
                d = _load_json(f)
                if d is None:
                    continue
                s = d.get('overall_score')
                v = d.get('violation_rate')
                if s is not None:
                    scores.append(float(s))
                if v is not None:
                    viols.append(float(v))
            sm, ss = _agg(scores)
            vm, vs = _agg(viols)
            if sm is not None:
                data[ctrl][sc] = {
                    'score_mean': sm, 'score_std': ss or 0.0,
                    'viol_mean':  vm if vm is not None else float('nan'),
                    'viol_std':   vs or 0.0,
                    'n': len(scores),
                }

    # ── SAC models (in-distribution: train_scenario == test_scenario) ─────────
    for obs, suffix, ctrl_key in [
        ('full',   '',        'sac_full'),
        ('simple', '_simple', 'sac_simple'),
    ]:
        for test_sc in SCENARIOS:
            scores, viols = [], []
            # Each train scenario model evaluated on test_sc
            for train_sc in SCENARIOS:
                for seed in SEEDS:
                    fname = (f'sac_{train_sc}_safety_first'
                             f'_seed{seed}{suffix}_on_{test_sc}.json')
                    f = per_run / fname
                    d = _load_json(f)
                    if d is None:
                        continue
                    r = d.get('record', d)
                    # Only in-distribution (train == test)
                    if r.get('train_scenario', train_sc) != test_sc:
                        continue
                    s = r.get('overall_score')
                    v = r.get('violation_rate')
                    if s is not None:
                        scores.append(float(s))
                    if v is not None:
                        viols.append(float(v))
            sm, ss = _agg(scores)
            vm, vs = _agg(viols)
            if sm is not None:
                data[ctrl_key][test_sc] = {
                    'score_mean': sm, 'score_std': ss or 0.0,
                    'viol_mean':  vm if vm is not None else float('nan'),
                    'viol_std':   vs or 0.0,
                    'n': len(scores),
                }

    return data


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt(mean: float, std: float, bold: bool, fmt_str: str = '.3f') -> str:
    if mean is None or np.isnan(mean):
        return r'---'
    s = f'{mean:{fmt_str}}'
    if std > 0.001:
        s += rf' {{\tiny $\pm${std:{fmt_str}}}} '
    cell = rf'${s}$' if '.' in s else s
    return rf'\mathbf{{{cell[1:-1]}}}' if bold and cell.startswith('$') else cell


def _fmt_viol(mean: float, std: float, bold: bool) -> str:
    """Format violation rate as percentage."""
    if mean is None or np.isnan(mean):
        return r'---'
    pct = mean * 100
    s   = f'{pct:.1f}\\%'
    if std > 0.001:
        s += rf' {{\tiny $\pm${std*100:.1f}\%}}'
    return rf'\textbf{{{s}}}' if bold else s


# ── Find best controller per scenario ────────────────────────────────────────

def _best_ctrls(data: dict) -> tuple[dict, dict]:
    """Return {scenario: best_ctrl_key} for score and violation separately."""
    best_score = {}
    best_viol  = {}
    for sc in SCENARIOS:
        sc_scores = {k: data[k][sc]['score_mean']
                     for k in CTRL_KEYS if sc in data[k]}
        sc_viols  = {k: data[k][sc]['viol_mean']
                     for k in CTRL_KEYS if sc in data[k]
                     and not np.isnan(data[k][sc]['viol_mean'])}
        if sc_scores:
            best_score[sc] = max(sc_scores, key=sc_scores.get)
        if sc_viols:
            best_viol[sc]  = min(sc_viols,  key=sc_viols.get)
    return best_score, best_viol


# ── LaTeX tables ──────────────────────────────────────────────────────────────

_HEADER = """\
% Generated by analysis/make_table_comparison.py
% Requires \\usepackage{{booktabs, multirow}}
% Bold = best value in that column (per scenario).
% Baseline controllers (Constant/PID/Cas.-PID) use a single seed (no ± shown).
% SAC = in-distribution evaluation (train scenario == test scenario), n=3 seeds.
% NMPC* = oracle disturbance model (upper bound).
"""


def _col_spec(n_ctrl: int) -> str:
    return r'@{}l' + 'r' * n_ctrl + r'@{}'


def _build_table(
    data: dict,
    metric: str,          # 'score' or 'viol'
    caption: str,
    label: str,
    best_map: dict,
) -> str:
    fmt_fn = _fmt_viol if metric == 'viol' else _fmt

    lines = [
        _HEADER,
        r'\begin{table*}[htbp]',
        r'  \centering',
        rf'  \caption{{{caption}}}',
        rf'  \label{{{label}}}',
        r'  \small',
        rf'  \begin{{tabular}}{{{_col_spec(len(CONTROLLERS))}}}',
        r'    \toprule',
        # Header
        r'    \textbf{Scenario} & '
        + ' & '.join(rf'\textbf{{{CTRL_NAMES[k]}}}' for k in CTRL_KEYS)
        + r' \\',
        r'    \midrule',
    ]

    for sc in SCENARIOS:
        cells = [SCENARIO_LABELS[sc]]
        for ctrl_key in CTRL_KEYS:
            d = data.get(ctrl_key, {}).get(sc)
            if d is None:
                cells.append(r'---')
                continue
            mean = d[f'{metric}_mean']
            std  = d[f'{metric}_std']
            bold = (best_map.get(sc) == ctrl_key)
            if metric == 'viol':
                cells.append(_fmt_viol(mean, std, bold))
            else:
                cells.append(_fmt(mean, std, bold))
        lines.append('    ' + ' & '.join(cells) + r' \\')

    lines += [
        r'    \bottomrule',
        r'  \end{tabular}',
        r'\end{table*}',
    ]
    return '\n'.join(lines)


def _build_combined_table(data: dict, best_score: dict, best_viol: dict) -> str:
    """
    Combined table: each cell shows 'score / viol%'.
    Smaller font for the violation sub-value.
    """
    lines = [
        _HEADER,
        r'\begin{table*}[htbp]',
        r'  \centering',
        r'  \caption{%',
        r'    Controller comparison across all scenarios.'
        r'    Each cell: \textbf{overall score}\,/\,{\small violation rate}.'
        r'    Bold score = best in column; bold rate = lowest violation in column.'
        r'    Baselines: single seed (no $\pm$).'
        r'    SAC: in-distribution evaluation, $n=3$ seeds ($\pm$ std).'
        r'    $^*$NMPC uses oracle disturbance model (upper bound).',
        r'  }',
        r'  \label{tab:comparison_combined}',
        r'  \small',
        rf'  \begin{{tabular}}{{{_col_spec(len(CONTROLLERS))}}}',
        r'    \toprule',
        r'    \textbf{Scenario} & '
        + ' & '.join(rf'\textbf{{{CTRL_NAMES[k]}}}' for k in CTRL_KEYS)
        + r' \\',
        r'    \midrule',
    ]

    for sc in SCENARIOS:
        cells = [SCENARIO_LABELS[sc]]
        for ctrl_key in CTRL_KEYS:
            d = data.get(ctrl_key, {}).get(sc)
            if d is None:
                cells.append(r'---')
                continue

            sm, ss = d['score_mean'], d['score_std']
            vm, vs = d['viol_mean'],  d['viol_std']
            bold_s = (best_score.get(sc) == ctrl_key)
            bold_v = (best_viol.get(sc)  == ctrl_key)

            # Score part
            if np.isnan(sm):
                score_str = '---'
            else:
                s = f'{sm:.3f}'
                if ss > 0.001:
                    s += rf'{{\tiny $\pm${ss:.3f}}}'
                score_str = (rf'\mathbf{{{s}}}' if bold_s else s)

            # Violation part
            if np.isnan(vm):
                viol_str = '---'
            else:
                v = f'{vm*100:.1f}\\%'
                if vs > 0.001:
                    v += rf'{{\tiny $\pm${vs*100:.1f}\%}}'
                viol_str = (rf'\mathbf{{{v}}}' if bold_v else v)

            cell = rf'\shortstack{{{score_str}\\[-1pt]{{\small {viol_str}}}}}'
            cells.append(cell)

        lines.append('    ' + ' & '.join(cells) + r' \\')
        if sc != SCENARIOS[-1]:
            lines.append(r'    \addlinespace[2pt]')

    lines += [
        r'    \bottomrule',
        r'  \end{tabular}',
        r'\end{table*}',
    ]
    return '\n'.join(lines)


# ── Preview ───────────────────────────────────────────────────────────────────

def print_preview(data: dict) -> None:
    w = 10
    hdr = f"{'Scenario':<14}" + ''.join(f"{CTRL_NAMES[k]:>{w}}" for k in CTRL_KEYS)
    print('\n── Overall Score ──')
    print(hdr)
    print('─' * len(hdr))
    for sc in SCENARIOS:
        row = f'{SCENARIO_LABELS[sc]:<14}'
        for k in CTRL_KEYS:
            d = data.get(k, {}).get(sc)
            if d is None:
                row += f"{'---':>{w}}"
            else:
                m, s = d['score_mean'], d['score_std']
                cell = f'{m:.3f}' + (f'±{s:.3f}' if s > 0.001 else '')
                row += f'{cell:>{w}}'
        print(row)

    print('\n── Violation Rate ──')
    print(hdr)
    print('─' * len(hdr))
    for sc in SCENARIOS:
        row = f'{SCENARIO_LABELS[sc]:<14}'
        for k in CTRL_KEYS:
            d = data.get(k, {}).get(sc)
            if d is None:
                row += f"{'---':>{w}}"
            else:
                m, s = d['viol_mean'], d['viol_std']
                cell = f'{m*100:.1f}%' + (f'±{s*100:.1f}' if s > 0.001 else '')
                row += f'{cell:>{w}}'
        print(row)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate all-controller comparison LaTeX tables.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--eval-dir',   type=pathlib.Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument('--output-dir', type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    print('Loading evaluation data ...')
    data = load_all(args.eval_dir)

    # coverage report
    for k in CTRL_KEYS:
        n = len(data.get(k, {}))
        print(f'  {CTRL_NAMES[k]:<12}: {n}/{len(SCENARIOS)} scenarios loaded')

    print_preview(data)

    best_score, best_viol = _best_ctrls(data)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Score table
    tex_s = _build_table(
        data, 'score',
        caption=(r'Overall score ($\uparrow$) for all controllers across all scenarios '
                 r'(mean $\pm$ std where $n{>}1$). Bold = best per scenario.'),
        label='tab:comparison_score',
        best_map=best_score,
    )
    (args.output_dir / 'table_comparison_score.tex').write_text(tex_s)

    # Violation table
    tex_v = _build_table(
        data, 'viol',
        caption=(r'Violation rate ($\downarrow$) for all controllers across all scenarios. '
                 r'Bold = lowest violation rate per scenario.'),
        label='tab:comparison_viol',
        best_map=best_viol,
    )
    (args.output_dir / 'table_comparison_viol.tex').write_text(tex_v)

    # Combined table
    tex_c = _build_combined_table(data, best_score, best_viol)
    (args.output_dir / 'table_comparison_combined.tex').write_text(tex_c)

    print(f'\nSaved → {args.output_dir}/')
    print('  table_comparison_score.tex')
    print('  table_comparison_viol.tex')
    print('  table_comparison_combined.tex')


if __name__ == '__main__':
    main()

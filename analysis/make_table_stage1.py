"""
LaTeX Table — Stage 1 SAC Training Results
===========================================

Reads TensorBoard eval/mean_reward from all Stage 1 training runs and
generates a publication-ready LaTeX table (booktabs style) summarising:

  - Final eval reward  (last checkpoint, mean ± std over seeds)
  - Peak eval reward   (best checkpoint in the run,  mean ± std over seeds)
  - Convergence step   (first step reaching 80 % of the [min→peak] range)

Outputs:
  <output-dir>/table_stage1_training.tex  — stand-alone LaTeX snippet
  stdout                                  — human-readable preview

Usage:
    python analysis/make_table_stage1.py

    python analysis/make_table_stage1.py \\
        --training-dir /path/to/results/sac_single_scenario/training \\
        --output-dir   /path/to/tables \\
        --seeds 42 123 456
"""

import argparse
import pathlib
import warnings
from typing import Optional

import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

SCENARIOS = [
    'nominal', 'high_load', 'low_load',
    'shock_load', 'temperature_drop', 'cold_winter',
]
SCENARIO_LABELS = {
    'nominal':         'Nominal',
    'high_load':       'High Load',
    'low_load':        'Low Load',
    'shock_load':      'Shock Load',
    'temperature_drop':'Temp. Drop',
    'cold_winter':     'Cold Winter',
}

DEFAULT_SEEDS   = [42, 123, 456]
REWARD_CONFIG   = 'safety_first'
CONV_THRESHOLD  = 0.80   # fraction of [min→peak] range used for convergence detection

DEFAULT_TRAINING_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / 'results' / 'sac_single_scenario' / 'training'
)
DEFAULT_OUTPUT_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / 'results' / 'tables'
)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_eval_curve(run_dir: pathlib.Path):
    """Return (steps, rewards) arrays or (None, None)."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        raise ImportError("pip install tensorboard")

    tb_root  = run_dir / 'tensorboard'
    sac_dirs = sorted(tb_root.glob('SAC_*')) if tb_root.exists() else []
    if not sac_dirs:
        return None, None

    ea = EventAccumulator(str(sac_dirs[-1]))
    ea.Reload()
    if 'eval/mean_reward' not in ea.Tags().get('scalars', []):
        return None, None

    ev    = ea.Scalars('eval/mean_reward')
    steps = np.array([e.step  for e in ev], dtype=float)
    vals  = np.array([e.value for e in ev], dtype=float)
    return steps, vals


def _convergence_step(steps: np.ndarray, vals: np.ndarray,
                      threshold: float = CONV_THRESHOLD) -> Optional[float]:
    """Return the first step where val >= min + threshold*(peak - min)."""
    lo, hi = vals.min(), vals.max()
    if hi <= lo:
        return None
    target = lo + threshold * (hi - lo)
    idx = next((i for i, v in enumerate(vals) if v >= target), None)
    return float(steps[idx]) if idx is not None else None


def load_all_stats(
    training_dir: pathlib.Path,
    seeds: list[int],
) -> dict:
    """
    Returns {scenario: {obs_mode: {'final_mean', 'final_std',
                                   'peak_mean',  'peak_std',
                                   'conv_mean',  'n'}}}
    """
    data = {}
    for sc in SCENARIOS:
        data[sc] = {}
        for obs in ('full', 'simple'):
            suffix  = '_simple' if obs == 'simple' else ''
            finals, peaks, convs = [], [], []

            for seed in seeds:
                run_name = f'sac_{sc}_{REWARD_CONFIG}_seed{seed}{suffix}'
                steps, vals = _load_eval_curve(training_dir / run_name)
                if vals is None:
                    warnings.warn(f'Missing: {run_name}', stacklevel=2)
                    continue
                finals.append(float(vals[-1]))
                peaks.append(float(vals.max()))
                cs = _convergence_step(steps, vals)
                if cs is not None:
                    convs.append(cs / 1_000)   # → k steps

            n = len(finals)
            if n == 0:
                continue
            ddof = 1 if n > 1 else 0
            data[sc][obs] = {
                'final_mean': float(np.mean(finals)),
                'final_std':  float(np.std(finals, ddof=ddof)),
                'peak_mean':  float(np.mean(peaks)),
                'peak_std':   float(np.std(peaks,  ddof=ddof)),
                'conv_mean':  float(np.mean(convs)) if convs else float('nan'),
                'n': n,
            }
    return data


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_reward(mean: float, std: float, bold: bool = False) -> str:
    """Format 'mean ± std' for a reward value; bold if requested."""
    if np.isnan(mean):
        return r'\multicolumn{1}{c}{---}'
    s = rf'${mean:+.0f} \pm {std:.0f}$'
    return rf'\textbf{{{s}}}' if bold else s


def _fmt_conv(conv_k: float) -> str:
    if np.isnan(conv_k):
        return r'---'
    return rf'$\approx {conv_k:.0f}$'


# ── LaTeX builder ─────────────────────────────────────────────────────────────

PREAMBLE = r"""% ── Stage 1 SAC Training Results ─────────────────────────────────────────────
% Generated by analysis/make_table_stage1.py
% Requires: \usepackage{booktabs, multirow, siunitx}  (or plain tabular)
%
% Columns:
%   Scenario | Full obs — final ↑ | Full obs — peak ↑
%           | Compact obs — final ↑ | Compact obs — peak ↑
%           | Conv. (k steps)
%
% Bold = better of {full, simple} for that metric.
% Conv. = first step reaching 80 % of [min → peak] range, averaged over seeds.
% High Load never converges to a positive reward (marked †).
"""


def build_latex(data: dict, seeds: list[int]) -> str:
    n_seeds = len(seeds)

    col_spec = r'@{}l' + r'rr' * 2 + r'r@{}'   # scenario + 4 reward cols + conv

    lines = [
        PREAMBLE,
        r'\begin{table}[htbp]',
        r'  \centering',
        r'  \caption{%',
        r'    SAC training results — Stage~1 (safety-first reward, '
        rf'    $n={n_seeds}$ seeds per cell, 300\,k training steps).%',
        r'    Final reward: last evaluation checkpoint.'
        r'    Peak reward: best checkpoint during training.'
        r'    Conv.\,step: first step reaching 80\,\% of the min$\to$peak range.',
        r'    Bold: better observation mode per metric.'
        r'    $^\dagger$ High Load never achieves a positive reward.',
        r'  }',
        r'  \label{tab:stage1_training}',
        rf'  \begin{{tabular}}{{{col_spec}}}',
        r'    \toprule',
        # Header row 1
        r'    & \multicolumn{2}{c}{\textbf{Full obs (13-dim)}}'
        r'    & \multicolumn{2}{c}{\textbf{Compact obs (5-dim)}}'
        r'    & \\',
        r'    \cmidrule(lr){2-3} \cmidrule(lr){4-5}',
        # Header row 2
        r'    \textbf{Scenario}'
        r'    & \textbf{Final} $\uparrow$'
        r'    & \textbf{Peak} $\uparrow$'
        r'    & \textbf{Final} $\uparrow$'
        r'    & \textbf{Peak} $\uparrow$'
        r'    & \textbf{Conv.\ (k)} $\downarrow$ \\',
        r'    \midrule',
    ]

    for sc in SCENARIOS:
        label = SCENARIO_LABELS[sc]
        df    = data[sc].get('full',   {})
        ds    = data[sc].get('simple', {})

        # Determine bold (higher final / higher peak per obs mode)
        bold_full_final  = (df.get('final_mean', -1e9) >= ds.get('final_mean', -1e9))
        bold_simp_final  = not bold_full_final
        bold_full_peak   = (df.get('peak_mean',  -1e9) >= ds.get('peak_mean',  -1e9))
        bold_simp_peak   = not bold_full_peak

        ff = _fmt_reward(df.get('final_mean', float('nan')),
                         df.get('final_std',  0.0), bold=bold_full_final)
        fp = _fmt_reward(df.get('peak_mean',  float('nan')),
                         df.get('peak_std',   0.0), bold=bold_full_peak)
        sf = _fmt_reward(ds.get('final_mean', float('nan')),
                         ds.get('final_std',  0.0), bold=bold_simp_final)
        sp = _fmt_reward(ds.get('peak_mean',  float('nan')),
                         ds.get('peak_std',   0.0), bold=bold_simp_peak)

        # Convergence: show full / simple; if high_load use dagger
        c_full = _fmt_conv(df.get('conv_mean', float('nan')))
        c_simp = _fmt_conv(ds.get('conv_mean', float('nan')))
        if c_full == c_simp:
            conv_str = c_full
        else:
            conv_str = rf'{c_full} / {c_simp}'

        dagger = r'$^\dagger$' if sc == 'high_load' else ''
        row = rf'    {label}{dagger} & {ff} & {fp} & {sf} & {sp} & {conv_str} \\'
        lines.append(row)

    lines += [
        r'    \bottomrule',
        r'  \end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


# ── Plain-text preview ────────────────────────────────────────────────────────

def print_preview(data: dict, seeds: list[int]) -> None:
    header = (
        f"{'Scenario':<18}"
        f"{'Full final':>16}{'Full peak':>14}"
        f"{'Simp final':>16}{'Simp peak':>14}"
        f"{'Conv (full/simp)':>18}"
    )
    print()
    print(header)
    print('-' * len(header))
    for sc in SCENARIOS:
        df = data[sc].get('full',   {})
        ds = data[sc].get('simple', {})

        def _r(d, k_m, k_s):
            m = d.get(k_m, float('nan'))
            s = d.get(k_s, 0.0)
            return '---' if np.isnan(m) else f'{m:+.0f}±{s:.0f}'

        cf = f"{df.get('conv_mean', float('nan')):.0f}k" if not np.isnan(df.get('conv_mean', float('nan'))) else '---'
        cs = f"{ds.get('conv_mean', float('nan')):.0f}k" if not np.isnan(ds.get('conv_mean', float('nan'))) else '---'

        dagger = '†' if sc == 'high_load' else ''
        print(
            f"{SCENARIO_LABELS[sc]+dagger:<18}"
            f"{_r(df,'final_mean','final_std'):>16}"
            f"{_r(df,'peak_mean','peak_std'):>14}"
            f"{_r(ds,'final_mean','final_std'):>16}"
            f"{_r(ds,'peak_mean','peak_std'):>14}"
            f"{cf+'/'+cs:>18}"
        )
    print(f'\n(n={len(seeds)} seeds: {seeds})')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate LaTeX training-results table (Stage 1).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--training-dir', type=pathlib.Path,
                        default=DEFAULT_TRAINING_DIR)
    parser.add_argument('--output-dir',   type=pathlib.Path,
                        default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    args = parser.parse_args()

    print('Loading Stage 1 training data ...')
    data = load_all_stats(args.training_dir, args.seeds)

    print_preview(data, args.seeds)

    latex = build_latex(data, args.seeds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / 'table_stage1_training.tex'
    out_path.write_text(latex, encoding='utf-8')
    print(f'\nLaTeX saved → {out_path}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""The hyperparameter sweep and the class comparison it supports, as a table.

The sweep was drawn as a heat map and the class comparison as intervals
beside it, but every claim the text makes about them is a count or a value:
how many cells clear the envelope, how many improve on the reported setting,
how far apart the class medians sit.  A reader checking those against a
colour scale has to judge each cell by eye, so the numbers are tabulated
instead.

The class rows use the stratified scheme of ``stratified_stats`` --- families
within a class, then weights within a family, then seeds within a weight ---
which is the scheme Section~6 describes and the one the reported intervals
come from.  Resampling families alone, as the figure did, gives a different
and narrower statistic that does not match the text.

    python3 scripts/build_ablation_table.py [outdir]

Writes ``table_ablation.tex``; ``outdir`` defaults to the manuscript.
"""
from __future__ import annotations

import os
import random
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stratified_stats as SS
from plot_ablations import FAMS, LABEL, OFF, ON, SETTINGS, load_main, load_sweep

# The manuscript carries this table inline, so writing it beside paper.tex
# would leave a file nothing reads.  Run scripts/sync_tables.py to put a
# regenerated table into paper.tex; a path given here is used as-is.
PAPER = '.'

# The seed the reported intervals were drawn with.
SEED = 20260919
B = 4000

# Column headings, in the order the sweep varies them.
COLS = [('Reported', None),
        (r'$10^{-4}$', 'LR_1e-4'), (r'$10^{-3}$', 'LR_1e-3'),
        ('64,\\,64', 'ARCH_64_64'), ('400,\\,300', 'ARCH_400_300')]


def grid():
    """Median difference from the envelope, one cell per family and setting."""
    base, sweep = load_main(), load_sweep()
    M = np.full((len(FAMS), len(COLS)), np.nan)
    for i, fam in enumerate(FAMS):
        if base.get(fam):
            M[i, 0] = st.median(base[fam])
        for j, (_, key) in enumerate(COLS):
            if key and sweep[key].get(fam):
                M[i, j] = st.median(sweep[key][fam])
    return M


def class_gap(which, rng):
    """Class difference and interval under the scheme Section 6 describes."""
    G, _, _ = SS.load(which)
    on = [a for a in SS.REWARD_ON if G.get(a)]
    off = [a for a in SS.REWARD_OFF if G.get(a)]
    point = (SS.class_median(G, on, rng, False)
             - SS.class_median(G, off, rng, False))
    draws = sorted(SS.class_median(G, on, rng, True)
                   - SS.class_median(G, off, rng, True) for _ in range(B))
    return point, draws[int(.025 * B)], draws[int(.975 * B)]


def sweep_gap(M, j):
    """The class difference at one swept setting, from that column's cells.

    A swept cell is three seeds at one weight, so the stratified scheme has
    no weight level to resample and would not be the same estimator as the
    reported column.  The difference of the two class medians over the
    column's cells is what the text states, and is reported without an
    interval rather than with one the design does not support.
    """
    on = [M[i, j] for i, f in enumerate(FAMS) if f in ON and not np.isnan(M[i, j])]
    off = [M[i, j] for i, f in enumerate(FAMS) if f in OFF and not np.isnan(M[i, j])]
    if not on or not off:
        return None
    return st.median(on) - st.median(off)


def cell(v, reported):
    """A cell, marked where it clears the envelope."""
    if np.isnan(v):
        return '---'
    s = f'{v:+.0f}'
    return f'\\textbf{{{s}}}' if v > 0 else s


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else PAPER
    M = grid()
    rng = random.Random(SEED)

    L = [r'\begin{table*}[t]', r'\centering',
         r'\caption{Hyperparameter sweep and the class comparison it supports,'
         ' as the median difference from the conventional envelope'
         ' (m$^3$/d). Each swept cell is three seeds; the reported column is'
         ' the four weights and ten seeds of Table~\\ref{tab:main}. Bold'
         ' marks a cell above the envelope: ten of the sixteen on-policy'
         ' swept cells, against one of the twenty off-policy cells. ARS has'
         ' neither a gradient step size nor a value network and so shares'
         ' none of the swept settings. The last two rows are the'
         ' difference of the two class medians; the reported column'
         ' carries the interval of Section~\\ref{sec:results}, resampled by'
         ' families then weights then seeds. A swept cell is one weight, so'
         ' that scheme has no weight level there and no interval is given.}',
         r'  \label{tab:ablation}',
         r'\begin{tabular}{l r rr rr}', r'\toprule',
         r' & & \multicolumn{2}{c}{Learning rate}'
         r' & \multicolumn{2}{c}{Network} \\',
         r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}',
         ' & ' + ' & '.join(h for h, _ in COLS) + r' \\',
         r'\midrule']

    for i, fam in enumerate(FAMS):
        if fam in OFF and FAMS[i - 1] in ON:
            L.append(r'\midrule')
        row = ' & '.join(cell(M[i, j], M[i, 0]) for j in range(len(COLS)))
        L.append(f'{LABEL[fam]} & {row} \\\\')

    # The class difference: the reported setting and each swept one, under
    # the scheme the text uses.
    L.append(r'\midrule')
    rep = class_gap('reward', rng)
    sweeps = [sweep_gap(M, j) for j in range(1, len(COLS))]
    pts = ' & '.join([f'{rep[0]:+.0f}']
                     + [f'{g:+.0f}' if g is not None else '---'
                        for g in sweeps])
    L.append(r'\addlinespace[2pt]')
    L.append(f'On-policy $-$ off-policy & {pts} \\\\')
    L.append(f'\\quad 95\\,\\% interval & \\scriptsize '
             f'$[{rep[1]:+.0f},{rep[2]:+.0f}]$ & '
             + ' & '.join([''] * len(sweeps)) + r' \\')

    L += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']

    p = os.path.join(out, 'table_ablation.tex')
    open(p, 'w').write('\n'.join(L) + '\n')

    on_pos = sum(1 for i, f in enumerate(FAMS) if f in ON
                 for j in range(1, len(COLS)) if M[i, j] > 0)
    off_pos = sum(1 for i, f in enumerate(FAMS) if f in OFF
                  for j in range(1, len(COLS)) if M[i, j] > 0)
    print(f'table_ablation.tex  {len(FAMS)} families x {len(COLS)} settings; '
          f'{on_pos} on-policy and {off_pos} off-policy swept cells above '
          f'the envelope')
    print(f'  reported     {rep[0]:+5.0f}  [{rep[1]:+.0f}, {rep[2]:+.0f}]')
    for (h, _), g in zip(COLS[1:], sweeps):
        if g is not None:
            print(f'  {h:<12} {g:+5.0f}')


if __name__ == '__main__':
    main()

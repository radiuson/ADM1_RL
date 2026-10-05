#!/usr/bin/env python3
"""The frontier drawn as the difference from the envelope.

The published version plots absolute methane against violation rate, so the
conventional envelope runs through the middle of the cloud and whether a
marker sits above or below it -- the one claim the figure makes -- has to be
judged by eye against a curve.  Here the envelope is subtracted, so it
becomes the zero line and the claim reads off the sign.

Panel (b) is then free to carry something the magnified copy could not: the
class difference within each violation band, which is the comparison with
the operating point held fixed.

    python3 scripts/plot_frontier_delta.py [outdir]

Writes fig_frontier_delta.pdf beside the other figures.
"""
from __future__ import annotations

import collections
import os
import statistics as st
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from build_tables import envelope
from plot_figures import (C_ENV, C_GRID, C_OFF, C_ON, OFF_POLICY, ON_POLICY,
                          load_reward)

PAPER = os.path.expanduser('~/code/biogas/ADM1/papers/mypaper/img')

# The bands Section 9.2 compares the classes within, with the operating
# point held fixed.
BANDS = [(5, 10), (10, 20), (20, 35)]

LABEL = {'ppo': 'PPO', 'a2c': 'A2C', 'trpo': 'TRPO',
         'recurrentppo': 'RecurrentPPO', 'sac': 'SAC', 'tqc': 'TQC',
         'ddpg': 'DDPG', 'td3': 'TD3', 'crossq': 'CrossQ'}

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})


def deltas(R, E):
    """Each run's (violation rate, difference from the envelope)."""
    ex = [p[0] for p in E]
    ey = [p[1] for p in E]
    out = collections.defaultdict(list)
    for k in ON_POLICY + OFF_POLICY:
        for r in R.get(k, []):
            if ex[0] <= r['viol'] <= ex[-1]:
                out[k].append((r['viol'],
                               r['ch4'] - float(np.interp(r['viol'], ex, ey))))
    return out


def panel_runs(ax, D):
    """Every run against the envelope, which is now the zero line."""
    # Shading the half-planes says which side is which before any marker is
    # read; it is the whole claim of the figure.
    ax.axhspan(0, 1e4, color=C_ON, alpha=0.05, lw=0, zorder=0)
    ax.axhspan(-1e4, 0, color=C_OFF, alpha=0.05, lw=0, zorder=0)
    ax.axhline(0, color=C_ENV, lw=1.2, zorder=4)

    for keys, colour, marker, name in ((ON_POLICY, C_ON, 'o', 'On-policy'),
                                       (OFF_POLICY, C_OFF, '^', 'Off-policy')):
        xs = [v for k in keys for v, _ in D.get(k, [])]
        ys = [d for k in keys for _, d in D.get(k, [])]
        above = sum(1 for d in ys if d > 0)
        ax.scatter(xs, ys, s=11, c=colour, marker=marker, alpha=0.40,
                   linewidths=0, zorder=2,
                   label=f'{name}: {above} of {len(ys)} above')

    # One outlined marker per family at its median, as in the published
    # version: the claim is about families, not single runs.
    placed = []
    for keys, colour, marker in ((ON_POLICY, C_ON, 'o'),
                                 (OFF_POLICY, C_OFF, '^')):
        for k in keys:
            if not D.get(k):
                continue
            mv = st.median([v for v, _ in D[k]])
            md = st.median([d for _, d in D[k]])
            ax.plot([mv], [md], marker=marker, ms=8.5, mfc=colour,
                    mec='white', mew=1.3, ls='', zorder=5)
            placed.append((md, mv, k, colour))

    # Five of the nine medians sit within 30 m3/d of each other, so the
    # labels are walked apart in order and tied back with a leader.
    gap = 17.0
    taken = []
    for md, mv, k, colour in sorted(placed):
        ty = md if not taken else max(md, taken[-1] + gap)
        taken.append(ty)
        ax.annotate(LABEL[k], (mv, md), xytext=(mv + 1.6, ty),
                    textcoords='data', fontsize=6, color=colour,
                    va='center', zorder=6,
                    arrowprops=dict(arrowstyle='-', color=colour,
                                    lw=0.4, alpha=0.5,
                                    shrinkA=0, shrinkB=4))

    ax.set_xlabel('Steps above the soft limit (%)')
    ax.set_ylabel('Difference from the conventional envelope (m$^3$/d)')
    ax.set_title('(a) every run, scored against the envelope', loc='left')
    ax.set_xlim(-1, 50)
    # One A2C run at -265 would otherwise set the scale and flatten the
    # rest; the axis holds the body of the data and the legend counts
    # every run, including the ones off the bottom.
    ax.set_ylim(-200, 130)
    ax.grid(axis='y', color=C_GRID, lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='lower left', frameon=False,
              title='above the line = more methane than\n'
                    'any conventional tuning at that rate',
              title_fontsize=6.8, alignment='left')


def panel_bands(ax, D):
    """The class difference within each band, the operating point fixed."""
    xs = np.arange(len(BANDS))
    on_m, off_m, gaps = [], [], []
    for lo, hi in BANDS:
        o = [d for k in ON_POLICY for v, d in D.get(k, []) if lo <= v < hi]
        f = [d for k in OFF_POLICY for v, d in D.get(k, []) if lo <= v < hi]
        on_m.append(st.median(o))
        off_m.append(st.median(f))
        gaps.append(st.median(o) - st.median(f))

    ax.axhline(0, color=C_ENV, lw=1.2, zorder=3)
    w = 0.33
    ax.bar(xs - w / 2, on_m, w, color=C_ON, zorder=2, label='On-policy')
    ax.bar(xs + w / 2, off_m, w, color=C_OFF, zorder=2, label='Off-policy')
    # Each gap sits just under the zero line, clear of both bars and of
    # the title, so all three are placed alike.
    for x, g in zip(xs, gaps):
        ax.annotate(f'{g:+.0f}', (x, -6), ha='center', va='top',
                    fontsize=6.8, color=C_ENV, zorder=4)

    ax.set_xticks(xs)
    ax.set_xticklabels([f'{lo}–{hi}' for lo, hi in BANDS])
    ax.set_xlabel('Violation band (%)')
    ax.set_ylabel('Difference from the envelope (m$^3$/d)')
    ax.set_title('(b) class medians within a band', loc='left')
    ax.grid(axis='y', color=C_GRID, lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='lower left', frameon=False, labelspacing=0.3)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else PAPER
    os.makedirs(out, exist_ok=True)
    B, E = envelope()
    D = deltas(load_reward(), E)

    fig = plt.figure(figsize=(7.2, 3.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.0], wspace=0.28)
    panel_runs(fig.add_subplot(gs[0]), D)
    panel_bands(fig.add_subplot(gs[1]), D)

    p = os.path.join(out, 'fig_frontier_delta.pdf')
    fig.savefig(p)
    plt.close(fig)

    n = sum(len(v) for v in D.values())
    print(f'fig_frontier_delta.pdf  {n} runs scored, '
          f'{len(E)} envelope points, {len(BANDS)} bands')


if __name__ == '__main__':
    main()

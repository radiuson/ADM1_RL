#!/usr/bin/env python3
"""The penalty weight against the algorithm, on one response.

Both design choices are referred to the same quantity -- the difference from
the conventional envelope at matched violation rate -- so the figure shows
what each of them moves: the lines are how far a weight sweep carries one
family, the vertical spread between them is how far the choice of family does.

    python3 scripts/plot_weight.py [outdir]

``outdir`` defaults to the manuscript directory.
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

from effect_sizes import ON, OFF, load

PAPER = os.path.expanduser('~/code/biogas/ADM1/papers/mypaper')

WEIGHTS = ('lw0p5', 'lw1', 'lw2', 'lw5')
WLAB = ('0.5', '1', '2', '5')
LABEL = {'ppo': 'PPO', 'a2c': 'A2C', 'trpo': 'TRPO',
         'recurrentppo': 'RecurrentPPO', 'ars': 'ARS', 'sac': 'SAC',
         'tqc': 'TQC', 'ddpg': 'DDPG', 'td3': 'TD3', 'crossq': 'CrossQ'}

C_ON, C_OFF, C_FREE = '#0072B2', '#D55E00', '#009E73'
C_GRID, C_RULE = '#BBBBBB', '#333333'

# The four components of the decomposition, in the order they are reported.
PARTS = [('Algorithm family', 25.5, '#0072B2'),
         ('Penalty weight', 0.2, '#D55E00'),
         ('Interaction', 22.0, '#9467BD'),
         ('Between seeds', 52.3, '#BBBBBB')]

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})


def colour(fam):
    return C_ON if fam in ON else (C_OFF if fam in OFF else C_FREE)


def panel_slopes(ax, med):
    """One line per family across the swept weights."""
    x = np.arange(len(WEIGHTS))
    ends = []
    for fam, ms in med.items():
        y = [ms.get(w) for w in WEIGHTS]
        if any(v is None for v in y):
            continue
        c = colour(fam)
        ax.plot(x, y, '-o', color=c, ms=3, lw=1.1, alpha=0.85, zorder=2)
        # Label at whichever end leaves the label clear of the other lines.
        ends.append((y[-1], LABEL[fam], c))
    # Labels at the right edge collide where families finish close together,
    # so they are nudged apart while keeping their order.
    ends.sort()
    gap = 13.0
    placed = []
    for yv, lab, c in ends:
        pos = yv if not placed else max(yv, placed[-1] + gap)
        placed.append(pos)
        ax.annotate(lab, (x[-1], yv), xytext=(x[-1] + 0.12, pos),
                    textcoords='data', va='center', fontsize=6.5, color=c)
    ax.axhline(0, color=C_RULE, lw=1.0, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(WLAB)
    ax.set_xlim(-0.15, len(WEIGHTS) - 1 + 0.9)
    ax.set_xlabel('VFA penalty weight')
    ax.set_ylabel('Difference from the envelope (m$^3$/d)')
    ax.set_title('(a) what the weight moves within a family', loc='left')
    ax.grid(axis='y', color=C_GRID, lw=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    h = [plt.Line2D([], [], color=c, lw=1.1, marker='o', ms=3)
         for c in (C_ON, C_OFF, C_FREE)]
    ax.legend(h, ['on-policy', 'off-policy', 'gradient-free'],
              loc='lower left', frameon=False, handletextpad=0.4,
              labelspacing=0.3)


def panel_variance(ax):
    """The decomposition as one bar, so the interaction is not hidden."""
    left = 0.0
    for name, pct, col in PARTS:
        ax.barh(0, pct, left=left, height=0.5, color=col,
                edgecolor='white', lw=0.8)
        # Only the wide segments can hold a label inside them.
        if pct >= 8:
            ax.text(left + pct / 2, 0, f'{pct:.1f}%', ha='center',
                    va='center', fontsize=7,
                    color='white' if col != '#BBBBBB' else 'black')
        left += pct
    # The weight's share is too thin to label in place.
    ax.annotate('0.2%', xy=(25.5 + 0.1, 0.26), xytext=(27, 0.62),
                fontsize=7, color=C_OFF,
                arrowprops=dict(arrowstyle='-', color=C_OFF, lw=0.7))
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.45, 0.85)
    ax.set_yticks([])
    ax.set_xlabel('Share of the variance in that response (%)')
    ax.set_title('(b) what each choice accounts for', loc='left')
    ax.spines['left'].set_visible(False)
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in PARTS]
    ax.legend(h, [n for n, _, _ in PARTS], loc='upper left',
              bbox_to_anchor=(0, -0.30), ncol=2, frameon=False,
              handlelength=1.1, handletextpad=0.5, columnspacing=1.4)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else PAPER
    os.makedirs(out, exist_ok=True)

    rows = load()
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    by_family = collections.defaultdict(list)
    for a, w, x in rows:
        by[a][w].append(x)
        by_family[a].append(x)
    med = {a: {w: st.median(v) for w, v in ws.items()} for a, ws in by.items()}

    fig = plt.figure(figsize=(7.2, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.22)
    panel_slopes(fig.add_subplot(gs[0]), med)
    panel_variance(fig.add_subplot(gs[1]))

    p = os.path.join(out, 'fig_weight.pdf')
    fig.savefig(p)
    plt.close(fig)

    spans = {a: max(m.values()) - min(m.values()) for a, m in med.items()
             if len(m) == len(WEIGHTS)}
    # The family median is taken over that family's runs, as the text does,
    # not over the four weight medians -- the two differ where a family's
    # weights are unevenly populated.
    fam = [st.median(v) for v in by_family.values()]
    print(f'fig_weight.pdf  {len(med)} families; family medians span '
          f'{max(fam) - min(fam):.0f} m3/d, weight spans '
          f'{min(spans.values()):.0f}-{max(spans.values()):.0f}')


if __name__ == '__main__':
    main()

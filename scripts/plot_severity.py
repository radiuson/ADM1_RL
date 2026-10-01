#!/usr/bin/env python3
"""What a pooled violation rate leaves out, in three panels.

A rate counts the steps above the soft limit and says nothing about which
scenario they fall in, how far past the limit they go, or how long they last.
The figure shows each of those alongside the rate it would otherwise be read
from.

    python3 scripts/plot_severity.py [outdir]

``outdir`` defaults to the manuscript directory.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import statistics as st
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from build_tables import CORRECTED, DATA, WEIGHTS

# The manuscript keeps its figures in img/; a path given on the command
# line is used as-is.
PAPER = os.path.expanduser('~/code/biogas/ADM1/papers/mypaper/img')

ON = ('ppo', 'a2c', 'trpo', 'recurrentppo')
OFF = ('sac', 'tqc', 'ddpg', 'td3', 'crossq')
FREE = ('ars',)
ORDER = ('trpo', 'ppo', 'a2c', 'recurrentppo', 'ars',
         'sac', 'tqc', 'ddpg', 'td3', 'crossq')
LABEL = {'ppo': 'PPO', 'a2c': 'A2C', 'trpo': 'TRPO',
         'recurrentppo': 'RecurrentPPO', 'ars': 'ARS', 'sac': 'SAC',
         'tqc': 'TQC', 'ddpg': 'DDPG', 'td3': 'TD3', 'crossq': 'CrossQ'}

C_ON, C_OFF, C_FREE = '#0072B2', '#D55E00', '#009E73'
C_GRID, C_RULE = '#BBBBBB', '#333333'
HARD_MG = 1500.0
SOFT_MG = 300.0

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})


def colour(fam):
    return C_ON if fam in ON else (C_OFF if fam in OFF else C_FREE)


def load():
    """Reward-penalty runs under the main design, one per (weight, seed)."""
    R = collections.defaultdict(list)
    for f in glob.glob(f'{DATA}/evres/*.json'):
        d = json.load(open(f))
        if d.get('steps') != 150000 or '_h5_' in d.get('model', ''):
            continue
        if d.get('w') not in WEIGHTS:
            continue
        a = (d.get('algo') or 'sac').lower()
        want = CORRECTED.get(a)
        if want and not d.get('model', '').startswith(want):
            continue
        R[a].append(d)
    for a in list(R):
        seen = {}
        for d in sorted(R[a], key=lambda x: x.get('model', '')):
            seen.setdefault((d.get('w'), d.get('seed')), d)
        R[a] = list(seen.values())
    return R


def panel_concentration(ax, R):
    """Pooled, reachable-only and worst-scenario rate for each family."""
    y = np.arange(len(ORDER))[::-1]
    for yi, fam in zip(y, ORDER):
        v = R[fam]
        pooled = st.median([x['viol'] for x in v])
        reach = st.median([x['reachable_viol'] for x in v])
        worst = st.median([x['worst_viol'] for x in v])
        c = colour(fam)
        ax.plot([pooled, worst], [yi, yi], color=c, lw=1.0, alpha=0.45,
                zorder=1, solid_capstyle='butt')
        ax.plot([pooled], [yi], 'o', color=c, ms=4.5, zorder=3)
        ax.plot([reach], [yi], 's', color=c, ms=3.6, mfc='white', mew=1.0,
                zorder=3)
        ax.plot([worst], [yi], 'D', color=c, ms=3.6, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[f] for f in ORDER])
    ax.set_xlabel('Steps above the soft limit (%)')
    ax.set_title('(a) where the excursions fall', loc='left')
    ax.grid(axis='x', color=C_GRID, lw=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    ax.margins(y=0.05)
    h = [plt.Line2D([], [], color=C_RULE, marker='o', ls='', ms=4.5),
         plt.Line2D([], [], color=C_RULE, marker='s', ls='', ms=3.6,
                    mfc='white', mew=1.0),
         plt.Line2D([], [], color=C_RULE, marker='D', ls='', ms=3.6)]
    ax.legend(h, ['pooled', 'reachable only', 'worst scenario'],
              loc='upper left', bbox_to_anchor=(0, -0.22), ncol=3,
              frameon=False, handletextpad=0.3, columnspacing=1.2)


def panel_hard(ax, R):
    """Runs that cross the hard limit, out of forty."""
    y = np.arange(len(ORDER))[::-1]
    for yi, fam in zip(y, ORDER):
        v = R[fam]
        n = sum(1 for x in v if x.get('viol_hard', 0) > 0)
        ax.barh(yi, n, color=colour(fam), height=0.6,
                alpha=1.0 if n else 0.25)
        if n:
            ax.text(n + 0.6, yi, f'{n}/{len(v)}', va='center', fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel('Runs past 1500 mg/L')
    ax.set_title('(b) hard-limit breaches', loc='left')
    ax.set_xlim(0, 30)
    ax.grid(axis='x', color=C_GRID, lw=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    ax.margins(y=0.05)


def panel_severity(ax, R):
    """Duration against depth, one point per run that leaves the band."""
    for fam in ORDER:
        v = [x for x in R[fam] if x['longest_run'] > 0]
        if not v:
            continue
        ax.scatter([x['longest_run'] for x in v],
                   [x['excess_max'] for x in v],
                   s=10, c=colour(fam), alpha=0.55, linewidths=0, zorder=2)
    # The hard limit measured from the soft limit, which is what excess_max is.
    ax.axhline(HARD_MG - SOFT_MG, color=C_RULE, lw=1.0, ls='--', zorder=3)
    ax.text(0.02, HARD_MG - SOFT_MG, 'hard limit', va='bottom', ha='left',
            fontsize=7, transform=ax.get_yaxis_transform())
    ax.set_xlabel('Longest unbroken excursion')
    ax.set_ylabel('Peak excess over the soft limit (mg/L)')
    ax.set_title('(c) how deep and how long', loc='left')
    ax.grid(color=C_GRID, lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    h = [plt.Line2D([], [], color=c, marker='o', ls='', ms=4)
         for c in (C_ON, C_OFF, C_FREE)]
    ax.legend(h, ['on-policy', 'off-policy', 'gradient-free'],
              loc='upper left', frameon=False, handletextpad=0.3,
              labelspacing=0.3)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else PAPER
    os.makedirs(out, exist_ok=True)
    R = load()

    fig = plt.figure(figsize=(7.4, 3.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.55, 1.2], wspace=0.42)
    panel_concentration(fig.add_subplot(gs[0]), R)
    panel_hard(fig.add_subplot(gs[1]), R)
    panel_severity(fig.add_subplot(gs[2]), R)

    p = os.path.join(out, 'fig_severity.pdf')
    fig.savefig(p)
    plt.close(fig)

    breach = {f: sum(1 for x in R[f] if x.get('viol_hard', 0) > 0)
              for f in ORDER}
    named = ', '.join(f'{LABEL[f]} {n}/40' for f, n in breach.items() if n)
    print(f'fig_severity.pdf  {sum(len(v) for v in R.values())} runs; '
          f'hard-limit breaches: {named or "none"}')


if __name__ == '__main__':
    main()

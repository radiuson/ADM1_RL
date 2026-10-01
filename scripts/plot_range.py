#!/usr/bin/env python3
"""What widening the conventional actuator range would change.

The learned policies are clipped to a feed-strength multiplier of 1.3, so the
reported envelope holds the conventional controllers to the same bound. This
draws the alternative -- the envelope the wider sweep to 1.7 would give -- and
the difference it makes to every family median.

    python3 scripts/plot_range.py [outdir]

``outdir`` defaults to the manuscript's img directory.
"""
from __future__ import annotations

import collections
import glob
import importlib
import json
import os
import statistics as st
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import build_tables as BT
from build_tables import CORRECTED, DATA, WEIGHTS

PAPER = os.path.expanduser('~/code/biogas/ADM1/papers/mypaper/img')

ON = ('ppo', 'a2c', 'trpo', 'recurrentppo')
OFF = ('sac', 'tqc', 'ddpg', 'td3', 'crossq')
ORDER = ('trpo', 'ppo', 'ars', 'a2c', 'recurrentppo',
         'sac', 'td3', 'tqc', 'ddpg', 'crossq')
LABEL = {'ppo': 'PPO', 'a2c': 'A2C', 'trpo': 'TRPO',
         'recurrentppo': 'RecurrentPPO', 'ars': 'ARS', 'sac': 'SAC',
         'tqc': 'TQC', 'ddpg': 'DDPG', 'td3': 'TD3', 'crossq': 'CrossQ'}

C_COMMON, C_WIDE = '#333333', '#D55E00'
C_ON, C_OFF, C_FREE = '#0072B2', '#D55E00', '#009E73'
C_GRID = '#BBBBBB'

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})


def both_envelopes():
    """The reported envelope and the one the wider sweep would give."""
    common = BT.envelope()
    os.environ['ADM1_FULL_RANGE'] = '1'
    importlib.reload(BT)
    wide = BT.envelope()
    del os.environ['ADM1_FULL_RANGE']
    importlib.reload(BT)
    return common, wide


def load_runs():
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


def med_against(runs, E):
    xs = [p[0] for p in E]
    ys = [p[1] for p in E]
    v = [r['ch4'] - float(np.interp(r['viol'], xs, ys))
         for r in runs if xs[0] <= r['viol'] <= xs[-1]]
    return st.median(v) if v else float('nan')


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else PAPER
    os.makedirs(out, exist_ok=True)

    (Bc, Ec), (Bw, Ew) = both_envelopes()
    R = load_runs()

    fig = plt.figure(figsize=(7.2, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.26)

    # -- the two envelopes, and the configurations the wider sweep adds -----
    ax = fig.add_subplot(gs[0])
    common_names = {n for _, _, n in Ec}
    extra = [b for b in Bw if b['name'] not in {x['name'] for x in Bc}]
    ax.scatter([b['viol'] for b in Bc], [b['ch4'] for b in Bc],
               s=6, c=C_GRID, marker='s', linewidths=0, zorder=1,
               label=f'Within the common range (n={len(Bc)})')
    ax.scatter([b['viol'] for b in extra], [b['ch4'] for b in extra],
               s=14, facecolors='none', edgecolors=C_WIDE, linewidths=0.8,
               zorder=2, label=f'Multiplier above 1.3 (n={len(extra)})')
    ax.plot([p[0] for p in Ew], [p[1] for p in Ew], color=C_WIDE, lw=1.6,
            ls='--', zorder=3, label=f'Wider envelope ({len(Ew)} points)')
    ax.plot([p[0] for p in Ec], [p[1] for p in Ec], color=C_COMMON, lw=1.4,
            zorder=4, label=f'Reported envelope ({len(Ec)} points)')
    ax.set_xlabel('Steps above the soft limit (%)')
    ax.set_ylabel('Mean methane production (m$^3$/d)')
    ax.set_title('(a) the two envelopes', loc='left')
    ax.set_xlim(-1, 50)
    ax.grid(axis='y', color=C_GRID, lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=False, labelspacing=0.3)
    # Every added configuration sits below 2 % violation, where the two
    # envelopes are closest; the inset is the only place they can be told
    # apart at all.
    axi = ax.inset_axes([0.46, 0.40, 0.36, 0.40])
    axi.scatter([b['viol'] for b in Bc], [b['ch4'] for b in Bc],
                s=5, c=C_GRID, marker='s', linewidths=0)
    axi.scatter([b['viol'] for b in extra], [b['ch4'] for b in extra],
                s=14, facecolors='none', edgecolors=C_WIDE, linewidths=0.8)
    axi.plot([p[0] for p in Ew], [p[1] for p in Ew], color=C_WIDE, lw=1.4,
             ls='--')
    axi.plot([p[0] for p in Ec], [p[1] for p in Ec], color=C_COMMON, lw=1.2)
    axi.set_xlim(-0.08, 2.0)
    axi.set_ylim(1450, 1800)
    axi.tick_params(labelsize=6)
    axi.set_title('0–2 % detail', fontsize=6.5, pad=2)
    ax.indicate_inset_zoom(axi, edgecolor=C_GRID)

    # -- what that does to each family median ------------------------------
    ax2 = fig.add_subplot(gs[1])
    y = np.arange(len(ORDER))[::-1]
    for yi, fam in zip(y, ORDER):
        c = C_ON if fam in ON else (C_OFF if fam in OFF else C_FREE)
        mc, mw = med_against(R[fam], Ec), med_against(R[fam], Ew)
        ax2.plot([mw, mc], [yi, yi], color=c, lw=1.0, alpha=0.4, zorder=1)
        ax2.plot([mw], [yi], 'o', color=c, ms=4, mfc='white', mew=1.1,
                 zorder=2)
        ax2.plot([mc], [yi], 'o', color=c, ms=4.5, zorder=3)
    ax2.axvline(0, color=C_COMMON, lw=1.0, zorder=1)
    ax2.set_yticks(y)
    ax2.set_yticklabels([LABEL[f] for f in ORDER])
    ax2.set_xlabel('Family median against each envelope (m$^3$/d)')
    ax2.set_title('(b) what it changes', loc='left')
    ax2.grid(axis='x', color=C_GRID, lw=0.4, alpha=0.6)
    ax2.set_axisbelow(True)
    ax2.margins(y=0.05)
    h = [plt.Line2D([], [], color=C_COMMON, marker='o', ls='', ms=4.5),
         plt.Line2D([], [], color=C_COMMON, marker='o', ls='', ms=4,
                    mfc='white', mew=1.1)]
    ax2.legend(h, ['reported envelope', 'wider envelope'],
               loc='lower right', frameon=False, handletextpad=0.3,
               labelspacing=0.3,
               title='the two coincide to\nunder 1 m$^3$/d', title_fontsize=7)

    p = os.path.join(out, 'fig_range.pdf')
    fig.savefig(p)
    plt.close(fig)

    shifts = [abs(med_against(R[f], Ec) - med_against(R[f], Ew))
              for f in ORDER]
    print(f'fig_range.pdf  {len(Bc)} vs {len(Bw)} evaluations, '
          f'{len(Ec)} vs {len(Ew)} envelope points; '
          f'largest family-median shift {max(shifts):.1f} m3/d')


if __name__ == '__main__':
    main()

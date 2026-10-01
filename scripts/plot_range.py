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
# Matches the frontier figure: the conventional sweep is the
# baseline, so it is darker than the gridlines it sits on.
C_CONV = '#8E8E8E'

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

    # Everything the widening does happens between 0.5 and 1.5 % violation:
    # the 16 added evaluations all fall there, and the three envelope points
    # they contribute are the only places the two curves differ at all.  Over
    # the rest of the sweep the wider envelope is the reported one, point for
    # point, so a panel spanning the full axis draws one curve twice and
    # hides the one region worth looking at.
    fig = plt.figure(figsize=(4.6, 3.2))
    ax = fig.add_subplot(1, 1, 1)
    extra = [b for b in Bw if b['name'] not in {x['name'] for x in Bc}]
    added = [p for p in Ew if p not in Ec]

    ax.scatter([b['viol'] for b in Bc], [b['ch4'] for b in Bc],
               s=11, c=C_CONV, marker='s', linewidths=0, alpha=0.75,
               zorder=1,
               label=f'Common range (n={len(Bc)})')
    ax.scatter([b['viol'] for b in extra], [b['ch4'] for b in extra],
               s=22, facecolors='none', edgecolors=C_WIDE, linewidths=1.0,
               zorder=2, label=f'Multiplier >1.3 (n={len(extra)})')
    ax.plot([p[0] for p in Ew], [p[1] for p in Ew], color=C_WIDE, lw=2.0,
            ls='-', alpha=0.55, zorder=3,
            label=f'Wider envelope ({len(Ew)} pts)')
    ax.plot([p[0] for p in Ec], [p[1] for p in Ec], color=C_COMMON, lw=1.3,
            zorder=4, label=f'Reported envelope ({len(Ec)} pts)')
    # The three points where the two curves part company.
    ax.scatter([p[0] for p in added], [p[1] for p in added],
               s=46, facecolors='none', edgecolors=C_WIDE, linewidths=1.6,
               zorder=5, label=f'Added envelope points (n={len(added)})')

    ax.set_xlabel('Steps above the soft limit (%)')
    ax.set_ylabel('Mean methane (m$^3$/d)')
    ax.set_xlim(-0.08, 2.5)
    ax.set_ylim(1400, 1820)
    ax.grid(axis='y', color=C_GRID, lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=False, labelspacing=0.3,
              handletextpad=0.5, fontsize=6.5)
    # Beyond this point the two envelopes are the same 79 points, so the
    # panel stops rather than carrying 45 % of empty axis.
    ax.set_title('Past 1.5 %, the two envelopes are the same points',
                 loc='left', fontsize=7, style='italic', color=C_COMMON)

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

#!/usr/bin/env python3
"""The frontier before and after the conventional controllers are given the
same dynamic freedom as the learned policies.

The three conventional families hold the feed-strength multiplier at a
constant chosen for the episode, so the envelope they define is the best of a
static-tuning set, while a learned policy sets both inputs at every step.
Panel (a) is that comparison.  Panel (b) adds a conventional family with the
same freedom -- a PI loop on the feed rate and a proportional loop with a dead
band on the multiplier -- and redraws the envelope over both sets.

    python3 scripts/plot_frontier_dual.py [outdir]
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_tables import DATA
from plot_figures import (C_CONV, C_ENV, C_GRID, C_OFF, C_ON, OFF_POLICY,
                          ON_POLICY, load_reward)

C_DUAL = '#CC79A7'          # Okabe-Ito reddish purple, distinct from both
PAPER = os.path.expanduser('~/code/biogas/ADM1/papers/mypaper/img')

LABEL = {'ppo': 'PPO', 'a2c': 'A2C', 'trpo': 'TRPO',
         'recurrentppo': 'RecurrentPPO', 'sac': 'SAC', 'tqc': 'TQC',
         'ddpg': 'DDPG', 'td3': 'TD3', 'crossq': 'CrossQ'}


def upper_envelope(recs):
    pts = sorted((r['viol'], r['ch4']) for r in recs)
    E, best = [], -1.0
    for v, c in pts:
        if c > best:
            E.append((v, c))
            best = c
    return [a for a, _ in E], [b for _, b in E]


def draw(ax, static, dual, R, with_dual, title):
    """One panel: the conventional cloud, its envelope, and the runs."""
    ax.scatter([r['viol'] for r in static], [r['ch4'] for r in static],
               s=6, c=C_CONV, marker='s', linewidths=0, alpha=0.40, zorder=1,
               label=f'Static tuning ({len(static)})')
    if with_dual:
        ax.scatter([r['viol'] for r in dual], [r['ch4'] for r in dual],
                   s=5, c=C_DUAL, marker='D', linewidths=0, alpha=0.18,
                   zorder=0, label=f'Dual-loop PI ({len(dual)})')

    recs = static + dual if with_dual else static
    xs, ys = upper_envelope(recs)
    ax.plot(xs, ys, c=C_ENV, lw=1.4, zorder=4,
            label=f'Conventional envelope ({len(xs)} points)')

    for keys, colour, marker, name in ((ON_POLICY, C_ON, 'o', 'On-policy'),
                                       (OFF_POLICY, C_OFF, '^', 'Off-policy')):
        v = [d['viol'] for k in keys for d in R.get(k, [])]
        c = [d['ch4'] for k in keys for d in R.get(k, [])]
        ax.scatter(v, c, s=9, c=colour, marker=marker, alpha=0.35,
                   linewidths=0, zorder=2, label=f'{name} run')

    # Family medians, as in the published figure.
    for keys, colour, marker in ((ON_POLICY, C_ON, 'o'),
                                 (OFF_POLICY, C_OFF, '^')):
        for k in keys:
            if not R.get(k):
                continue
            ax.plot([st.median([d['viol'] for d in R[k]])],
                    [st.median([d['ch4'] for d in R[k]])],
                    marker=marker, ms=7.0, mfc=colour, mec='white', mew=1.1,
                    ls='', zorder=5)

    above = sum(1 for k in ON_POLICY + OFF_POLICY for d in R.get(k, [])
                if xs[0] <= d['viol'] <= xs[-1]
                and d['ch4'] > float(np.interp(d['viol'], xs, ys)))
    scored = sum(1 for k in ON_POLICY + OFF_POLICY for d in R.get(k, [])
                 if xs[0] <= d['viol'] <= xs[-1])
    ax.set_title(title, loc='left')
    ax.set_xlabel('Steps above the soft limit (%)')
    ax.set_xlim(-1, 50)
    ax.set_ylim(600, 2250)
    ax.grid(axis='y', color=C_GRID, lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    return above, scored


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else PAPER
    static = [json.load(open(f)) for f in glob.glob(f'{DATA}/evbase/*.json')]
    dual = [json.load(open(f)) for f in glob.glob(
        os.path.join(os.path.dirname(HERE), 'results_dualloop', '*.json'))]
    R = load_reward()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.1), sharey=True,
                                   gridspec_kw={'wspace': 0.08})
    a1, n1 = draw(ax1, static, dual, R, False,
                  '(a) conventional controllers tuned statically')
    a2, n2 = draw(ax2, static, dual, R, True,
                  '(b) with a dual-loop conventional family')
    ax1.set_ylabel('Mean methane flow (m$^3$/d)')
    ax2.tick_params(labelleft=False)
    ax1.legend(loc='lower right', frameon=False, fontsize=6.2,
               handletextpad=0.4, borderaxespad=0.3, labelspacing=0.35)
    ax2.legend(loc='lower right', frameon=False, fontsize=6.2,
               handletextpad=0.4, borderaxespad=0.3, labelspacing=0.35)

    for ax, (a, n) in ((ax1, (a1, n1)), (ax2, (a2, n2))):
        ax.text(0.97, 0.96, f'{a} of {n} runs above the envelope',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=6.3, color=C_ENV,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none',
                          alpha=0.8))

    p = os.path.join(out, 'fig_frontier_dual.pdf')
    fig.savefig(p)
    plt.close(fig)
    print(f'fig_frontier_dual.pdf  {len(static)} static, {len(dual)} dual-loop; '
          f'runs above the envelope {a1}/{n1} -> {a2}/{n2}')


if __name__ == '__main__':
    main()

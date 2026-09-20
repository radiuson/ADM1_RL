#!/usr/bin/env python3
"""Regenerate the two figures the manuscript includes.

Both figures were originally produced by throwaway code, so they could not be
rebuilt when the evaluation changed underneath them.  This script reads the
same evaluation records and the same conventional envelope that
``build_tables.py`` reads, so a figure and the table beside it cannot disagree.

    python3 scripts/plot_figures.py [outdir]

``outdir`` defaults to the manuscript directory, which is where the figures
have to land for a compile to pick them up.

fig_frontier.pdf  production against violation rate, every reward-penalty run
                  against the conventional Pareto envelope
fig_tracking.pdf  attained against specified violation rate for the fourteen
                  constrained families
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

from build_tables import CORRECTED, WEIGHTS, envelope

SP = ('/tmp/claude-1000/-home-ihpc-code/'
      'd8e23994-23f1-4be7-bbbc-30cb1d90de5a/scratchpad')
PAPER = os.path.expanduser('~/code/biogas/ADM1/papers/mypaper')

# The episode is 60 control steps, so a cost limit of d excursions specifies a
# violation rate of d/60.  This is the mapping the constrained runs are set up
# with, and the diagonal of the tracking figure.
EPISODE_STEPS = 60

# Families are drawn by class rather than individually: the classification is
# what the figures are evidence for, and fourteen separate colours would not be
# readable at column width.
ON_POLICY = ('ppo', 'a2c', 'trpo', 'recurrentppo')
OFF_POLICY = ('sac', 'tqc', 'ddpg', 'td3', 'crossq')
GRADIENT_FREE = ('ars',)

CMDP_OFF = ('SACLag', 'SACPID', 'DDPGLag', 'TD3Lag')

# The reference plant's own exceedance rate, once its VFA readings are mapped
# through the observed model-to-plant ratio.  The ratio is not constant, so its
# endpoints bracket the plant rather than placing it at one rate.
PLANT_LO, PLANT_HI = 0.1, 23.8

# Dual-update rule, which is what the tracking figure separates on.
PROJECTION = ('CPO', 'PCPO', 'CUP', 'FOCOPS', 'OnCRPO', 'P3O')
PID = ('TRPOPID', 'CPPOPID', 'SACPID')
DUAL_ASCENT = ('PPOLag', 'TRPOLag', 'SACLag', 'DDPGLag', 'TD3Lag')

# A colour-blind-safe set; the two classes also differ in marker so the figure
# survives greyscale printing.
C_ON, C_OFF, C_FREE = '#0072B2', '#D55E00', '#009E73'
C_ENV, C_GRID = '#333333', '#BBBBBB'

plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})


def load_reward():
    """Reward-penalty runs, filtered exactly as build_tables filters them."""
    R = collections.defaultdict(list)
    for f in glob.glob(f'{SP}/sev/evres/*.json'):
        d = json.load(open(f))
        if d.get('steps') != 150000 or '_h5_' in d.get('model', ''):
            continue
        a = d.get('algo', '').lower()
        if d.get('w') not in WEIGHTS:
            continue
        if a in CORRECTED and CORRECTED[a] not in d.get('model', ''):
            continue
        R[a].append(d)
    for a in list(R):
        seen = {}
        for d in sorted(R[a], key=lambda x: x.get('model', '')):
            seen.setdefault((d.get('w'), d.get('seed')), d)
        R[a] = list(seen.values())
    return R


def load_cmdp():
    R = collections.defaultdict(list)
    for f in glob.glob(f'{SP}/sev/evcmdp/*.json'):
        d = json.load(open(f))
        if d.get('cost_limit') is None:      # the unconstrained reference
            continue
        R[d['algo']].append(d)
    return R


def fig_frontier(out):
    B, E = envelope()
    R = load_reward()

    fig, ax = plt.subplots(figsize=(7.0, 3.2))

    # Where the reference plant's own record falls once its VFA readings are
    # mapped through the observed model-to-plant ratio.  The ratio is not
    # constant, so its endpoints give a band rather than a line; see the
    # scope-of-anchoring section.
    # Neutral grey, because every saturated colour on this panel already names
    # an algorithm class.
    ax.axvspan(PLANT_LO, PLANT_HI, color='#8C8C8C', alpha=0.13, lw=0, zorder=0,
               label=f'Reference plant record ({PLANT_LO:g}–{PLANT_HI:g}%)')

    # The conventional configurations as a cloud, with their upper envelope on
    # top: the envelope is the comparison, the cloud shows how densely it is
    # supported.
    ax.scatter([b['viol'] for b in B], [b['ch4'] for b in B],
               s=7, c=C_GRID, marker='s', linewidths=0, zorder=1,
               label=f'Conventional configurations (n={len(B)})')
    ax.plot([e[0] for e in E], [e[1] for e in E],
            c=C_ENV, lw=1.4, zorder=3,
            label=f'Conventional Pareto envelope ({len(E)} points)')

    for keys, colour, marker, name in (
            (ON_POLICY, C_ON, 'o', 'On-policy'),
            (OFF_POLICY, C_OFF, '^', 'Off-policy'),
            (GRADIENT_FREE, C_FREE, 'D', 'Gradient-free (ARS)')):
        xs = [r['viol'] for k in keys for r in R.get(k, [])]
        ys = [r['ch4'] for k in keys for r in R.get(k, [])]
        ax.scatter(xs, ys, s=11, c=colour, marker=marker, alpha=0.65,
                   linewidths=0, zorder=2, label=f'{name} (n={len(xs)})')

    ax.set_xlabel('Steps above the VFA soft limit (% of control steps)')
    ax.set_ylabel('Mean methane production (m$^3$/d)')
    ax.set_xlim(left=-1)
    ax.legend(loc='lower right', frameon=False, ncol=1)
    ax.grid(axis='y', color=C_GRID, lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    p = os.path.join(out, 'fig_frontier.pdf')
    fig.savefig(p)
    plt.close(fig)
    npts = sum(len(v) for v in R.values())
    print(f'fig_frontier.pdf  {len(B)} conventional, {len(E)} envelope points, '
          f'{npts} RL runs')


def fig_tracking(out):
    R = load_cmdp()

    # Both axes are the same quantity in the same units, so the panel is square
    # and the diagonal sits at 45 degrees; on a stretched panel, distance from
    # it would not read as specification error.
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.set_aspect('equal', adjustable='box')

    limits = sorted({d['cost_limit'] for v in R.values() for d in v})
    targets = [c / EPISODE_STEPS * 100 for c in limits]

    # Per-family medians at each limit: these are the points the figure plots,
    # and the highest of them sets the panel's extent.
    med = {}
    for fam, rs in R.items():
        by = collections.defaultdict(list)
        for d in rs:
            by[d['cost_limit']].append(d['viol'])
        med[fam] = {c: st.median(v) for c, v in by.items()}
    top = max(v for m in med.values() for v in m.values())

    # The panel is squared on the data, so the diagonal runs corner to corner
    # and distance from it reads directly as specification error.
    hi = max(max(targets), top) * 1.06
    ax.plot([0, hi], [0, hi], c=C_ENV, lw=1.0, ls='--', zorder=3,
            label='Exact specification')
    # Above the diagonal the controller uses more of the excursion budget than
    # was specified; below it, less.  A faint tint is enough to name the side:
    # anything stronger competes with the fourteen tracking lines.
    ax.fill_between([0, hi], [0, hi], hi, color=C_OFF, alpha=0.05,
                    zorder=0, lw=0,
                    label='Attained rate exceeds specification')

    style = {
        'Projection': (PROJECTION, C_ON, 'o'),
        'PID dual update': (PID, C_FREE, 's'),
        'Plain dual ascent': (DUAL_ASCENT, C_OFF, '^'),
    }
    for name, (fams, colour, marker) in style.items():
        for i, fam in enumerate(fams):
            if fam not in med:
                continue
            cs = sorted(med[fam])
            xs = [c / EPISODE_STEPS * 100 for c in cs]
            ys = [med[fam][c] for c in cs]
            # Off-policy families are drawn open, so the class is legible
            # without adding a second legend.
            off = fam in CMDP_OFF
            ax.plot(xs, ys, c=colour, lw=0.9, alpha=0.75, zorder=2,
                    marker=marker, ms=3.4,
                    mfc='white' if off else colour, mew=0.9,
                    ls=':' if off else '-',
                    label=name if i == 0 else None)

    ax.set_xlabel('Specified violation rate (%), from the cost limit')
    ax.set_ylabel('Attained violation rate (%)')
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xticks(targets)
    ax.set_xticklabels([f'{t:.0f}' for t in targets])
    ax.grid(color=C_GRID, lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    # The open marker and dotted line carry the class distinction, so they need
    # a key of their own.  The lines fan out across the whole square, leaving
    # no corner free, so the legend sits beside the panel.
    h, l = ax.get_legend_handles_labels()
    h.append(plt.Line2D([], [], c=C_ENV, lw=0.9, ls=':', marker='o', ms=3.4,
                        mfc='white', mew=0.9))
    l.append('Off-policy family')
    ax.legend(h, l, loc='upper left', bbox_to_anchor=(1.03, 1.0),
              frameon=False, borderaxespad=0, handlelength=1.8,
              labelspacing=0.4)

    p = os.path.join(out, 'fig_tracking.pdf')
    fig.savefig(p)
    plt.close(fig)
    print(f'fig_tracking.pdf  {len(R)} constrained families, '
          f'{len(limits)} cost limits, {sum(len(v) for v in R.values())} runs')


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else PAPER
    os.makedirs(out, exist_ok=True)
    fig_frontier(out)
    fig_tracking(out)


if __name__ == '__main__':
    main()

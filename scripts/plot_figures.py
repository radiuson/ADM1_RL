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

from build_tables import CORRECTED, DATA, WEIGHTS, envelope

# The manuscript keeps its figures in img/; a path given on the command
# line is used as-is.
PAPER = os.path.expanduser('~/code/biogas/ADM1/papers/mypaper/img')

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

# Dual-update rule, which is what the tracking figure separates on.
PROJECTION = ('CPO', 'PCPO', 'CUP', 'FOCOPS', 'OnCRPO', 'P3O')
PID = ('TRPOPID', 'CPPOPID', 'SACPID')
DUAL_ASCENT = ('PPOLag', 'TRPOLag', 'SACLag', 'DDPGLag', 'TD3Lag')

# A colour-blind-safe set; the two classes also differ in marker so the figure
# survives greyscale printing.
C_ON, C_OFF, C_FREE = '#0072B2', '#D55E00', '#009E73'
C_ENV, C_GRID = '#333333', '#BBBBBB'
# The conventional sweep is the baseline the whole comparison is
# against, not background: it needs to read as a population at
# print size, so it is darker and larger than the gridline grey.
C_CONV = '#8E8E8E'

# The conventional grid is three controller types, and they do not reach the
# envelope equally: 76 of its 79 points are PI, three are constant feed and
# none is rule-based.  One grey for all of them hides that, so each is drawn
# with its own marker in the one tone.
CONV_KINDS = (('pi', 's', 'PI loops'),
              ('const', 'P', 'Constant feed'),
              ('rule', 'X', 'Rule-based'))

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
    for f in glob.glob(f'{DATA}/evres/*.json'):
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
    for f in glob.glob(f'{DATA}/evcmdp/*.json'):
        d = json.load(open(f))
        if d.get('cost_limit') is None:      # the unconstrained reference
            continue
        R[d['algo']].append(d)
    return R


# Most of the evaluations sit at low violation rates -- the region a plant
# would actually be operated in -- where a single panel spanning the full
# sweep compresses them into its left fifth.  The second panel is that
# region at its own scale.
ZOOM_HI = 10.0


def _hull(P):
    """Convex hull of a point set, by Andrew's monotone chain.

    Written out rather than taken from scipy, because every figure in this
    repository is reproducible with numpy and matplotlib alone.
    """
    P = sorted(set(P))
    if len(P) < 3:
        return P

    def half(pts):
        out = []
        for p in pts:
            while len(out) >= 2:
                (x1, y1), (x2, y2) = out[-2], out[-1]
                if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) > 0:
                    break
                out.pop()
            out.append(p)
        return out[:-1]

    return half(P) + half(P[::-1])


def _core(P, keep=0.8):
    """The densest `keep` of a point set, by distance from its centre.

    Both classes span most of the axis, so a hull around every run would
    give two outlines that each enclose the other's points and say nothing.
    Dropping the outermost fifth leaves the region each class actually
    occupies.  Distance is measured after scaling each axis to the spread
    of the combined data, so neither axis dominates the ranking.
    """
    import math
    xs = [p[0] for p in P]
    ys = [p[1] for p in P]
    cx, cy = sorted(xs)[len(xs) // 2], sorted(ys)[len(ys) // 2]
    sx = (max(xs) - min(xs)) or 1.0
    sy = (max(ys) - min(ys)) or 1.0
    ranked = sorted(P, key=lambda p: math.hypot((p[0] - cx) / sx,
                                                (p[1] - cy) / sy))
    return ranked[:max(3, int(round(keep * len(ranked))))]


def _draw_frontier(ax, B, E, R, legend):
    """The cloud, its envelope, and the runs, on whatever range ax carries."""
    # The conventional configurations as a cloud, with their upper envelope on
    # top: the envelope is the comparison, the cloud shows how densely it is
    # supported.
    for kind, mk, name in CONV_KINDS:
        sub = [b for b in B if b.get('kind') == kind]
        if not sub:
            continue
        ax.scatter([b['viol'] for b in sub], [b['ch4'] for b in sub],
                   s=10 if kind == 'pi' else 14, c=C_CONV, marker=mk,
                   linewidths=0, alpha=0.75, zorder=1,
                   label=f'{name} (n={len(sub)})')
    # The envelope is a running maximum, so it is supported at the
    # configurations that set a new record and interpolated between them.
    # Marking those points shows that structure, which a plain line through
    # them reads as a smooth frontier instead.
    ax.plot([e[0] for e in E], [e[1] for e in E],
            c=C_ENV, lw=1.4, zorder=3,
            label=f'Best conventional at each rate ({len(E)} pts)')
    ax.scatter([e[0] for e in E], [e[1] for e in E],
               s=4.5, c=C_ENV, marker='o', linewidths=0, zorder=4)

    # ARS is a single unresolved reference family, and drawing it in a third
    # saturated colour drew the eye to its few low-production runs as though
    # they were a finding.  The panel carries the two classes the comparison
    # is between; ARS is reported in Table 2 and discussed in Section 9.
    classes = ((ON_POLICY, C_ON, 'o', 'On-policy'),
               (OFF_POLICY, C_OFF, '^', 'Off-policy'))

    for keys, colour, marker, name in classes:
        xs = [r['viol'] for k in keys for r in R.get(k, [])]
        ys = [r['ch4'] for k in keys for r in R.get(k, [])]
        ax.scatter(xs, ys, s=11, c=colour, marker=marker, alpha=0.38,
                   linewidths=0, zorder=2, label=f'{name} (n={len(xs)})')

    # Individual runs of the two classes overlap heavily, and the claim the
    # section makes is about family medians rather than about single runs.
    # Each family is therefore also drawn once, at its median violation rate
    # and offset from the envelope there by its median difference -- the
    # quantity Table 2 reports.  Drawn outlined so they read as summaries
    # over the cloud rather than as further runs.
    ex, ey = [e[0] for e in E], [e[1] for e in E]
    for keys, colour, marker, _ in classes:
        for k in keys:
            runs = R.get(k, [])
            if not runs:
                continue
            mv = st.median([r['viol'] for r in runs])
            ds = [r['ch4'] - float(np.interp(r['viol'], ex, ey))
                  for r in runs if ex[0] <= r['viol'] <= ex[-1]]
            if not ds:
                continue
            ax.plot([mv], [float(np.interp(mv, ex, ey)) + st.median(ds)],
                    marker=marker, ms=8.5, mfc=colour, mec='white', mew=1.3,
                    ls='', zorder=5)

    # An outline round the densest 80 % of each class, so the two
    # populations read as groups rather than as 360 separate marks.
    for keys, colour, _, _ in classes:
        P = [(r['viol'], r['ch4']) for k in keys for r in R.get(k, [])]
        H = _hull(_core(P))
        if len(H) < 3:
            continue
        ax.fill([p[0] for p in H], [p[1] for p in H], facecolor=colour,
                alpha=0.07, zorder=1, lw=0)
        ax.plot([p[0] for p in H] + [H[0][0]],
                [p[1] for p in H] + [H[0][1]],
                color=colour, lw=1.0, alpha=0.55, zorder=4)

    ax.grid(axis='y', color=C_GRID, lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    if legend:
        # The title of the legend carries the reading rule, which is the
        # one thing a reader needs and the panel has no free space for.
        ax.legend(loc='lower right', frameon=False, ncol=1,
                  title='A marker above the line beats every\n'
                        'conventional tuning at that rate',
                  title_fontsize=6.8, alignment='left')


def fig_frontier(out):
    B, E = envelope()
    R = load_reward()

    fig, (ax, axz) = plt.subplots(
        1, 2, figsize=(7.2, 3.2), sharey=True,
        gridspec_kw={'width_ratios': [1.55, 1.0], 'wspace': 0.06})

    _draw_frontier(ax, B, E, R, legend=True)
    ax.set_xlim(left=-1)
    ax.set_ylabel('Mean methane flow (m$^3$/d)')
    ax.set_title('(a) full range', loc='left')


    _draw_frontier(axz, B, E, R, legend=False)
    axz.set_xlim(-0.3, ZOOM_HI)
    axz.set_title(f'(b) detail: 0 to {ZOOM_HI:g}% of steps above the limit',
                  loc='left')
    # The zoom carries the shared y axis, so its own ticks would be a
    # duplicate column of numbers between the panels.
    axz.tick_params(labelleft=False)
    # The region (b) magnifies, marked on (a).  indicate_inset_zoom draws
    # connectors between axes that are not nested, which cross the figure;
    # the span alone reads correctly.
    ax.axvspan(0, ZOOM_HI, facecolor='none', edgecolor=C_ENV, lw=0.8,
               ls=(0, (3, 2)), zorder=4)

    # One axis label for a shared quantity, centred under both panels.
    fig.supxlabel('Steps above the soft limit (%)',
                  fontsize=8, y=0.015)

    p = os.path.join(out, 'fig_frontier.pdf')
    fig.savefig(p)
    plt.close(fig)
    drawn = ON_POLICY + OFF_POLICY
    npts = sum(len(R.get(k, [])) for k in drawn)
    nb = sum(1 for b in B if b['viol'] <= ZOOM_HI)
    nr = sum(1 for k in drawn for r in R.get(k, [])
             if r['viol'] <= ZOOM_HI)
    print(f'fig_frontier.pdf  {len(B)} conventional, {len(E)} envelope points, '
          f'{npts} RL runs; zoom holds {nb} conventional and {nr} RL')


def fig_tracking(out):
    """One panel per dual-update rule, which is what the figure separates on.

    Fourteen families on a single square overlap enough that no individual
    line can be followed, and the comparison the text draws is between the
    three rules rather than between families.  Splitting on the rule puts
    each group against its own copy of the diagonal, so the ordering the
    caption reports is read off the panels directly.
    """
    R = load_cmdp()

    limits = sorted({d['cost_limit'] for v in R.values() for d in v})
    targets = [c / EPISODE_STEPS * 100 for c in limits]

    # Per-family medians at each limit: these are the points the figure plots,
    # and the highest of them sets the panels' common extent.
    med = {}
    for fam, rs in R.items():
        by = collections.defaultdict(list)
        for d in rs:
            by[d['cost_limit']].append(d['viol'])
        med[fam] = {c: st.median(v) for c, v in by.items()}
    top = max(v for m in med.values() for v in m.values())
    hi = max(max(targets), top) * 1.06

    groups = [('(a) projection', PROJECTION, C_ON, 'o'),
              ('(b) PID dual update', PID, C_FREE, 's'),
              ('(c) plain dual ascent', DUAL_ASCENT, C_OFF, '^')]

    # Every panel is squared on the same range, so the diagonal sits at 45
    # degrees in each and the three are directly comparable.
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9),
                             sharex=True, sharey=True)

    for ax, (title, fams, colour, marker) in zip(axes, groups):
        ax.set_aspect('equal', adjustable='box')
        ax.fill_between([0, hi], [0, hi], hi, color=C_OFF, alpha=0.05,
                        zorder=0, lw=0)
        ax.plot([0, hi], [0, hi], c=C_ENV, lw=1.0, ls='--', zorder=3)

        drawn, ends = 0, []
        for fam in fams:
            if fam not in med:
                continue
            cs = sorted(med[fam])
            xs = [c / EPISODE_STEPS * 100 for c in cs]
            ys = [med[fam][c] for c in cs]
            # Off-policy families are drawn open, so the class stays legible
            # within a panel that is otherwise one colour.
            off = fam in CMDP_OFF
            ax.plot(xs, ys, c=colour, lw=0.9, alpha=0.8, zorder=2,
                    marker=marker, ms=3.4,
                    mfc='white' if off else colour, mew=0.9,
                    ls=':' if off else '-')
            ends.append((ys[-1], xs[-1], fam))
            drawn += 1

        # Each line is named at its end, which three panels leave room for
        # and one did not.  Families that finish close together are nudged
        # apart while keeping their order, so no two labels overlap.
        # A family that tracks near zero would otherwise be labelled on the
        # axis line itself, so the stack starts far enough up to clear it.
        ends.sort()
        gap, placed = hi * 0.062, []
        floor = hi * 0.022
        for yv, xv, fam in ends:
            pos = max(yv, floor) if not placed else max(yv, placed[-1] + gap)
            placed.append(pos)
            ax.annotate(fam, (xv, yv), xytext=(xv + hi * 0.022, pos),
                        textcoords='data', fontsize=6, color=colour,
                        va='center', annotation_clip=False)

        ax.set_title(f'{title} (n={drawn})', loc='left', fontsize=8)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_xticks(targets)
        ax.set_xticklabels([f'{t:.0f}' for t in targets])
        ax.grid(color=C_GRID, lw=0.4, alpha=0.5)
        ax.set_axisbelow(True)

    axes[0].set_ylabel('Attained violation rate (%)')
    axes[1].set_xlabel('Specified violation rate (%), from the cost limit')

    h = [plt.Line2D([], [], c=C_ENV, lw=1.0, ls='--'),
         plt.Line2D([], [], c=C_ENV, lw=0.9, ls=':', marker='o', ms=3.4,
                    mfc='white', mew=0.9),
         plt.Rectangle((0, 0), 1, 1, color=C_OFF, alpha=0.12)]
    axes[0].legend(h, ['Exact specification', 'Off-policy family',
                       'Attained exceeds specified'],
                   loc='upper left', bbox_to_anchor=(0, -0.20), ncol=3,
                   frameon=False, handlelength=1.8, handletextpad=0.5,
                   columnspacing=1.6, borderaxespad=0)

    p = os.path.join(out, 'fig_tracking.pdf')
    fig.savefig(p)
    plt.close(fig)
    print(f'fig_tracking.pdf  {len(R)} constrained families in '
          f'{len(groups)} panels, {len(limits)} cost limits, '
          f'{sum(len(v) for v in R.values())} runs')


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else PAPER
    os.makedirs(out, exist_ok=True)
    fig_frontier(out)
    fig_tracking(out)


if __name__ == '__main__':
    main()

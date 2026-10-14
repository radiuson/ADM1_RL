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
from matplotlib.legend_handler import HandlerTuple
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
# The text defines the projection rule as CPO, PCPO and CUP.  FOCOPS solves
# in distribution space, OnCRPO switches target on feasibility and P3O uses an
# exact penalty, so they are none of the three dual rules and get their own
# panel rather than being folded into projection.
PROJECTION = ('CPO', 'PCPO', 'CUP')
OTHER_ONPOLICY = ('FOCOPS', 'OnCRPO', 'P3O')
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

# The conventional grid is four controller families, and they do not reach
# the envelope equally: 522 of its 558 points come from the dual-loop family,
# 33 from the single-loop PI tunings, three from constant feed and none from
# the rule-based set.  One grey for all of them hides that, so each is drawn
# with its own marker in the one tone.
CONV_KINDS = (('dual', 'D', 'Dual-loop PI'),
              ('pi', 's', 'PI loops'),
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


_DELTA_LABEL = {'ppo': 'PPO', 'a2c': 'A2C', 'trpo': 'TRPO',
                'recurrentppo': 'RecurrentPPO', 'sac': 'SAC', 'tqc': 'TQC',
                'ddpg': 'DDPG', 'td3': 'TD3', 'crossq': 'CrossQ'}


def _deltas(E, R):
    """Each run as (violation rate, difference from the envelope).

    Runs outside the envelope's support have no value to be scored against
    and are left out, as they are everywhere else in the paper.
    """
    ex = [p[0] for p in E]
    ey = [p[1] for p in E]
    out = {}
    for k in ON_POLICY + OFF_POLICY:
        pts = []
        for r in R.get(k, []):
            if ex[0] <= r['viol'] <= ex[-1]:
                pts.append((r['viol'],
                            r['ch4'] - float(np.interp(r['viol'], ex, ey))))
        out[k] = pts
    return out


def _draw_delta(ax, E, R):
    """The same runs with the envelope subtracted, so zero is the claim."""
    # Which half-plane a marker falls in is the whole reading.  The tint
    # only has to separate the halves, so it stays well below the markers
    # in weight, and the two words name the sides in place of a sentence.
    ax.axhspan(0, 1e4, color=C_ON, alpha=0.03, lw=0, zorder=0)
    ax.axhspan(-1e4, 0, color=C_OFF, alpha=0.03, lw=0, zorder=0)
    ax.axhline(0, color=C_ENV, lw=1.1, zorder=4)

    D = _deltas(E, R)

    for keys, colour, marker, name in ((ON_POLICY, C_ON, 'o', 'On-policy'),
                                       (OFF_POLICY, C_OFF, '^', 'Off-policy')):
        xs = [v for k in keys for v, _ in D.get(k, [])]
        ys = [d for k in keys for _, d in D.get(k, [])]
        ax.scatter(xs, ys, s=10, c=colour, marker=marker, alpha=0.40,
                   linewidths=0, zorder=2)

    # One outlined marker per family at its median: the claim is about
    # families, not single runs.
    placed = []
    for keys, colour, marker in ((ON_POLICY, C_ON, 'o'),
                                 (OFF_POLICY, C_OFF, '^')):
        for k in keys:
            if not D.get(k):
                continue
            mv = st.median([v for v, _ in D[k]])
            md = st.median([d for _, d in D[k]])
            ax.plot([mv], [md], marker=marker, ms=7.5, mfc=colour,
                    mec='white', mew=1.2, ls='', zorder=5)
            placed.append((md, mv, k, colour))

    # Nine medians fall within a band a few tens of m3/d deep, so the
    # labels are stacked on one rule rather than each finding its own
    # offset: every label sits to the left of its marker at a column the
    # cloud has thinned out by, and the rows are walked apart from the
    # bottom up.  One rule for all nine is what makes a leader easy to
    # follow at print size.
    placed.sort()
    n = len(placed)
    # Spread the rows over the span the medians occupy, widened enough that
    # no two labels touch at 6 pt.
    lo = min(md for md, _, _, _ in placed)
    hi = max(md for md, _, _, _ in placed)
    step = max(18.5, (hi - lo) / max(n - 1, 1))
    base = (lo + hi) / 2 - step * (n - 1) / 2 - 4.0
    # One column for all nine, placed to the right of the cloud.  The runs
    # thin out past 35 % -- two of them fall in the band the column spans,
    # against about a hundred below 10 % -- so the names cover no data
    # there, whereas a column on the left sat over the densest part of the
    # panel.  Leaders run back to each marker.
    col_x = 33.0
    for i, (md, mv, k, colour) in enumerate(placed):
        ax.annotate(_DELTA_LABEL[k], (mv, md),
                    xytext=(col_x, base + i * step),
                    textcoords='data', fontsize=6.3, color=colour,
                    va='center', ha='left', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.18', fc='white',
                              ec='none', alpha=0.78),
                    arrowprops=dict(arrowstyle='-', color=colour,
                                    lw=0.4, alpha=0.55,
                                    shrinkA=0, shrinkB=4))

    ax.set_xlabel('Steps above the soft limit (%)')
    ax.set_ylabel('Difference from conventional envelope (m$^3$/d)')
    ax.set_xlim(-1, 50)
    # A symmetric log scale past 100 m3/d holds the runs that reach -265
    # without the body of the data being flattened to accommodate them, so
    # every scored run is drawn.
    ax.set_yscale('symlog', linthresh=100, linscale=0.9)
    ax.set_ylim(-320, 160)
    ax.set_yticks([-300, -200, -100, -50, 0, 50, 100])
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.grid(axis='y', color=C_GRID, lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    # Naming the two sides replaces the sentence that used to run across
    # the cloud.
    # On the left, where the label column no longer is.
    for y, txt, col in ((0.965, 'Above envelope', C_ON),
                        (0.035, 'Below envelope', C_OFF)):
        ax.text(0.015, y, txt, transform=ax.transAxes, ha='left',
                va='top' if y > 0.5 else 'bottom', fontsize=6.2,
                color=col, alpha=0.85, zorder=1)

    # Both panels use the same colour and shape for the same thing, so the
    # legend in (a) covers this one too; the labelled medians carry the
    # rest.  A second copy would only take space from the data.


def _draw_frontier(ax, B, E, R, legend):
    """The cloud, its envelope, and the runs, on whatever range ax carries."""
    # The conventional configurations as a cloud, with their upper envelope on
    # top: the envelope is the comparison, the cloud shows how densely it is
    # supported.
    # Each family keeps its own marker, which is what shows that the
    # envelope is carried by the PI loops, but they share one legend entry:
    # three entries for one visual layer crowded the panel, and the caption
    # names the markers.
    conv_handles = []
    for kind, mk, name in CONV_KINDS:
        sub = [b for b in B if b.get('kind') == kind]
        if not sub:
            continue
        # The dual-loop grid is two thousand configurations against a few
        # hundred for the other three, so it is drawn smaller and fainter:
        # at one weight it covers the panel and the envelope it supports
        # can no longer be read off it.
        small = kind == 'dual'
        ax.scatter([b['viol'] for b in sub], [b['ch4'] for b in sub],
                   s=4 if small else (7 if kind == 'pi' else 10),
                   c=C_CONV, marker=mk, linewidths=0,
                   alpha=0.16 if small else 0.45, zorder=1)
        conv_handles.append(
            plt.Line2D([], [], marker=mk, ls='', ms=3.4, mfc=C_CONV,
                       mec=C_CONV, alpha=0.75))
    # The envelope is a running maximum, so it is supported at the
    # configurations that set a new record and interpolated between them.
    # Marking those points shows that structure, which a plain line through
    # them reads as a smooth frontier instead.
    ax.plot([e[0] for e in E], [e[1] for e in E],
            c=C_ENV, lw=1.4, zorder=3,
            label=f'Conventional Pareto envelope\n({len(E)} supporting points)')
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
        # What the envelope means is stated in the caption; a title here
        # would run across the cloud, which fills the panel's free space.
        med = (plt.Line2D([], [], marker='o', ls='', ms=7.0, mfc=C_ON,
                          mec='white', mew=1.1),
               plt.Line2D([], [], marker='^', ls='', ms=7.0, mfc=C_OFF,
                          mec='white', mew=1.1))
        h, lb = ax.get_legend_handles_labels()
        nconv = sum(1 for kind, _, _ in CONV_KINDS
                    if any(b.get('kind') == kind for b in B))
        # Placed inside the axes at the foot: the long envelope entry needs
        # the full panel width, which a corner anchored to the frame does
        # not leave it, and the region below the cloud carries no data.
        ax.legend([tuple(conv_handles)] + h + [med],
                  [f'Conventional configurations ({len(B)})'] + lb
                  + ['Family median'],
                  loc='lower center', bbox_to_anchor=(0.5, -0.02),
                  frameon=False, ncol=1, fontsize=6.2,
                  handletextpad=0.5, labelspacing=0.42,
                  handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)})


def _draw_zero(ax, E, Rc):
    """The comparison at zero violations, which is where the paper separates.

    Panel (a) reads the whole range at once, and over that range the
    conventional envelope is the better curve.  The claim the section makes
    is narrower: at a violation rate of exactly zero the conventional set
    stops at a bound, and most of the constrained on-policy families carry
    runs above it.  That is one number per family against one horizontal
    line, so it is drawn as a column rather than as a second cloud.
    """
    ex, ey = [e[0] for e in E], [e[1] for e in E]
    best = max(c for v, c in zip(ex, ey) if v < 0.01)

    # Families that reach zero at all, ordered by their median there: the
    # ordering is the reading, so it is the axis.
    rows = []
    for fam, runs in Rc.items():
        z = [r['ch4'] for r in runs if r['viol'] < 0.01]
        if not z:
            continue
        rows.append((st.median(z), fam, z))
    rows.sort()

    # The range is set from the data first: a span drawn to infinity would
    # otherwise set it, and twelve families a few hundred apart collapse
    # onto the line.
    allz = [c for _, _, z in rows for c in z]
    lo, hi = min(allz), max(allz)
    # The labels run outward from the line, so the right side -- where
    # eight of the twelve sit -- needs the wider margin of the two.
    pad = 0.05 * (hi - lo)
    ax.set_xlim(lo - pad, hi + 0.34 * (hi - lo))
    ax.axvline(best, color=C_ENV, lw=1.2, zorder=3)
    ax.axvspan(best, hi + 0.34 * (hi - lo), color=C_ON, alpha=0.04,
               lw=0, zorder=0)
    ax.axvspan(lo - pad, best, color=C_OFF, alpha=0.04, lw=0, zorder=0)

    for i, (md, fam, z) in enumerate(rows):
        off = fam in CMDP_OFF
        colour = C_OFF if off else C_ON
        # Every zero-violation run of the family, then its median outlined
        # over them: the spread is what shows the median is not one seed.
        ax.scatter(z, [i] * len(z), s=9, c=colour, marker='^' if off else 'o',
                   alpha=0.40, linewidths=0, zorder=2)
        ax.plot([md], [i], marker='^' if off else 'o', ms=7.0, mfc=colour,
                mec='white', mew=1.2, ls='', zorder=5)
        # Named on the side the family falls on, so the name reads outward
        # from the line and no label crosses it.
        out_r = md > best
        ax.annotate(f'{fam} ({len(z)})', (md, i),
                    xytext=(6 if out_r else -6, 0),
                    textcoords='offset points', fontsize=6.5, color=colour,
                    ha='left' if out_r else 'right', va='center', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.14', fc='white',
                              ec='none', alpha=0.78))

    ax.set_ylim(-0.7, len(rows) + 0.25)
    ax.set_yticks([])
    ax.set_xlabel('Mean methane flow at zero violations (m$^3$/d)')
    ax.grid(axis='x', color=C_GRID, lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    ax.annotate(f'Best conventional\nconfiguration ({best:.0f})',
                (best, len(rows) - 0.35), xytext=(-5, 0),
                textcoords='offset points', fontsize=6.2, color=C_ENV,
                ha='right', va='top', zorder=6,
                bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none',
                          alpha=0.82))
    return rows, best


def fig_frontier(out):
    B, E = envelope()
    R = load_reward()
    Rc = load_cmdp()

    # Two readings the section needs in turn.  (a) carries the whole
    # production-excursion range, where the conventional envelope is the
    # better curve and the learned runs sit below it.  (b) restricts to a
    # violation rate of zero, where the conventional set stops at a bound
    # and the ordering between the two classes is not the same.
    fig, (ax, axz) = plt.subplots(
        1, 2, figsize=(7.1, 3.2),
        gridspec_kw={'wspace': 0.20})

    _draw_frontier(ax, B, E, R, legend=True)
    ax.set_xlim(left=-1)
    ax.set_ylabel('Mean methane flow (m$^3$/d)')
    ax.set_xlabel('Steps above the soft limit (%)')
    ax.set_title('(a) Production\u2013excursion trade-off', loc='left')

    rows, best = _draw_zero(axz, E, Rc)
    axz.set_title('(b) At zero violations', loc='left')

    p = os.path.join(out, 'fig_frontier.pdf')
    fig.savefig(p)
    plt.close(fig)
    drawn = ON_POLICY + OFF_POLICY
    npts = sum(len(R.get(k, [])) for k in drawn)
    above = sum(1 for md, f, _ in rows if md > best)
    print(f'fig_frontier.pdf  {len(B)} conventional, {len(E)} envelope points, '
          f'{npts} reward-penalty runs; zero-violation column {len(rows)} '
          f'families, {above} above {best:.0f}')


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
              ('(c) plain dual ascent', DUAL_ASCENT, C_OFF, '^'),
              ('(d) other on-policy', OTHER_ONPOLICY, C_GRID, 'v')]

    # Every panel is squared on the same range, so the diagonal sits at 45
    # degrees in each and the four are directly comparable.
    fig, axes = plt.subplots(1, 4, figsize=(9.4, 2.9),
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

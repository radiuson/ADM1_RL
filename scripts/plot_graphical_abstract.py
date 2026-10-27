#!/usr/bin/env python3
"""The three data panels of the graphical abstract.

The graphical abstract lays out the benchmark as a flow: the shared
environment, the conventional sweep, the learned controllers, the common
scoring rule, and then the two regimes the paper separates.  Three of its
boxes carry real data rather than labels, and those are drawn here so they
come from the same records as the tables:

  panel 2  the conventional sweep and the Pareto envelope it defines
  panel 5  regime I, zero excursions: family medians against the best tuning
  panel 6  regime II, positive excursions: per-run differences from the envelope

Each is written as its own PDF and PNG so the layout can place them in the
poster boxes without rescaling type.  The surrounding boxes, arrows and
captions are set in the drawing, not here.

    python3 scripts/plot_graphical_abstract.py [outdir]
"""
from __future__ import annotations

import os
import statistics as st
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_tables import envelope
from plot_figures import CMDP_OFF, C_ON, C_OFF, C_ENV, load_cmdp

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(HERE), 'img_abstract')

# The abstract is read at a glance and often at half column width, so type is
# larger relative to the axes than in the manuscript figures.
plt.rcParams.update({
    'font.size': 8.5,
    'axes.labelsize': 8.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.6,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

C_GREY = '#9a948c'


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'),
                    bbox_inches='tight', pad_inches=0.02,
                    transparent=(ext == 'png'))
    plt.close(fig)


def frame(ax):
    """Two spines and a light horizontal grid: the poster boxes supply the rest."""
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis='y', color='#e3ded6', lw=0.6)
    ax.set_axisbelow(True)


def panel_envelope(B, E):
    """Every tuned configuration, with the envelope its best points trace."""
    fig, ax = plt.subplots(figsize=(6.16, 2.15))

    # A linear rate axis, so a rate of exactly zero is simply x = 0 and the
    # 446 configurations there need no special placement.
    HI = 48

    ax.scatter([d['viol'] for d in B], [d['ch4'] for d in B],
               s=2.0, c=C_GREY, alpha=0.5, lw=0, rasterized=True)

    # The envelope is a step: at a rate where a better tuning appears it holds
    # several points at one x.  Join the top of each step -- the value
    # attainable there -- so the curve carries no risers.
    zero = max(c for v, c, _ in E if v <= 0)
    top = {}
    for v, c, _ in E:
        if v > 0:
            top[v] = max(top.get(v, 0.0), c)
    ex = [0.0] + sorted(top)
    ey = [zero] + [top[v] for v in sorted(top)]
    ax.plot(ex, ey, color='#111111', lw=1.5, zorder=5,
            solid_joinstyle='round', solid_capstyle='round')

    # Markers along the curve rather than at every point: 43 at this width
    # would read as a thick line.  Spacing them by rate alone would bunch
    # them below 10 %, where most of the rise is, so they are spaced along
    # the curve's drawn length.
    lx = np.array(ex) / HI
    ly = (np.array(ey) - 600) / (2200 - 600)
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(lx), np.diff(ly)))])
    want = np.linspace(0, arc[-1], 16)
    mi = sorted({int(np.argmin(np.abs(arc - w))) for w in want})
    ax.plot([ex[i] for i in mi], [ey[i] for i in mi], ls='', marker='o',
            ms=3.4, mfc='#111111', mec='#111111', zorder=6)

    ax.set_xlim(-1.3, HI)
    # The zero-rate group runs down to 659 m3/d and is the population regime I
    # is about, so the window covers it rather than cutting it off.
    ax.set_ylim(600, 2200)
    ax.set_xlabel('Excursion rate (%)', labelpad=2)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_yticks([800, 1200, 1600, 2000])
    frame(ax)
    # The reference puts the unit over the axis rather than along it, which
    # keeps the panel short enough for the poster row.
    ax.set_ylabel('')
    ax.text(0.0, 1.04, 'Methane (m$^3$/d)', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=8.2, color='#1c1a17')
    ax.text(0.5, -0.30, 'Best tuning at each permitted rate',
            transform=ax.transAxes, ha='center', va='top', fontsize=7.8,
            color='#4a443c')
    ax.text(0.985, 0.055,
            f'{len(B)} configurations $\\rightarrow$ {len(E)}-point Pareto envelope',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=7.6, color='#4a443c')
    save(fig, 'ga_panel2_envelope')


def panel_zero(Rc, best):
    """Regime I: the family medians that reach a zero excursion rate."""
    rows = []
    for fam, runs in Rc.items():
        z = [d['ch4'] for d in runs if d['viol'] < 0.01]
        if z:
            rows.append((st.median(z), fam, z, fam in CMDP_OFF))
    rows.sort(key=lambda r: -r[0])

    fig, ax = plt.subplots(figsize=(4.4, 2.15))
    ax.axhline(best, color='#111111', lw=1.1, ls=(0, (4, 2.4)), zorder=3)

    for k, (med, fam, z, off) in enumerate(rows):
        c = C_OFF if off else C_ON
        # The runs behind each median, so the marker is not read as one number.
        ax.scatter([k] * len(z), z, s=5, c=c, alpha=0.28, lw=0,
                   marker='^' if off else 'o', zorder=4)
        ax.plot([k], [med], marker='^' if off else 'o', ms=6.2, mfc=c,
                mec='white', mew=1.0, ls='', zorder=6)

    ax.set_xlim(-0.7, len(rows) - 0.3)
    ax.set_ylim(700, 2050)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[1] for r in rows], rotation=42, ha='right',
                       fontsize=6.6)
    ax.set_ylabel('Methane flow (m$^3$/d)', labelpad=3)
    ax.set_yticks([800, 1200, 1600])
    frame(ax)
    ax.grid(False, axis='x')

    n_on_above = sum(1 for m, f, z, o in rows if not o and m > best)
    # The dashed line is the comparison.  Every family left of SACLag sits
    # above it and every family right of it below, so the label goes in the
    # empty upper right, tied to the line by a short leader.
    ax.annotate(f'best conventional {best:.0f}',
                xy=(len(rows) - 1.1, best),
                xytext=(len(rows) - 0.45, best + 285),
                fontsize=7.2, color='#111111', ha='right', va='bottom',
                arrowprops=dict(arrowstyle='-', lw=0.6, color='#111111',
                                shrinkA=1, shrinkB=2))
    ax.text(0.5, -0.46,
            f'{n_on_above} of 10 on-policy family medians exceed '
            f'the best conventional result',
            transform=ax.transAxes, ha='center', va='top', fontsize=7.8,
            color='#4a443c')
    h = [plt.Line2D([], [], ls='', marker='o', ms=5.4, mfc=C_ON, mec='white',
                    mew=0.9, label='On-policy'),
         plt.Line2D([], [], ls='', marker='^', ms=5.4, mfc=C_OFF, mec='white',
                    mew=0.9, label='Off-policy')]
    ax.legend(handles=h, loc='lower left', frameon=False, handletextpad=0.35,
              borderpad=0.1, labelspacing=0.3)
    save(fig, 'ga_panel5_zero')


def panel_positive(Rc, xs, ys):
    """Regime II: every run with a positive rate, scored against the envelope."""
    on, off = [], []
    for fam, runs in Rc.items():
        for d in runs:
            v = d['viol']
            if v <= 0 or not (xs[0] <= v <= xs[-1]):
                continue
            dd = d['ch4'] - float(np.interp(v, xs, ys))
            (off if fam in CMDP_OFF else on).append((v, dd))

    fig, ax = plt.subplots(figsize=(4.4, 2.15))
    ax.axhline(0, color='#111111', lw=1.1, ls=(0, (4, 2.4)), zorder=3)

    for pts, c, m in ((on, C_ON, 'o'), (off, C_OFF, '^')):
        ax.scatter([v for v, _ in pts], [d for _, d in pts], s=4.5, c=c,
                   alpha=0.3, lw=0, marker=m, zorder=4, rasterized=True)

    # Class medians in log-spaced bands, so the two curves can be compared
    # where the runs actually sit rather than at one pooled point.
    edges = np.logspace(np.log10(0.2), np.log10(35), 8)
    for pts, c, m in ((on, C_ON, 'o'), (off, C_OFF, '^')):
        bx, by = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = [d for v, d in pts if lo <= v < hi]
            if len(sel) >= 4:
                bx.append(np.sqrt(lo * hi))
                by.append(st.median(sel))
        ax.plot(bx, by, color=c, lw=1.2, marker=m, ms=5.0, mfc=c,
                mec='white', mew=0.9, zorder=6)

    ax.set_xscale('log')
    ax.set_xlim(0.18, 42)
    ax.set_ylim(-900, 220)
    ax.set_xlabel('Excursion rate (%)', labelpad=2)
    ax.set_ylabel('Methane difference (m$^3$/d)', labelpad=3)
    ax.set_yticks([-800, -600, -400, -200, 0, 200])
    frame(ax)

    above = sum(1 for _, d in on + off if d > 0)
    ax.annotate('conventional envelope', xy=(0.21, 0), xytext=(0.21, 80),
                fontsize=7.2, color='#111111', ha='left', va='bottom')
    ax.text(0.5, -0.30,
            f'the conventional envelope exceeds every learned-family median '
            f'({above} of {len(on) + len(off)} runs above it)',
            transform=ax.transAxes, ha='center', va='top', fontsize=7.8,
            color='#4a443c')
    h = [plt.Line2D([], [], color=C_ON, lw=1.2, marker='o', ms=5.0,
                    mfc=C_ON, mec='white', mew=0.9, label='On-policy'),
         plt.Line2D([], [], color=C_OFF, lw=1.2, marker='^', ms=5.0,
                    mfc=C_OFF, mec='white', mew=0.9, label='Off-policy')]
    ax.legend(handles=h, loc='lower left', frameon=False, handletextpad=0.35,
              borderpad=0.1, labelspacing=0.3)
    save(fig, 'ga_panel6_positive')


def main() -> None:
    B, E = envelope()
    xs = [e[0] for e in E]
    ys = [e[1] for e in E]
    best = max(c for v, c in zip(xs, ys) if v < 0.01)
    Rc = load_cmdp()

    panel_envelope(B, E)
    panel_zero(Rc, best)
    panel_positive(Rc, xs, ys)

    print(f'ga_panel2_envelope  {len(B)} configurations -> {len(E)} points, '
          f'{min(xs):.2f}-{max(xs):.1f}%, zero-violation best {best:.0f}')
    nz = sum(1 for runs in Rc.values()
             for d in runs if d['viol'] < 0.01)
    print(f'ga_panel5_zero      {nz} zero-violation runs')
    npos = sum(1 for runs in Rc.values() for d in runs
               if 0 < d['viol'] <= xs[-1])
    print(f'ga_panel6_positive  {npos} runs at a positive rate')
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()

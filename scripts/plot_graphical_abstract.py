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
    # A rate of exactly zero has no place on a log axis, and 446 of the
    # configurations sit there.  They are their own regime, so they get a
    # narrow band of their own on the left rather than being nudged onto the
    # axis, where they would read as a vertical stripe at one rate.
    fig, (az, ax) = plt.subplots(
        1, 2, figsize=(3.5, 2.05), sharey=True,
        gridspec_kw=dict(width_ratios=[1, 5.4], wspace=0.06))

    zb = [d['ch4'] for d in B if d['viol'] < 0.01]
    rng = np.random.default_rng(7)
    az.scatter(rng.uniform(-0.38, 0.38, len(zb)), zb, s=1.6, c=C_GREY,
               alpha=0.45, lw=0, rasterized=True)
    zero = max(zb)
    az.plot([-0.5, 0.5], [zero, zero], color=C_ENV, lw=1.3, zorder=5,
            solid_capstyle='butt')
    az.set_xlim(-0.6, 0.6)
    az.set_xticks([0])
    az.set_xticklabels(['0'])
    az.set_ylabel('Methane flow (m$^3$/d)', labelpad=3)
    az.spines['right'].set_visible(False)
    az.spines['top'].set_visible(False)
    az.grid(True, axis='y', color='#e3ded6', lw=0.6)
    az.set_axisbelow(True)
    az.annotate(f'{zero:.0f}', xy=(0, zero), xytext=(0, zero + 95),
                fontsize=7.4, color=C_ENV, ha='center')

    pos = [d for d in B if d['viol'] >= 0.01]
    ax.scatter([d['viol'] for d in pos], [d['ch4'] for d in pos], s=1.6,
               c=C_GREY, alpha=0.45, lw=0, rasterized=True)
    Ep = [(v, c) for v, c, _ in E if v >= 0.01]
    # The envelope steps from the zero-rate optimum to the first positive
    # rate, so the segment across the break is drawn rather than implied.
    ax.plot([Ep[0][0]] + [v for v, _ in Ep], [zero] + [c for _, c in Ep],
            color=C_ENV, lw=1.3, zorder=5)

    ax.set_xscale('log')
    ax.set_xlim(0.18, 60)
    ax.set_ylim(500, 2250)
    ax.set_xlabel('Excursion rate (%)', labelpad=2)
    ax.set_xticks([1, 10])
    ax.set_xticklabels(['1', '10'])
    ax.set_yticks([500, 1000, 1500, 2000])
    frame(ax)
    ax.tick_params(axis='y', length=0)
    ax.text(0.98, 0.055,
            f'{len(B)} configurations\n{len(E)}-point envelope',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=7.4, color='#4a443c', linespacing=1.35)
    save(fig, 'ga_panel2_envelope')


def panel_zero(Rc, best):
    """Regime I: the family medians that reach a zero excursion rate."""
    rows = []
    for fam, runs in Rc.items():
        z = [d['ch4'] for d in runs if d['viol'] < 0.01]
        if z:
            rows.append((st.median(z), fam, z, fam in CMDP_OFF))
    rows.sort(key=lambda r: -r[0])

    fig, ax = plt.subplots(figsize=(3.5, 2.05))
    ax.axhline(best, color=C_ENV, lw=1.1, ls=(0, (4, 2.4)), zorder=3)

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
                fontsize=7.0, color=C_ENV, ha='right', va='bottom',
                arrowprops=dict(arrowstyle='-', lw=0.6, color=C_ENV,
                                shrinkA=1, shrinkB=2))
    ax.set_title(f'{n_on_above} of 10 on-policy family medians exceed it',
                 fontsize=7.6, color='#4a443c', pad=4, loc='left')
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

    fig, ax = plt.subplots(figsize=(3.5, 2.05))
    ax.axhline(0, color=C_ENV, lw=1.1, ls=(0, (4, 2.4)), zorder=3)

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
    ax.set_title(f'the envelope leads every family median '
                 f'({above} of {len(on) + len(off)} runs above it)',
                 fontsize=7.6, color='#4a443c', pad=4, loc='left')
    ax.annotate('conventional envelope', xy=(0.21, 0), xytext=(0.21, 95),
                fontsize=7.0, color=C_ENV, ha='left', va='bottom')
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

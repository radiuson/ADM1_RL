#!/usr/bin/env python3
"""Where a learned policy holds VFA, against the conventional records it is
compared with.

Section 9 reports that a single pairing moves with the conventional tuning
chosen, so neither side of the comparison can be read off one record.  This
draws both as populations: every archived PI evaluation whose violation rate
falls in the matched band, and every seed of the constrained family at the
cost limit that lands in the same band.

The response is each record's own median VFA, which is the quantity the text
reports -- a median across records, not a median of pooled samples.  Taking it
from the records rather than from a replay keeps every record in the figure:
the earliest evaluations stored only summaries, and those happen to be the
high-VFA end of the band, so a replay-based figure could not show the upper
half of the distribution at all.

    python3 scripts/plot_vfa_dist.py [outdir]

``outdir`` defaults to the manuscript's img directory.
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

from build_tables import DATA

# The manuscript keeps its figures in img/; a path given on the command
# line is used as-is.
PAPER = os.path.expanduser('~/code/biogas/ADM1/papers/mypaper/img')

# The band Section 9 matches on: conventional records whose violation rate
# brackets the constrained policy's.
BAND_LO, BAND_HI = 4.0, 6.0

# The pairing the text discusses.
RL_FAMILY, RL_LIMIT = 'TRPOPID', 3.0

SOFT_MG = 300.0

C_PI, C_RL = '#0072B2', '#D55E00'
C_GRID, C_RULE = '#BBBBBB', '#333333'

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})


def load_pi():
    """Archived PI evaluations in the matched violation band.

    ``build_tables`` restricts the envelope to the common actuator range; every
    record in this band already satisfies that bound, so no further filtering
    applies here and the count matches the text.
    """
    R = []
    for f in glob.glob(f'{DATA}/evbase/*.json'):
        d = json.load(open(f))
        if not str(d.get('kind', '')).lower().startswith('pi'):
            continue
        if BAND_LO <= d['viol'] <= BAND_HI:
            R.append(d)
    return sorted(R, key=lambda d: d['vfa_med'])


def load_rl():
    R = []
    for f in glob.glob(f'{DATA}/evcmdp/*.json'):
        d = json.load(open(f))
        if d.get('algo') == RL_FAMILY and d.get('cost_limit') == RL_LIMIT:
            R.append(d)
    return sorted(R, key=lambda d: d['vfa_med'])


def strip(ax, y, vals, colour, marker, mfc=None, label=None):
    """One point per record, jittered only enough to separate ties."""
    rng = np.random.default_rng(0)
    off = rng.uniform(-0.085, 0.085, len(vals))
    ax.plot(vals, np.full(len(vals), y) + off, marker, color=colour,
            ms=4.0, ls='', mfc=colour if mfc is None else mfc, mew=0.9,
            alpha=0.85, zorder=3, label=label)


def panel_dist(ax, pi, rl):
    """Both populations on the one response, with their medians."""
    pv = [d['vfa_med'] for d in pi]
    rv = [d['vfa_med'] for d in rl]

    # One record's gains could not be identified from its summary, so it is
    # the only one that cannot be replayed; it is marked apart rather than
    # dropped, since it carries a median like any other.
    rep = [d['vfa_med'] for d in pi if d.get('Kc') is not None]
    smy = [d['vfa_med'] for d in pi if d.get('Kc') is None]

    for y, vals, colour in ((1.0, pv, C_PI), (0.0, rv, C_RL)):
        lo, hi = min(vals), max(vals)
        ax.plot([lo, hi], [y, y], color=colour, lw=1.0, alpha=0.3, zorder=1)
        m = st.median(vals)
        ax.plot([m, m], [y - 0.17, y + 0.17], color=colour, lw=1.8, zorder=4)
        ax.annotate(f'median {m:.0f}', (m, y + 0.21), ha='center',
                    fontsize=7, color=colour)

    strip(ax, 1.0, rep, C_PI, 'o',
          label=f'Conventional PI (n={len(rep) + len(smy)})')
    strip(ax, 1.0, smy, C_PI, 'o', mfc='white',
          label='gains not identified' if smy else None)
    strip(ax, 0.0, rv, C_RL, '^',
          label=f'{RL_FAMILY} seeds (n={len(rv)})')

    ax.axvline(SOFT_MG, color=C_RULE, lw=1.0, ls='--', zorder=2)
    ax.annotate('soft limit 300', (SOFT_MG - 4, 1.52), ha='right',
                fontsize=7, color=C_RULE)

    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels(['Conventional PI', f'{RL_FAMILY}\nlimit {RL_LIMIT:g}'])
    ax.set_ylim(-0.62, 1.62)
    ax.set_xlim(115, 312)
    ax.set_xlabel('Median VFA (mg/L)')
    ax.set_title('(a) one point per record',
                 loc='left')
    ax.grid(axis='x', color=C_GRID, lw=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc='lower left', frameon=False, labelspacing=0.3,
              handletextpad=0.4)


def panel_rate(ax, pi, rl):
    """The same records against the rate they were matched on.

    The constrained family is run at a cost limit, not at a rate, so its
    seeds scatter either side of the band; the ones outside it are drawn
    open, since they are not part of the matched comparison.
    """
    ax.axvspan(BAND_LO, BAND_HI, color=C_GRID, alpha=0.25, lw=0, zorder=0,
               label=f'Matched band {BAND_LO:g}–{BAND_HI:g}%')
    ax.plot([d['viol'] for d in pi], [d['vfa_med'] for d in pi],
            'o', color=C_PI, ms=4.0, ls='', mew=0.9, alpha=0.85,
            zorder=3, label='PI')
    for inside in (True, False):
        S = [d for d in rl if (BAND_LO <= d['viol'] <= BAND_HI) is inside]
        if not S:
            continue
        ax.plot([d['viol'] for d in S], [d['vfa_med'] for d in S],
                '^', color=C_RL, ms=4.0, ls='',
                mfc=C_RL if inside else 'white', mew=0.9, alpha=0.85,
                zorder=3,
                label=f'{RL_FAMILY}' if inside else 'outside the band')
    ax.axhline(SOFT_MG, color=C_RULE, lw=1.0, ls='--', zorder=2)
    ax.set_xlabel('Steps above the soft limit (%)')
    ax.set_ylabel('Median VFA (mg/L)')
    ax.set_title('(b) against the matched rate', loc='left')
    ax.grid(color=C_GRID, lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.margins(y=0.14)
    ax.legend(loc='upper left', bbox_to_anchor=(-0.02, -0.21), ncol=2,
              frameon=False, labelspacing=0.25, handletextpad=0.4,
              columnspacing=1.0, borderaxespad=0)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else PAPER
    os.makedirs(out, exist_ok=True)

    pi, rl = load_pi(), load_rl()

    fig = plt.figure(figsize=(7.2, 3.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.28)
    panel_dist(fig.add_subplot(gs[0]), pi, rl)
    panel_rate(fig.add_subplot(gs[1]), pi, rl)

    p = os.path.join(out, 'fig_vfadist.pdf')
    fig.savefig(p)
    plt.close(fig)

    pv = [d['vfa_med'] for d in pi]
    rv = [d['vfa_med'] for d in rl]
    print(f'fig_vfadist.pdf  {len(pi)} PI evaluations '
          f'({len({d["name"] for d in pi})} distinct tunings), '
          f'median VFA {min(pv):.0f}-{max(pv):.0f}, '
          f'median across records {st.median(pv):.0f} mg/L; '
          f'{RL_FAMILY} {len(rl)} seeds, {min(rv):.0f}-{max(rv):.0f}, '
          f'median {st.median(rv):.0f} mg/L')


if __name__ == '__main__':
    main()

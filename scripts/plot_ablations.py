#!/usr/bin/env python3
"""The two figures that carry Section 7, drawn from the per-run records.

fig_ablation.pdf   the class difference under every condition it was tested
                   at, as one interval per condition against a zero line
fig_hpsweep.pdf    the per-family result at each swept hyperparameter, so the
                   two exceptions the text names are visible as cells

    python3 scripts/plot_ablations.py [outdir]

``outdir`` defaults to the manuscript directory.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import random
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

ON = ('ppo', 'a2c', 'trpo', 'recurrentppo')
OFF = ('sac', 'tqc', 'ddpg', 'td3', 'crossq')
FAMS = ON + OFF
LABEL = {'ppo': 'PPO', 'a2c': 'A2C', 'trpo': 'TRPO',
         'recurrentppo': 'RecurrentPPO', 'sac': 'SAC', 'tqc': 'TQC',
         'ddpg': 'DDPG', 'td3': 'TD3', 'crossq': 'CrossQ'}

# The swept settings, in the order the sweep varies them.
SETTINGS = [('LR_1e-4', 'lr $10^{-4}$'), ('LR_1e-3', 'lr $10^{-3}$'),
            ('ARCH_64_64', 'net 64,64'), ('ARCH_400_300', 'net 400,300')]

C_ON, C_OFF, C_RULE = '#0072B2', '#D55E00', '#333333'
C_GRID = '#BBBBBB'

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

_, E = envelope()
XS = [p[0] for p in E]
YS = [p[1] for p in E]


def delta(d):
    """Methane above the conventional envelope at this run's violation rate."""
    if not (XS[0] <= d['viol'] <= XS[-1]):
        return None
    return d['ch4'] - float(np.interp(d['viol'], XS, YS))


def _keep(d, steps):
    """The main design: four swept weights, corrected settings, one budget."""
    if d.get('steps') != steps or '_h5_' in d.get('model', ''):
        return False
    if d.get('w') not in WEIGHTS:
        return False
    a = (d.get('algo') or 'sac').lower()
    want = CORRECTED.get(a)
    return not (want and not d.get('model', '').startswith(want))


def load_main(steps=150000, stacked=False):
    R = collections.defaultdict(list)
    for f in glob.glob(f'{DATA}/evres/*.json'):
        d = json.load(open(f))
        a = (d.get('algo') or 'sac').lower()
        if stacked:
            if '_h5_' not in d.get('model', ''):
                continue
        elif not _keep(d, steps):
            continue
        v = delta(d)
        if v is not None:
            R[a].append(v)
    return R


def load_sweep():
    """Per-setting, per-family deltas from the hyperparameter sweep."""
    S = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in glob.glob(f'{DATA}/evres_hp/*.json'):
        d = json.load(open(f))
        cell = d['model'].split('models_hp/')[1].split('/')[0]
        for fam in FAMS:
            if cell.startswith(fam + '_'):
                v = delta(d)
                if v is not None:
                    S[cell[len(fam) + 1:]][fam].append(v)
                break
    return S


def class_gap(per_family, rng, B=4000):
    """Difference of class medians, and its interval over resampled families."""
    on = [st.median(per_family[a]) for a in ON if per_family.get(a)]
    off = [st.median(per_family[a]) for a in OFF if per_family.get(a)]
    if not on or not off:
        return None
    point = st.median(on) - st.median(off)
    draws = sorted(st.median([rng.choice(on) for _ in on])
                   - st.median([rng.choice(off) for _ in off])
                   for _ in range(B))
    return point, draws[int(.025 * B)], draws[int(.975 * B)]


def fig_ablation(out):
    """One interval per condition the class comparison was repeated under."""
    rng = random.Random(20261001)
    rows = []

    base = load_main()
    rows.append(('Reported setting, 150k steps', class_gap(base, rng)))
    rows.append(('Doubled budget, 300k steps',
                 class_gap(load_main(300000), rng)))

    sweep = load_sweep()
    for key, label in SETTINGS:
        rows.append((label, class_gap(sweep[key], rng)))

    rows = [(lab, g) for lab, g in rows if g]
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    y = np.arange(len(rows))[::-1]

    ax.axvline(0, color=C_RULE, lw=1.0, zorder=1)
    for yi, (_, (p, lo, hi)) in zip(y, rows):
        # An interval clear of zero is the claim; one crossing it is not, and
        # the figure should not hide which is which.
        clear = lo > 0
        col = C_ON if clear else '#9A9A9A'
        ax.plot([lo, hi], [yi, yi], color=col, lw=1.6, solid_capstyle='butt',
                zorder=2)
        ax.plot([p], [yi], 'o', color=col, ms=5, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([lab for lab, _ in rows])
    ax.set_xlabel('On-policy minus off-policy class median (m$^3$/d)')
    ax.grid(axis='x', color=C_GRID, lw=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    ax.margins(y=0.12)

    h = [plt.Line2D([], [], color=C_ON, lw=1.6, marker='o', ms=5),
         plt.Line2D([], [], color='#9A9A9A', lw=1.6, marker='o', ms=5)]
    ax.legend(h, ['Interval excludes zero', 'Interval includes zero'],
              loc='upper left', bbox_to_anchor=(0, -0.28), ncol=2,
              frameon=False, handlelength=1.6)

    p = os.path.join(out, 'fig_ablation.pdf')
    fig.savefig(p)
    plt.close(fig)
    print(f'fig_ablation.pdf  {len(rows)} conditions, '
          f'{sum(1 for _, (_, lo, _) in rows if lo > 0)} with intervals above zero')


def fig_hpsweep(out):
    """Family by swept setting, so the two exceptions are visible as cells."""
    sweep = load_sweep()
    base = load_main()

    cols = ['Reported'] + [lab for _, lab in SETTINGS]
    M = np.full((len(FAMS), len(cols)), np.nan)
    for i, fam in enumerate(FAMS):
        if base.get(fam):
            M[i, 0] = st.median(base[fam])
        for j, (key, _) in enumerate(SETTINGS, start=1):
            v = sweep[key].get(fam)
            if v:
                M[i, j] = st.median(v)

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    # Diverging about zero, the line the whole paper is read against, and
    # symmetric so equal distances above and below read alike. One cell
    # (A2C at the low learning rate) is far enough out to flatten every
    # other cell if it set the scale, so the scale is clipped to the rest
    # and that cell's own number carries its size.
    lim = float(np.nanpercentile(np.abs(M), 90))
    im = ax.imshow(M, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')

    for i in range(len(FAMS)):
        for j in range(len(cols)):
            if np.isnan(M[i, j]):
                continue
            # White on the saturated ends, black in the pale middle.
            shade = 'white' if abs(M[i, j]) > 0.55 * lim else 'black'
            ax.text(j, i, f'{M[i, j]:+.0f}', ha='center', va='center',
                    color=shade, fontsize=7)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(len(FAMS)))
    ax.set_yticklabels([LABEL[f] for f in FAMS])
    # The class boundary: everything above it is on-policy.
    ax.axhline(len(ON) - 0.5, color='black', lw=1.2)
    ax.text(-0.32, (len(ON) - 1) / 2, 'on-policy', rotation=90,
            va='center', ha='center', fontsize=7,
            transform=ax.get_yaxis_transform())
    ax.text(-0.32, len(ON) + (len(OFF) - 1) / 2, 'off-policy', rotation=90,
            va='center', ha='center', fontsize=7,
            transform=ax.get_yaxis_transform())
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(FAMS), 1), minor=True)
    ax.grid(which='minor', color='white', lw=0.8)
    ax.tick_params(which='minor', length=0)

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.set_label('Median difference from the envelope (m$^3$/d)', fontsize=7)
    cb.ax.tick_params(labelsize=7)

    p = os.path.join(out, 'fig_hpsweep.pdf')
    fig.savefig(p)
    plt.close(fig)
    pos_off = sum(1 for i, f in enumerate(FAMS) if f in OFF
                  for j in range(1, len(cols)) if M[i, j] > 0)
    print(f'fig_hpsweep.pdf  {len(FAMS)} families x {len(cols)} settings, '
          f'{pos_off} off-policy swept cells above the envelope')


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else PAPER
    os.makedirs(out, exist_ok=True)
    fig_ablation(out)
    fig_hpsweep(out)


if __name__ == '__main__':
    main()

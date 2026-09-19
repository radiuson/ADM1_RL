"""Figures for the production-constraint comparison.

Two figures: the frontier each controller family traces, and the mechanism
behind the gap between them.
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'

import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
LIMIT = 300.0                      # mg/L as acetic acid

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 8.5,
    'axes.titlesize': 9,
    'axes.linewidth': 0.7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'legend.fontsize': 7.5,
    'legend.frameon': False,
    'lines.linewidth': 1.3,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

C_RL, C_PI, C_RULE, C_CONST = '#1b4f72', '#c0620f', '#4a7c3f', '#7d7d7d'

# Measured frontiers -----------------------------------------------------------
RL = [(31.2, 2024), (26.6, 1968), (18.3, 1882), (13.5, 1856), (6.7, 1826),
      (4.7, 1811), (3.6, 1807), (1.3, 1729), (1.0, 1753)]
CONST = [(0.2, 794), (0.5, 1025), (2.6, 1181), (7.6, 1353), (10.0, 1509),
         (15.7, 1659), (20.1, 1820)]
RULE = [(1.4, 1377), (1.7, 1547), (3.1, 1646), (5.0, 1688), (16.4, 1751),
        (20.9, 1789)]
PI = [(8.8, 1628), (11.4, 1695), (14.7, 1739), (18.3, 1758), (20.2, 1774),
      (20.9, 1795)]

# Margin use, from the matched-rate comparison ---------------------------------
BAND = [('RL  w=20', 1.0, 27.9, C_RL), ('Rule 150', 1.4, 1.7, C_RULE),
        ('Constant 94', 2.6, 6.7, C_CONST), ('RL  w=1', 13.5, 34.8, C_RL),
        ('PI  sp150', 11.4, 12.1, C_PI), ('RL  w=0.5', 18.3, 28.3, C_RL),
        ('PI  sp250', 18.3, 7.8, C_PI), ('Constant 159', 20.1, 5.3, C_CONST)]


def _sorted(pairs):
    a = np.array(sorted(pairs))
    return a[:, 0], a[:, 1]


def figure_frontier():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9),
                                  gridspec_kw={'width_ratios': [1.25, 1]})
    for pts, c, m, lab in ((CONST, C_CONST, 's', 'Constant feed'),
                           (RULE, C_RULE, '^', 'Rule-based'),
                           (PI, C_PI, 'D', 'PI on VFA'),
                           (RL, C_RL, 'o', 'Learned policy')):
        x, y = _sorted(pts)
        ax.plot(x, y, marker=m, color=c, label=lab, markersize=3.6,
                markerfacecolor='white' if c != C_RL else c,
                markeredgewidth=1.0, zorder=3 if c == C_RL else 2)
    ax.set_xlabel('Time above the 300 mg/L action level  (%)')
    ax.set_ylabel('Methane production  (m$^3$ d$^{-1}$)')
    ax.set_xlim(0, 33)
    ax.legend(loc='lower right')
    ax.text(-0.16, 1.02, 'a', transform=ax.transAxes, fontweight='bold',
            fontsize=10, va='bottom')

    rx, ry = _sorted(RL)
    for pts, c, m, lab in ((CONST, C_CONST, 's', 'Constant feed'),
                           (RULE, C_RULE, '^', 'Rule-based'),
                           (PI, C_PI, 'D', 'PI on VFA')):
        x, y = _sorted(pts)
        keep = (x >= rx.min()) & (x <= rx.max())
        gain = 100 * (np.interp(x[keep], rx, ry) - y[keep]) / y[keep]
        ax2.plot(x[keep], gain, marker=m, color=c, label=lab, markersize=3.6,
                 markerfacecolor='white', markeredgewidth=1.0, linestyle='--')
    ax2.axhline(0, color='k', lw=0.6)
    ax2.set_xlabel('Time above the action level  (%)')
    ax2.set_ylabel('Production gain of the learned\npolicy at matched rate  (%)')
    ax2.set_xlim(0, 22)
    ax2.text(-0.20, 1.02, 'b', transform=ax2.transAxes, fontweight='bold',
             fontsize=10, va='bottom')
    fig.tight_layout(w_pad=2.0)
    fig.savefig(HERE / 'fig_frontier.png')
    plt.close(fig)


def figure_mechanism(data):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.3),
                                  gridspec_kw={'width_ratios': [1.05, 1]})
    edges = np.arange(0, 725, 35)
    series = (('Learned policy  (w = 0.5)', data['rl_w05'], C_RL, '-'),
              ('PI on VFA  (setpoint 250)', data['pi_sp250'], C_PI, '--'))
    ax.axvspan(0.7 * LIMIT, LIMIT, color='#ececec', zorder=0, lw=0)
    peak = 0.0
    for lab, v, c, ls in series:
        v = np.asarray(v)
        h, _ = np.histogram(v, bins=edges)
        h = 100 * h / h.sum()
        peak = max(peak, h.max())
        ax.step(np.r_[edges[:-1], edges[-1]], np.r_[h, h[-1]], where='post',
                color=c, ls=ls, label=lab, zorder=3)
    ax.axvline(LIMIT, color='k', lw=0.8, ls=':')
    top = peak * 1.12
    ax.text(0.85 * LIMIT, peak * 1.02, 'margin band', ha='center',
            fontsize=6.8, color='0.40')
    ax.annotate('action level', xy=(LIMIT, peak * 0.62),
                xytext=(LIMIT + 118, peak * 0.62), fontsize=6.8,
                va='center',
                arrowprops=dict(arrowstyle='-', lw=0.6, color='0.35'))
    ax.set_xlim(0, 700)
    ax.set_ylim(0, top)
    ax.set_xlabel('VFA  (mg L$^{-1}$ as acetic acid)')
    ax.set_ylabel('Share of operating time  (%)')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=1,
              handlelength=1.8)
    ax.text(-0.17, 1.15, 'a', transform=ax.transAxes, fontweight='bold',
            fontsize=10, va='bottom')

    labs = [b[0] for b in BAND]
    vals = [b[2] for b in BAND]
    cols = [b[3] for b in BAND]
    ypos = np.arange(len(labs))[::-1]
    ax2.barh(ypos, vals, color=cols, height=0.68)
    for y, v in zip(ypos, vals):
        ax2.text(v + 0.8, y, f'{v:.1f}', va='center', fontsize=6.8)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([f'{l}   ({r:.1f} %)' for l, r in
                         zip(labs, [b[1] for b in BAND])])
    ax2.set_xlabel('Operating time within\nthe margin band  (%)')
    ax2.set_xlim(0, 45)
    ax2.tick_params(axis='y', length=0)
    ax2.text(-0.62, 1.15, 'b', transform=ax2.transAxes, fontweight='bold',
             fontsize=10, va='bottom')
    fig.tight_layout(w_pad=2.4)
    fig.savefig(HERE / 'fig_mechanism.png')
    plt.close(fig)


if __name__ == '__main__':
    figure_frontier()
    print('wrote fig_frontier.png')
    p = Path('/tmp/figdata.pkl')
    if p.exists():
        figure_mechanism(pickle.load(open(p, 'rb')))
        print('wrote fig_mechanism.png')
    else:
        print('fig_mechanism skipped: /tmp/figdata.pkl not ready')

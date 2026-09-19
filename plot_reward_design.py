#!/usr/bin/env python3
"""Reward design figure: equation breakdown + VFA & pH penalty curves."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

FIGDIR = Path(__file__).parent / 'results/eval_planA/figures'
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family':    'DejaVu Sans',
    'font.size':      9,
    'axes.linewidth': 0.8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

GREEN_BG  = '#e8f5e9'
GREEN_BD  = '#4caf50'
BLUE_1    = '#1565c0'   # ST-SAC
ORANGE_1  = '#e65100'   # MS-SAC
SAFE_COL  = '#e8f5e9'
TEXT_DARK = '#1a2332'

# ── figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(9.8, 4.8))

outer = FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                        boxstyle='round,pad=0.02',
                        linewidth=1.5, edgecolor=GREEN_BD,
                        facecolor=GREEN_BG,
                        transform=fig.transFigure, zorder=0)
fig.add_artist(outer)

# title only over left panel
fig.text(0.22, 0.975, 'Reward Function Design',
         ha='center', va='top', fontsize=11, fontweight='bold', color=TEXT_DARK)

# ── LEFT PANEL: equation ─────────────────────────────────────────────────────
ax_eq = fig.add_axes([0.03, 0.06, 0.41, 0.87])
ax_eq.set_xlim(0, 1); ax_eq.set_ylim(0, 1); ax_eq.axis('off')

def txt(ax, x, y, s, **kw):
    ax.text(x, y, s, transform=ax.transAxes, va='top', **kw)

# total reward box
ax_eq.add_patch(FancyBboxPatch((0.0, 0.83), 1.0, 0.16,
    boxstyle='round,pad=0.02', linewidth=1.0,
    edgecolor='#1565c0', facecolor='#e3f2fd', zorder=2))
txt(ax_eq, 0.5, 0.99,
    r'$r = r_{\mathrm{prod}} + r_{\mathrm{safety}} + r_{\mathrm{energy}} + r_{\mathrm{stab}}$',
    ha='center', fontsize=10.5, fontweight='bold', color=BLUE_1)

# r_prod
ax_eq.add_patch(FancyBboxPatch((0.0, 0.71), 1.0, 0.10,
    boxstyle='round,pad=0.015', linewidth=0.7,
    edgecolor='#bdbdbd', facecolor='#e8f5e9', zorder=2))
txt(ax_eq, 0.02, 0.810, r'$r_{\mathrm{prod}}$',
    ha='left', fontsize=8.5, fontweight='bold', color=TEXT_DARK)
txt(ax_eq, 0.28, 0.810, r'$= q_{\mathrm{CH_4}} / 2000$',
    ha='left', fontsize=8.5, color=TEXT_DARK)
txt(ax_eq, 0.28, 0.724, r'$\approx 0.75\text{--}1.25$  (dominant positive term)',
    ha='left', fontsize=7.5, color='#546e7a', style='italic')

# r_safety (VFA + pH together, highlighted)
ax_eq.add_patch(FancyBboxPatch((0.0, 0.32), 1.0, 0.37,
    boxstyle='round,pad=0.015', linewidth=1.8,
    edgecolor=ORANGE_1, facecolor='#fff8f0', zorder=3))
txt(ax_eq, 0.02, 0.685, r'$r_{\mathrm{safety}}$',
    ha='left', fontsize=8.5, fontweight='bold', color=ORANGE_1)
# general form
txt(ax_eq, 0.02, 0.645,
    r'General form  ($x_i \in \{\mathrm{VFA, pH, NH_3}\}$):',
    ha='left', fontsize=8.0, color=TEXT_DARK)
txt(ax_eq, 0.04, 0.604,
    r'$= 0$                     if $x_i$ in safe range',
    ha='left', fontsize=8.2, color=TEXT_DARK)
txt(ax_eq, 0.04, 0.563,
    r'$= -(c_i + w_i \cdot \epsilon_i)$    otherwise',
    ha='left', fontsize=8.2, color=TEXT_DARK)
txt(ax_eq, 0.04, 0.522,
    r'$\epsilon_i = $ deviation from boundary',
    ha='left', fontsize=7.8, color='#546e7a', style='italic')

# parameter table
ax_eq.add_patch(FancyBboxPatch((0.02, 0.33), 0.96, 0.17,
    boxstyle='round,pad=0.01', linewidth=0.8,
    edgecolor='#ffcc80', facecolor='#fff3e0', zorder=4))

# table header
txt(ax_eq, 0.05, 0.498, 'Constraint',
    ha='left', fontsize=7.5, fontweight='bold', color=TEXT_DARK)
txt(ax_eq, 0.44, 0.498, 'Safe range',
    ha='left', fontsize=7.5, fontweight='bold', color=TEXT_DARK)
txt(ax_eq, 0.68, 0.498, '$w$',
    ha='left', fontsize=7.5, fontweight='bold', color=TEXT_DARK)
txt(ax_eq, 0.80, 0.498, '$c$',
    ha='left', fontsize=7.5, fontweight='bold', color=TEXT_DARK)
txt(ax_eq, 0.88, 0.498, 'note',
    ha='left', fontsize=7.5, fontweight='bold', color=TEXT_DARK)

rows_tbl = [
    (0.458, 'VFA',  r'$\leq 0.30$ kg COD m$^{-3}$', '20', '0', r'sweep axis'),
    (0.418, 'pH',   r'$[6.8,\;7.8]$',                 '8',  '0', r'fixed'),
    (0.378, r'NH$_3$', r'$\leq 0.002$ kmol m$^{-3}$', '150','0', r'rarely active'),
]
for (y, name, rng, w, c, note) in rows_tbl:
    txt(ax_eq, 0.05, y, name,    ha='left', fontsize=7.8, color=TEXT_DARK)
    txt(ax_eq, 0.44, y, rng,     ha='left', fontsize=7.5, color=TEXT_DARK)
    txt(ax_eq, 0.68, y, w,       ha='left', fontsize=7.5, color=BLUE_1, fontweight='bold')
    txt(ax_eq, 0.80, y, c,       ha='left', fontsize=7.5, color=BLUE_1, fontweight='bold')
    txt(ax_eq, 0.88, y, note,    ha='left', fontsize=7.2, color='#78909c', style='italic')

# sweep arrow label
txt(ax_eq, 0.02, 0.348,
    r'$\uparrow$ $w_{\mathrm{VFA}}$, $c_{\mathrm{VFA}}$: sweep parameters (Plan A)',
    ha='left', fontsize=7.5, color=ORANGE_1, fontweight='bold')

# r_energy and r_stab
ax_eq.add_patch(FancyBboxPatch((0.0, 0.18), 1.0, 0.12,
    boxstyle='round,pad=0.015', linewidth=0.7,
    edgecolor='#bdbdbd', facecolor='#ede7f6', zorder=2))
txt(ax_eq, 0.02, 0.295, r'$r_{\mathrm{energy}}$',
    ha='left', fontsize=8.5, fontweight='bold', color=TEXT_DARK)
txt(ax_eq, 0.28, 0.295, r'$= -(q_{\mathrm{ad}}/300)^2 \times 0.2$',
    ha='left', fontsize=8.2, color=TEXT_DARK)
txt(ax_eq, 0.28, 0.196, r'pump power cost',
    ha='left', fontsize=7.5, color='#546e7a', style='italic')

ax_eq.add_patch(FancyBboxPatch((0.0, 0.05), 1.0, 0.10,
    boxstyle='round,pad=0.015', linewidth=0.7,
    edgecolor='#bdbdbd', facecolor='#f3e5f5', zorder=2))
txt(ax_eq, 0.02, 0.145, r'$r_{\mathrm{stab}}$',
    ha='left', fontsize=8.5, fontweight='bold', color=TEXT_DARK)
txt(ax_eq, 0.28, 0.145, r'$= -|\Delta q_{\mathrm{CH_4}}|/100 \times 0.1$',
    ha='left', fontsize=8.2, color=TEXT_DARK)
txt(ax_eq, 0.28, 0.062, r'CH$_4$ volatility penalty',
    ha='left', fontsize=7.5, color='#546e7a', style='italic')

# ── RIGHT PANEL: two stacked subplots ────────────────────────────────────────
def pen_linear(x, w, c, thresh, above=True):
    if above:
        e = np.maximum(x - thresh, 0)
        return np.where(x > thresh, -(c + w*e), 0.0)
    else:
        e = np.maximum(thresh - x, 0)
        return np.where(x < thresh, -(c + w*e), 0.0)

# ── TOP: VFA ──────────────────────────────────────────────────────────────────
ax_vfa = fig.add_axes([0.50, 0.55, 0.47, 0.38])

vfa = np.linspace(0.0, 0.60, 500)
thresh_vfa = 0.30

ax_vfa.axvspan(0, thresh_vfa, color=SAFE_COL, alpha=0.7, zorder=0)
ax_vfa.axvline(thresh_vfa, color='#4caf50', lw=1.0, ls='--', zorder=1)
ax_vfa.text(thresh_vfa + 0.005, 0.08, '0.30', fontsize=7.5, color='#388e3c', va='bottom')

# ST-SAC w=20, c=0
ax_vfa.plot(vfa, pen_linear(vfa, 20, 0, thresh_vfa),
            color=BLUE_1, lw=2.2, zorder=3,
            label=r'ST-SAC: $w\!=\!20,\;c\!=\!0$')

# MS-SAC w=10, c=1
p_ms = pen_linear(vfa, 10, 1.0, thresh_vfa)
ax_vfa.plot(vfa[vfa <= thresh_vfa], p_ms[vfa <= thresh_vfa], color=ORANGE_1, lw=2.0, zorder=3)
ax_vfa.plot(vfa[vfa > thresh_vfa],  p_ms[vfa > thresh_vfa],  color=ORANGE_1, lw=2.0, zorder=3,
            label=r'MS-SAC: $w\!=\!10,\;c\!=\!1$')
ax_vfa.plot([thresh_vfa, thresh_vfa],
            [0, pen_linear(np.array([thresh_vfa+1e-6]), 10, 1.0, thresh_vfa)[0]],
            color=ORANGE_1, lw=2.0, ls='--', zorder=3)

# c=1 cliff annotation
y_cliff = pen_linear(np.array([thresh_vfa+1e-6]), 10, 1.0, thresh_vfa)[0]
ax_vfa.annotate('', xy=(thresh_vfa+0.008, y_cliff), xytext=(thresh_vfa+0.008, 0),
                arrowprops=dict(arrowstyle='<->', color=ORANGE_1, lw=1.0))
ax_vfa.text(thresh_vfa+0.016, y_cliff/2, '$c$', fontsize=8, color=ORANGE_1, va='center')

# slope label
ax_vfa.text(0.46, -1.6, 'slope $= w$', fontsize=7.5, color=BLUE_1, ha='center',
            rotation=-50)

ax_vfa.text(0.13, 0.06, 'Safe\n$r=0$', fontsize=7.5, color='#388e3c',
            ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.18', fc=SAFE_COL, ec='#4caf50', lw=0.7))

ax_vfa.set_xlim(0, 0.60); ax_vfa.set_ylim(-4.5, 0.6)
ax_vfa.set_xlabel(r'VFA  (kg COD m$^{-3}$)', fontsize=8.5)
ax_vfa.set_ylabel(r'$r_{\mathrm{safety}}$', fontsize=9)
ax_vfa.set_title('VFA Penalty', fontsize=9.5, fontweight='bold', pad=5)
ax_vfa.legend(fontsize=7.5, loc='lower left', framealpha=0.9,
              handlelength=2.0, labelspacing=0.3)
ax_vfa.grid(True, alpha=0.25, lw=0.5)
ax_vfa.tick_params(labelsize=8)

# ── BOTTOM: pH ────────────────────────────────────────────────────────────────
ax_ph = fig.add_axes([0.50, 0.09, 0.47, 0.38])

ph = np.linspace(6.0, 8.6, 500)
ph_lo, ph_hi = 6.8, 7.8

ax_ph.axvspan(ph_lo, ph_hi, color=SAFE_COL, alpha=0.7, zorder=0)
ax_ph.axvline(ph_lo, color='#4caf50', lw=1.0, ls='--', zorder=1)
ax_ph.axvline(ph_hi, color='#4caf50', lw=1.0, ls='--', zorder=1)
ax_ph.text(ph_lo - 0.04, 0.08, '6.8', fontsize=7.5, color='#388e3c', va='bottom', ha='right')
ax_ph.text(ph_hi + 0.04, 0.08, '7.8', fontsize=7.5, color='#388e3c', va='bottom', ha='left')

# ST-SAC pH: w=8, c=0 — penalty on both sides
def ph_penalty(ph_arr, w, c):
    pen = np.zeros_like(ph_arr)
    low = ph_arr < ph_lo
    high = ph_arr > ph_hi
    pen[low]  = -(c + w * (ph_lo - ph_arr[low]))
    pen[high] = -(c + w * (ph_arr[high] - ph_hi))
    return pen

ax_ph.plot(ph, ph_penalty(ph, 8, 0), color=BLUE_1, lw=2.2, zorder=3,
           label=r'ST-SAC: $w_{\mathrm{pH}}\!=\!8,\;c\!=\!0$')

ax_ph.text((ph_lo+ph_hi)/2, 0.06, 'Safe\n$r=0$', fontsize=7.5,
           color='#388e3c', ha='center', va='bottom',
           bbox=dict(boxstyle='round,pad=0.18', fc=SAFE_COL, ec='#4caf50', lw=0.7))

# slope annotation (right side)
ax_ph.text(8.25, -1.6, 'slope $= w$', fontsize=7.5, color=BLUE_1,
           ha='center', rotation=-55)

ax_ph.set_xlim(6.0, 8.6); ax_ph.set_ylim(-4.5, 0.6)
ax_ph.set_xlabel('pH', fontsize=8.5)
ax_ph.set_ylabel(r'$r_{\mathrm{safety}}$', fontsize=9)
ax_ph.set_title('pH Penalty', fontsize=9.5, fontweight='bold', pad=5)
ax_ph.legend(fontsize=7.5, loc='lower center', framealpha=0.9,
             handlelength=2.0, labelspacing=0.3)
ax_ph.grid(True, alpha=0.25, lw=0.5)
ax_ph.tick_params(labelsize=8)

# shared y-axis label note
fig.text(0.975, 0.32, 'Same y-scale', fontsize=7, color='#90a4ae',
         ha='right', va='center', rotation=90)

# ── save ─────────────────────────────────────────────────────────────────────
fig.savefig(FIGDIR / 'reward_design.pdf', bbox_inches='tight', dpi=200)
fig.savefig(FIGDIR / 'reward_design.png', bbox_inches='tight', dpi=200)
plt.close(fig)
print('Saved: reward_design.pdf / .png')

#!/usr/bin/env python3
"""Compare 13D vs 5D Pareto frontiers."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

ROOT   = Path(__file__).parent
FIGDIR = ROOT / 'results/eval_planA/figures'
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8,
})

df13 = pd.read_csv(ROOT / 'results/eval_planA/sweep_quick_clean.csv')
df5  = pd.read_csv(ROOT / 'results/eval_planA/sweep_quick_5d_clean.csv')

def pareto_frontier(ch4, vr):
    n = len(ch4); dom = np.zeros(n, bool)
    for i in range(n):
        for j in range(n):
            if i==j: continue
            if ch4[j]>=ch4[i] and vr[j]<=vr[i] and (ch4[j]>ch4[i] or vr[j]<vr[i]):
                dom[i]=True; break
    return ~dom

SCENARIOS = ['nominal', 'high_load', 'cold_winter']
LABELS    = {'nominal': 'Nominal', 'high_load': 'High-load', 'cold_winter': 'Cold-winter'}
COLOR13   = '#2e7d32'   # green  — 13D
COLOR5    = '#1565c0'   # blue   — 5D

fig, axes = plt.subplots(1, 3, figsize=(14, 5.0))
fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.20, wspace=0.30)

for col, sc in enumerate(SCENARIOS):
    ax = axes[col]
    ax.set_title(LABELS[sc], fontsize=11, pad=6)

    for tag, df, color, ls, ms in [
        ('13D (full obs.)', df13, COLOR13, '-',  'o'),
        ('5D (compact obs.)', df5, COLOR5,  '--', 's'),
    ]:
        grp = df[(df['scenario']==sc)&(df['c_vfa']==0)].groupby('w_vfa').agg(
            ch4_mean=('ch4','mean'), vr_mean=('vr','mean'),
            ch4_std=('ch4','std'),   vr_std=('vr','std'),
        ).reset_index()
        ch4, vr = grp['ch4_mean'].values, grp['vr_mean'].values * 100
        mask = pareto_frontier(ch4, vr)
        pf = grp[mask].sort_values('ch4_mean')
        px, py = pf['ch4_mean'].values, pf['vr_mean'].values * 100
        pe_vr  = pf['vr_std'].fillna(0).values * 100

        ax.plot(px, py, ls, color=color, lw=2.5, alpha=0.90,
                solid_capstyle='round', zorder=4)
        ax.scatter(px, py, color=color, s=110, zorder=6, marker=ms,
                   edgecolors='white', linewidths=1.2)
        # error bars (VR only, low-VR points)
        for i in range(len(px)):
            if py[i] < 15:
                ax.errorbar(px[i], py[i], yerr=max(pe_vr[i]*0.5, 0.5),
                            fmt='none', ecolor=color, elinewidth=1.0,
                            capsize=3, zorder=3, alpha=0.7)

    ax.set_ylim(bottom=-3)   # set before invert so axis range is correct
    ax.invert_yaxis()
    ax.set_xlabel('CH$_4$ production (m³/d)', fontsize=10)
    if col == 0:
        ax.set_ylabel('Violation Rate (%)', fontsize=10)
    ax.grid(True, alpha=0.22, lw=0.5)
    ax.tick_params(labelsize=9)

h13 = Line2D([0],[0], color=COLOR13, marker='o', markerfacecolor=COLOR13,
             markeredgecolor='white', markersize=8, lw=2.2,
             label='13D observation (full)')
h5  = Line2D([0],[0], color=COLOR5,  marker='s', markerfacecolor=COLOR5,
             markeredgecolor='white', markersize=8, lw=2.2, linestyle='--',
             label='5D observation (compact)')
fig.legend(handles=[h13, h5], loc='lower center', ncol=2,
           bbox_to_anchor=(0.48, -0.04), framealpha=0.93,
           fontsize=10, handlelength=2.2, borderpad=0.6)

for ext in ('pdf', 'png'):
    fig.savefig(FIGDIR / f'tradeoff_13d_vs_5d.{ext}', bbox_inches='tight', dpi=200)
plt.close(fig)
print("Saved: tradeoff_13d_vs_5d.pdf / .png")

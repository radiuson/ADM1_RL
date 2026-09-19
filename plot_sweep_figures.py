#!/usr/bin/env python3
"""Regenerate all sweep figures with full 5-seed data."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.interpolate import PchipInterpolator

ROOT    = Path(__file__).parent
CSV     = ROOT / 'results/eval_planA/sweep_quick_clean.csv'
FIGDIR  = ROOT / 'results/eval_planA/figures'
FIGDIR.mkdir(parents=True, exist_ok=True)

# Reference config (safety_target / ST-SAC w=20)
REF_CSV = ROOT / 'results/eval_planA/ablation_results.csv'

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 150,
})

SCENARIO_LABELS = {
    'nominal':     'Nominal',
    'high_load':   'High-load',
    'cold_winter': 'Cold-winter',
}

# ── load sweep data (w-axis only, c=0) ──────────────────────────────────────
df = pd.read_csv(CSV)
dfw = df[df['c_vfa'] == 0].copy()

grp = dfw.groupby(['config', 'w_vfa', 'scenario']).agg(
    vr_mean  = ('vr',    'mean'),
    vr_std   = ('vr',    'std'),
    ch4_mean = ('ch4',   'mean'),
    ch4_std  = ('ch4',   'std'),
    n_seeds  = ('seed',  'nunique'),
).reset_index()

# ── load reference (ST-SAC w=20) if available ───────────────────────────────
ref_df = None
if REF_CSV.exists():
    ref_raw = pd.read_csv(REF_CSV)
    if 'config' in ref_raw.columns and 'safety_target' in ref_raw['config'].values:
        ref_df = ref_raw[ref_raw['config'] == 'safety_target'].groupby('scenario').agg(
            vr_mean  = ('vr',   'mean'),
            ch4_mean = ('ch4',  'mean'),
            n_seeds  = ('seed', 'nunique'),
        ).reset_index()


# ════════════════════════════════════════════════════════════════════════════
# Figure 1: 2×3 panel — all 3 scenarios, all w values
# ════════════════════════════════════════════════════════════════════════════
SCENARIOS = ['nominal', 'high_load', 'cold_winter']
fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex='col')
fig.suptitle('w-axis sweep: VFA penalty weight vs performance', fontsize=13, y=1.01)

for col, sc in enumerate(SCENARIOS):
    sc_grp = grp[grp['scenario'] == sc].sort_values('w_vfa')
    ws  = sc_grp['w_vfa'].values
    vr  = sc_grp['vr_mean'].values * 100
    vr_s = sc_grp['vr_std'].fillna(0).values * 100
    ch4 = sc_grp['ch4_mean'].values
    ch4_s = sc_grp['ch4_std'].fillna(0).values
    n_s = sc_grp['n_seeds'].values

    # colour by seed count
    colors = ['#1f77b4' if n == 5 else '#aec7e8' for n in n_s]
    markers = ['o' if n == 5 else 'o' for n in n_s]

    for ax_row, (yvals, ystd, ylabel, ylim) in enumerate([
        (vr,  vr_s,  'Violation rate VR (%)', (-5, 85)),
        (ch4, ch4_s, 'CH₄ production (m³/d)', (0, 3500)),
    ]):
        ax = axes[ax_row, col]
        # shaded band
        ax.fill_between(ws, yvals - ystd, yvals + ystd, alpha=0.18, color='#1f77b4')
        ax.plot(ws, yvals, '-', color='#aec7e8', lw=1.2, zorder=1)
        # scatter with per-point colour
        for x, y, c in zip(ws, yvals, colors):
            ax.scatter(x, y, color=c, s=50, zorder=3, linewidths=0.5, edgecolors='#333')

        # error bars
        ax.errorbar(ws, yvals, yerr=ystd, fmt='none', ecolor='#888', elinewidth=0.8, capsize=2, zorder=2)

        # reference star
        if ref_df is not None:
            ref_row = ref_df[ref_df['scenario'] == sc]
            if not ref_row.empty:
                ref_val = ref_row['vr_mean'].iloc[0]*100 if ax_row==0 else ref_row['ch4_mean'].iloc[0]
                ax.scatter(20, ref_val, marker='*', s=200, color='goldenrod',
                           zorder=5, linewidths=0.5, edgecolors='#333')

        ax.set_xlim(0, 65)
        ax.set_ylim(ylim)
        ax.axvline(20, color='goldenrod', lw=1, ls='--', alpha=0.6)
        ax.grid(True, alpha=0.3, lw=0.5)
        if ax_row == 0:
            ax.set_title(SCENARIO_LABELS[sc])
        if ax_row == 1:
            ax.set_xlabel('VFA penalty weight $w_{\\mathrm{VFA}}$')
        if col == 0:
            ax.set_ylabel(ylabel)

# legend
solid = Line2D([0],[0], marker='o', color='w', markerfacecolor='#1f77b4',
               markeredgecolor='#333', markersize=8, label='5 seeds')
light = Line2D([0],[0], marker='o', color='w', markerfacecolor='#aec7e8',
               markeredgecolor='#333', markersize=8, label='3 seeds')
star  = Line2D([0],[0], marker='*', color='w', markerfacecolor='goldenrod',
               markeredgecolor='#333', markersize=12, label='$w=20$ ref (15 seeds)')
fig.legend(handles=[solid, light, star], loc='lower center', ncol=3,
           bbox_to_anchor=(0.5, -0.04), framealpha=0.9)

fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(FIGDIR / f'w_sweep_all_scenarios.{ext}', bbox_inches='tight')
plt.close(fig)
print("Saved: w_sweep_all_scenarios")


# ════════════════════════════════════════════════════════════════════════════
# Figure 2: bubble chart — nominal, x=w, y=VR%, size=CH4
# ════════════════════════════════════════════════════════════════════════════
nom = grp[grp['scenario'] == 'nominal'].sort_values('w_vfa').copy()

# bubble size mapping
ch4_min, ch4_max = 600, 2800
smin, smax = 80, 900
ch4v = nom['ch4_mean'].values.clip(ch4_min, ch4_max)
sizes = smin + (ch4v - ch4_min) / (ch4_max - ch4_min) * (smax - smin)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title('Nominal scenario: violation rate vs $w_{\\mathrm{VFA}}$  '
             '(bubble size $\\propto$ CH₄ production)', pad=10)

# connect points with light line
ax.plot(nom['w_vfa'], nom['vr_mean']*100, '-', color='#aec7e8', lw=1.0, zorder=1)

for _, row in nom.iterrows():
    fc = '#1f77b4' if row['n_seeds'] == 5 else '#aec7e8'
    alpha = 0.85 if row['n_seeds'] == 5 else 0.55
    ax.errorbar(row['w_vfa'], row['vr_mean']*100,
                yerr=row['vr_std']*100 if not np.isnan(row['vr_std']) else 0,
                fmt='none', ecolor='#888', elinewidth=0.8, capsize=2, zorder=2)
    idx = nom[nom['w_vfa'] == row['w_vfa']].index[0]
    ax.scatter(row['w_vfa'], row['vr_mean']*100,
               s=sizes[nom.index.get_loc(idx)],
               color=fc, alpha=alpha, linewidths=0.6, edgecolors='#333', zorder=3)
    # label: w= and CH4
    ax.annotate(f"w={int(row['w_vfa'])}\n{int(row['ch4_mean'])}",
                xy=(row['w_vfa'], row['vr_mean']*100),
                xytext=(4, 4), textcoords='offset points',
                fontsize=7.5, color='#333')

# reference star
if ref_df is not None:
    ref_row = ref_df[ref_df['scenario'] == 'nominal']
    if not ref_row.empty:
        ax.scatter(20, ref_row['vr_mean'].iloc[0]*100,
                   marker='*', s=250, color='goldenrod',
                   zorder=5, linewidths=0.5, edgecolors='#333')

ax.axvline(20, color='goldenrod', lw=1.2, ls='--', alpha=0.7, label='ST-SAC ($w=20$)')
ax.set_xlabel('VFA penalty weight $w_{\\mathrm{VFA}}$')
ax.set_ylabel('Violation rate VR (%)')
ax.set_xlim(0, 65)
ax.set_ylim(-5, 80)
ax.grid(True, alpha=0.3, lw=0.5)

# bubble size legend
legend_ch4 = [800, 1500, 2200]
legend_handles = []
for lch4 in legend_ch4:
    s = smin + (lch4 - ch4_min) / (ch4_max - ch4_min) * (smax - smin)
    h = ax.scatter([], [], s=s, color='#1f77b4', alpha=0.7,
                   linewidths=0.5, edgecolors='#333', label=f'{lch4} m³/d')
    legend_handles.append(h)

seed_handles = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#1f77b4',
           markeredgecolor='#333', markersize=9, label='5 seeds'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#aec7e8',
           markeredgecolor='#333', markersize=9, label='3 seeds'),
    Line2D([0],[0], marker='*', color='w', markerfacecolor='goldenrod',
           markeredgecolor='#333', markersize=13, label='$w=20$ ref (15 seeds)'),
]

leg1 = ax.legend(handles=legend_handles, title='CH₄ (bubble size)',
                 loc='upper right', framealpha=0.9, fontsize=9)
ax.add_artist(leg1)
ax.legend(handles=seed_handles, loc='upper center', framealpha=0.9, fontsize=9)

fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(FIGDIR / f'w_nominal_bubble.{ext}', bbox_inches='tight')
plt.close(fig)
print("Saved: w_nominal_bubble")


# ════════════════════════════════════════════════════════════════════════════
# Figure 3: two-panel VR + CH4, monotone subset, nominal only
# ════════════════════════════════════════════════════════════════════════════
nom_all = grp[grp['scenario'] == 'nominal'].sort_values('w_vfa').copy()

# identify monotone (strictly decreasing VR) subset
vr_vals = nom_all['vr_mean'].values * 100
ws_all  = nom_all['w_vfa'].values

mono_mask = np.zeros(len(ws_all), dtype=bool)
cur_min = np.inf
for i in range(len(ws_all)):
    if vr_vals[i] < cur_min:
        cur_min = vr_vals[i]
        mono_mask[i] = True

nom_mono = nom_all[mono_mask].copy()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
fig.suptitle('Nominal scenario: monotone VR subset', fontsize=12)

for ax, col, ylabel, ylim, title in [
    (ax1, 'vr_mean',  'Violation rate VR (%)',   (-2, 70), 'Violation Rate'),
    (ax2, 'ch4_mean', 'CH₄ production (m³/d)', (0, 3000), 'CH₄ Production'),
]:
    ys  = nom_mono[col].values * (100 if 'vr' in col else 1)
    ys_std = nom_mono[col.replace('mean','std')].fillna(0).values * (100 if 'vr' in col else 1)
    xs  = nom_mono['w_vfa'].values

    # PCHIP smooth
    if len(xs) >= 3:
        xs_fine = np.linspace(xs.min(), xs.max(), 300)
        ys_fine = PchipInterpolator(xs, ys)(xs_fine)
        ax.plot(xs_fine, ys_fine, '-', color='#1f77b4', lw=1.8, zorder=2)

    ax.fill_between(xs, ys - ys_std, ys + ys_std, alpha=0.2, color='#1f77b4')
    ax.scatter(xs, ys, color='#1f77b4', s=70, zorder=4, linewidths=0.5, edgecolors='#333')

    # reference
    if ref_df is not None:
        ref_row = ref_df[ref_df['scenario'] == 'nominal']
        if not ref_row.empty:
            ref_val = ref_row['vr_mean'].iloc[0]*100 if 'vr' in col else ref_row['ch4_mean'].iloc[0]
            ax.scatter(20, ref_val, marker='*', s=200, color='goldenrod',
                       zorder=5, linewidths=0.5, edgecolors='#333', label='$w=20$ ref')

    ax.axvline(20, color='goldenrod', lw=1.2, ls='--', alpha=0.7)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3, lw=0.5)

ax2.set_xlabel('VFA penalty weight $w_{\\mathrm{VFA}}$')
ax1.legend(loc='upper right', fontsize=9)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(FIGDIR / f'w_vr_ch4_monotone.{ext}', bbox_inches='tight')
plt.close(fig)
print("Saved: w_vr_ch4_monotone")

print("\nAll figures regenerated in:", FIGDIR)

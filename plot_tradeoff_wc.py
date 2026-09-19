#!/usr/bin/env python3
"""
Safety-production tradeoff figure.
Narrative: reward parameters w and c guide the agent along the Pareto frontier.
Layout: 1×3 scatter (CH4 vs VR%), one panel per scenario.
  - background: all w-sweep points, blue gradient by log(w)
  - foreground: Pareto frontier (green line)
  - overlay:    c-sweep points (orange triangles)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from pathlib import Path

ROOT   = Path(__file__).parent
CSV    = ROOT / 'results/eval_planA/sweep_quick_clean.csv'
FIGDIR = ROOT / 'results/eval_planA/figures'
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
})

GREEN  = '#2e7d32'
ORANGE = '#e65100'
LBLUE  = '#90caf9'

# ── data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV)

grp = df.groupby(['w_vfa', 'c_vfa', 'scenario']).agg(
    vr_mean  = ('vr',  'mean'),
    vr_std   = ('vr',  'std'),
    ch4_mean = ('ch4', 'mean'),
    ch4_std  = ('ch4', 'std'),
).reset_index()

grp_w = grp[grp['c_vfa'] == 0].copy()
grp_c = grp[grp['w_vfa'] == 20].sort_values('c_vfa').copy()


# ── Pareto frontier (maximize CH4, minimize VR) ───────────────────────────────
def pareto_frontier(ch4_arr, vr_arr):
    """Return boolean mask of non-dominated points."""
    n = len(ch4_arr)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if ch4[j]>=ch4[i] and vr[j]<=vr[i] (one strict)
            if (ch4_arr[j] >= ch4_arr[i] and vr_arr[j] <= vr_arr[i] and
                    (ch4_arr[j] > ch4_arr[i] or vr_arr[j] < vr_arr[i])):
                dominated[i] = True
                break
    return ~dominated


# ── colour map for w ─────────────────────────────────────────────────────────
W_CMAP = plt.cm.Blues
w_all  = grp_w['w_vfa'].unique()
w_min, w_max = np.log(w_all.min()), np.log(w_all.max())


def w_color(w):
    norm = (np.log(w) - w_min) / (w_max - w_min)
    return W_CMAP(0.25 + 0.65 * norm)


SCENARIOS = ['nominal', 'high_load', 'cold_winter']
LABELS    = {'nominal': 'Nominal', 'high_load': 'High-load', 'cold_winter': 'Cold-winter'}

# ── baseline data ─────────────────────────────────────────────────────────────
BASELINE_CSV = ROOT / 'results/baselines_planA/full_evaluation_results.csv'
bl_df = pd.read_csv(BASELINE_CSV)
bl_df = bl_df[bl_df['scenario'].isin(SCENARIOS)].copy()
bl_df['vr_pct'] = bl_df['violation_rate'] * 100

BASELINE_STYLE = {
    'MPC':         dict(marker='*', color='#6a1b9a', s=120, zorder=7, label='MPC'),
    'Constant':    dict(marker='s', color='#37474f', s=70,  zorder=7, label='Constant'),
    'CascadedPID': dict(marker='D', color='#795548', s=70,  zorder=7, label='Cascaded PID'),
    'PID':         dict(marker='v', color='#c62828', s=80,  zorder=7, label='PID'),
}


# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
fig.subplots_adjust(left=0.07, right=0.88, top=0.93, bottom=0.18, wspace=0.30)

for col, sc in enumerate(SCENARIOS):
    ax = axes[col]
    ax.set_title(LABELS[sc], fontsize=11, pad=6)

    sw = grp_w[grp_w['scenario'] == sc].copy()
    ws     = sw['w_vfa'].values.astype(float)
    ch4_w  = sw['ch4_mean'].values
    vr_w   = sw['vr_mean'].values * 100
    e_ch4  = sw['ch4_std'].fillna(0).values
    e_vr   = sw['vr_std'].fillna(0).values * 100

    # ── Pareto frontier only (no background scatter) ──────────────────────
    mask = pareto_frontier(ch4_w, vr_w)
    pf_ch4 = ch4_w[mask]
    pf_vr  = vr_w[mask]
    pf_w   = ws[mask]
    pf_e_vr = e_vr[mask]

    sort_idx = np.argsort(pf_ch4)
    pf_ch4, pf_vr, pf_w, pf_e_vr = (
        pf_ch4[sort_idx], pf_vr[sort_idx],
        pf_w[sort_idx],   pf_e_vr[sort_idx],
    )

    ax.plot(pf_ch4, pf_vr, '-', color=GREEN, lw=2.2, zorder=4,
            alpha=0.85, solid_capstyle='round')
    # error bars: VR direction only, 0.5×std, only for VR < 15%
    for i_p in range(len(pf_ch4)):
        if pf_vr[i_p] < 15:
            ax.errorbar(pf_ch4[i_p], pf_vr[i_p],
                        yerr=pf_e_vr[i_p] * 0.5,
                        fmt='none', ecolor='#81c784', elinewidth=1.0,
                        capsize=3, zorder=5, alpha=0.85)
    # frontier points: blue gradient by w, green edge ring
    for i_p in range(len(pf_ch4)):
        ax.scatter(pf_ch4[i_p], pf_vr[i_p],
                   color=w_color(pf_w[i_p]), s=130, zorder=6,
                   linewidths=1.8, edgecolors=GREEN, marker='o')

    # annotate 3 points: topmost (lowest VR), middle, bottommost (highest VR)
    n_pf = len(pf_ch4)
    ann_idx = sorted({0, n_pf // 2, n_pf - 1})
    # top frontier annotation: push left to avoid MPC/Constant overlap
    top_dx = {'nominal': -44, 'high_load': -44, 'cold_winter': -44}
    top_dy = {'nominal':   4, 'high_load':   4, 'cold_winter':   4}
    for rank, i_p in enumerate(ann_idx):
        if rank == 0:          # top-left: lowest VR
            dx, dy = top_dx[sc], top_dy[sc]
        elif rank == len(ann_idx) - 1:  # bottom-right: highest VR
            dx, dy = -4, 8
        else:                  # middle / knee
            dx, dy = 8, 4
        ax.annotate(
            f'$w$={int(pf_w[i_p])}',
            xy=(pf_ch4[i_p], pf_vr[i_p]),
            xytext=(dx, dy), textcoords='offset points',
            fontsize=8, color='#1b5e20', fontweight='bold',
        )

    # ── c-sweep points (orange triangles, no line — data too noisy) ───────
    sc_c  = grp_c[grp_c['scenario'] == sc].sort_values('c_vfa')
    cs    = sc_c['c_vfa'].values
    ch4_c = sc_c['ch4_mean'].values
    vr_c  = sc_c['vr_mean'].values * 100
    e_ch4c = sc_c['ch4_std'].fillna(0).values
    e_vrc  = sc_c['vr_std'].fillna(0).values * 100

    ax.scatter(ch4_c, vr_c, color=ORANGE, s=80, zorder=4,
               linewidths=0.7, edgecolors='#bf360c', marker='^', alpha=0.90)


    # ── baseline controllers ──────────────────────────────────────────────
    bl_offset = {
        'nominal':     {'MPC': (6, -14), 'Constant': (6, -14), 'CascadedPID': (6, 4),  'PID': (6, 4)},
        'high_load':   {'MPC': (6, -14), 'Constant': (6, 4),   'CascadedPID': (-54, 4), 'PID': (6, 4)},
        'cold_winter': {'MPC': (6,   6), 'Constant': (-54, 4), 'CascadedPID': (6, 4),  'PID': (-54, 4)},
    }
    for ctrl, style in BASELINE_STYLE.items():
        row = bl_df[(bl_df['controller'] == ctrl) & (bl_df['scenario'] == sc)]
        if row.empty:
            continue
        ch4_b = row['ch4_avg_flow'].values[0]
        vr_b  = row['vr_pct'].values[0]
        kw = {k: v for k, v in style.items() if k != 'label'}
        ax.scatter(ch4_b, vr_b, edgecolors='white', linewidths=0.6, alpha=0.92, **kw)
        dx, dy = bl_offset[sc].get(ctrl, (6, 4))
        ax.annotate(ctrl.replace('Cascaded', 'C.'),
                    xy=(ch4_b, vr_b), xytext=(dx, dy),
                    textcoords='offset points', fontsize=7.5,
                    color=style['color'], fontweight='bold')

    ax.set_xlabel('CH$_4$ production (m³/d)', fontsize=10)
    if col == 0:
        ax.set_ylabel('Violation Rate (%)', fontsize=10)
    ax.set_ylim(bottom=-3)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.22, lw=0.5)
    ax.tick_params(labelsize=9)

# ── colour bar for w ──────────────────────────────────────────────────────────
cbar_ax = fig.add_axes([0.90, 0.18, 0.018, 0.60])
sm = plt.cm.ScalarMappable(
    cmap=W_CMAP,
    norm=mcolors.LogNorm(vmin=2, vmax=60))
sm.set_array([])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label('$w_{\\mathrm{VFA}}$', fontsize=10, labelpad=4)
cb.set_ticks([2, 5, 10, 20, 40, 60])
cb.set_ticklabels(['2', '5', '10', '20', '40', '60'], fontsize=8)
cbar_ax.yaxis.set_ticks_position('right')

# ── global legend ─────────────────────────────────────────────────────────────
h_pf = Line2D([0], [0], marker='o', color=GREEN, markerfacecolor='#4a90d9',
              markeredgecolor=GREEN, markersize=9, lw=2.2,
              label='$w$-sweep Pareto frontier ($c=0$)')
h_c  = Line2D([0], [0], marker='^', color=ORANGE, markerfacecolor=ORANGE,
              markeredgecolor='#bf360c', markersize=9, lw=0,
              label='$c$-sweep ($w=20$, $c=0.25$–$2.0$)')

h_baselines = [
    Line2D([0], [0], marker=style['marker'], color='w',
           markerfacecolor=style['color'], markeredgecolor='white',
           markersize=8, lw=0, label=style['label'])
    for style in BASELINE_STYLE.values()
]

fig.legend(handles=[h_pf, h_c] + h_baselines,
           loc='lower center', ncol=3,
           bbox_to_anchor=(0.45, -0.06),
           framealpha=0.93, fontsize=9.5,
           handlelength=2.0, handletextpad=0.5, borderpad=0.6)

for ext in ('pdf', 'png'):
    fig.savefig(FIGDIR / f'tradeoff_wc_sweep.{ext}', bbox_inches='tight', dpi=200)
plt.close(fig)
print("Saved: tradeoff_wc_sweep.pdf / .png")

#!/usr/bin/env python3
"""
Plot (w_VFA, c_VFA) sweep results.

Produces (in results/eval_planA/figures/):
  sweep_w_axis.pdf   — VR%, CH4, Score vs w (c=0 fixed)
  sweep_c_axis.pdf   — VR%, CH4, Score vs c (w=20 fixed)
  sweep_heatmap.pdf  — score heatmap over available (w, c) grid
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150,
})

ROOT = Path(__file__).parent
SUMMARY = ROOT / 'results/eval_planA/sweep_summary.csv'
OUT = ROOT / 'results/eval_planA/figures'
OUT.mkdir(parents=True, exist_ok=True)

ACCENT = '#1E8FB5'     # ST-SAC blue
ACCENT2 = '#C4AD30'    # highlight reference


def load():
    df = pd.read_csv(SUMMARY)
    # Macro-average across scenarios
    return df.groupby(['label','config','w_vfa','c_vfa']).agg(
        vr=('vr_mean','mean'), vr_e=('vr_std','mean'),
        ch4=('ch4_mean','mean'), ch4_e=('ch4_std','mean'),
        score=('score_mean','mean'),
    ).reset_index()


def _ref_mark(ax, x, y, label='ST-SAC\n(reference)'):
    ax.axvline(x, color=ACCENT2, lw=0.8, ls='--', alpha=0.6)
    ax.scatter([x], [y], s=80, color=ACCENT2, zorder=5,
               edgecolor='white', linewidth=1.5, label=label)


def fig_w_axis(df):
    """Three-panel plot: VR, CH4, Score vs w_VFA (c=0 series)."""
    sub = df[df['c_vfa'] == 0.0].sort_values('w_vfa')
    if sub.empty:
        print("  [skip] w-axis: no c=0 rows found"); return

    ref = sub[sub['w_vfa'] == 20]
    ref_vr  = ref['vr'].values[0]  * 100 if len(ref) else None
    ref_ch4 = ref['ch4'].values[0]        if len(ref) else None
    ref_sc  = ref['score'].values[0]      if len(ref) else None

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    ax1, ax2, ax3 = axes

    x  = sub['w_vfa'].values
    vr = sub['vr'].values * 100
    ch4= sub['ch4'].values
    sc = sub['score'].values

    ax1.plot(x, vr,  '-o', color=ACCENT, lw=1.8, ms=5)
    ax2.plot(x, ch4, '-o', color=ACCENT, lw=1.8, ms=5)
    ax3.plot(x, sc,  '-o', color=ACCENT, lw=1.8, ms=5)

    if ref_vr is not None:
        _ref_mark(ax1, 20, ref_vr)
        _ref_mark(ax2, 20, ref_ch4)
        _ref_mark(ax3, 20, ref_sc)

    ax1.set_ylabel('Exceedance rate (VR%)')
    ax1.set_title('VFA penalty weight sweep (c=0, w varies)')
    ax1.legend(fontsize=8)

    ax2.set_ylabel('CH₄ flow rate (m³/day)')
    ax3.set_ylabel('Aggregated score')
    ax3.set_xlabel('w_VFA (linear penalty scale)')

    for ax in axes:
        ax.grid(True, alpha=0.25, ls=':')

    fig.tight_layout()
    path = OUT / 'sweep_w_axis.pdf'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path}")


def fig_c_axis(df):
    """Three-panel: VR, CH4, Score vs c_VFA (w=20 series)."""
    sub = df[df['w_vfa'] == 20].sort_values('c_vfa')
    if sub.empty:
        print("  [skip] c-axis: no w=20 rows found"); return

    ref = sub[sub['c_vfa'] == 0.0]
    ref_vr  = ref['vr'].values[0]  * 100 if len(ref) else None
    ref_ch4 = ref['ch4'].values[0]        if len(ref) else None
    ref_sc  = ref['score'].values[0]      if len(ref) else None

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    ax1, ax2, ax3 = axes

    x  = sub['c_vfa'].values
    vr = sub['vr'].values * 100
    ch4= sub['ch4'].values
    sc = sub['score'].values

    ax1.plot(x, vr,  '-o', color='#C45B30', lw=1.8, ms=5)
    ax2.plot(x, ch4, '-o', color='#C45B30', lw=1.8, ms=5)
    ax3.plot(x, sc,  '-o', color='#C45B30', lw=1.8, ms=5)

    if ref_vr is not None:
        _ref_mark(ax1, 0, ref_vr);  _ref_mark(ax2, 0, ref_ch4);  _ref_mark(ax3, 0, ref_sc)

    ax1.set_ylabel('Exceedance rate (VR%)')
    ax1.set_title('VFA constant penalty sweep (w=20, c varies)')
    ax1.legend(fontsize=8)

    ax2.set_ylabel('CH₄ flow rate (m³/day)')
    ax3.set_ylabel('Aggregated score')
    ax3.set_xlabel('c_VFA (constant penalty per violation)')

    for ax in axes:
        ax.grid(True, alpha=0.25, ls=':')

    fig.tight_layout()
    path = OUT / 'sweep_c_axis.pdf'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path}")


def fig_heatmap(df):
    """Score (and VR) on a (w, c) bubble/heatmap."""
    # Use all available (w, c) pairs
    w_vals = sorted(df['w_vfa'].unique())
    c_vals = sorted(df['c_vfa'].unique())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, metric, label, cmap in zip(
        axes,
        ['score', 'vr'],
        ['Aggregated Score', 'VR (exceedance %)'],
        ['viridis', 'YlOrRd'],
    ):
        Z = np.full((len(c_vals), len(w_vals)), np.nan)
        for i, c in enumerate(c_vals):
            for j, w in enumerate(w_vals):
                row = df[(df['w_vfa'] == w) & (df['c_vfa'] == c)]
                if not row.empty:
                    v = row[metric].values[0]
                    Z[i, j] = v * 100 if metric == 'vr' else v

        im = ax.imshow(Z, aspect='auto', origin='lower',
                       extent=[-0.5, len(w_vals)-0.5, -0.5, len(c_vals)-0.5],
                       cmap=cmap)
        ax.set_xticks(range(len(w_vals)))
        ax.set_xticklabels([f'w={w:.0f}' for w in w_vals], rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(len(c_vals)))
        ax.set_yticklabels([f'c={c:.2f}' for c in c_vals], fontsize=7)
        ax.set_title(label)
        ax.set_xlabel('w_VFA'); ax.set_ylabel('c_VFA')

        # Annotate cells
        for i in range(len(c_vals)):
            for j in range(len(w_vals)):
                v = Z[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                            fontsize=6.5, color='white' if v < np.nanmean(Z) else 'black')

        fig.colorbar(im, ax=ax, shrink=0.8)

        # Mark ST-SAC reference
        if 20 in w_vals and 0.0 in c_vals:
            jref = w_vals.index(20)
            iref = c_vals.index(0.0)
            ax.add_patch(plt.Rectangle((jref-0.5, iref-0.5), 1, 1,
                                       fill=False, edgecolor=ACCENT2, lw=2))

    fig.suptitle('(w_VFA, c_VFA) sweep — macro-average across 6 scenarios', fontsize=11)
    fig.tight_layout()
    path = OUT / 'sweep_heatmap.pdf'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path}")


if __name__ == '__main__':
    df = load()
    print(f"Loaded {len(df)} summary rows, "
          f"w vals={sorted(df['w_vfa'].unique())}, "
          f"c vals={sorted(df['c_vfa'].unique())}")
    fig_w_axis(df)
    fig_c_axis(df)
    fig_heatmap(df)

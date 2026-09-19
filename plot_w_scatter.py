#!/usr/bin/env python3
"""
CH4 production vs VR% scatter — varying w_VFA (c=0 fixed).

Data sources (all auto-detected, newest sweep data takes precedence):
  ablation_summary.csv  → w=10 (linear-only), w=20 (ST-SAC)  [3 scenarios each]
  summary_table.csv     → w=20 (ST-SAC)                       [6 scenarios]
  sweep_quick.csv       → w=2,3,5,10,15,25,30,40,50,60        [3 scenarios, checkpoints]
  sweep_summary.csv     → full sweep, 6 scenarios              [after full training]

Run:
  python3 plot_w_scatter.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

ROOT = Path(__file__).parent
OUT  = ROOT / 'results/eval_planA/figures'
OUT.mkdir(parents=True, exist_ok=True)

SC_STYLE = {
    'nominal':          {'color': '#1565C0', 'marker': 'o',  'label': 'Nominal'},
    'high_load':        {'color': '#C62828', 'marker': 's',  'label': 'High Load'},
    'low_load':         {'color': '#2E7D32', 'marker': 'D',  'label': 'Low Load'},
    'shock_load':       {'color': '#6A1B9A', 'marker': '^',  'label': 'Shock Load'},
    'temperature_drop': {'color': '#E65100', 'marker': 'v',  'label': 'Temp Drop'},
    'cold_winter':      {'color': '#006064', 'marker': 'P',  'label': 'Cold Winter'},
}


def collect_w_data():
    """
    Build a unified DataFrame: columns = [w_vfa, c_vfa, scenario, vr_mean, ch4_mean].
    Priority: sweep_summary > sweep_quick > ablation/main.
    """
    frames = []

    # ── 1. Ablation data  ──────────────────────────────────────────────────────
    abl_csv = ROOT / 'results/eval_planA/ablation_summary.csv'
    if abl_csv.exists():
        abl = pd.read_csv(abl_csv)
        # Linear-only → w=10, c=0
        lin = abl[abl['config'].str.contains('Linear-only', na=False)].copy()
        lin['w_vfa'] = 10.0; lin['c_vfa'] = 0.0
        frames.append(lin[['w_vfa','c_vfa','scenario','vr_mean','ch4_mean']])
        # ST-SAC → w=20, c=0
        st = abl[abl['config'].str.contains('ST-SAC', na=False)].copy()
        st['w_vfa'] = 20.0; st['c_vfa'] = 0.0
        frames.append(st[['w_vfa','c_vfa','scenario','vr_mean','ch4_mean']])

    # ── 2. Main eval (ST-SAC, 6 scenarios) ────────────────────────────────────
    main_csv = ROOT / 'results/eval_planA/summary_table.csv'
    if main_csv.exists():
        main = pd.read_csv(main_csv)
        main = main[main['agent'] != 'agent'].copy()   # drop duplicate header rows
        main['vr_mean']  = main['vr_mean'].astype(float)
        main['ch4_mean'] = main['ch4_mean'].astype(float)
        st6 = main[main['agent'] == 'ST-SAC'].copy()
        st6['w_vfa'] = 20.0; st6['c_vfa'] = 0.0
        frames.append(st6[['w_vfa','c_vfa','scenario','vr_mean','ch4_mean']])

    # ── 3. Quick eval on sweep checkpoints ────────────────────────────────────
    qk_csv = ROOT / 'results/eval_planA/sweep_quick.csv'
    if qk_csv.exists():
        qk = pd.read_csv(qk_csv)
        # Average across seeds
        qk_g = qk.groupby(['w_vfa','c_vfa','scenario']).agg(
            vr_mean=('vr','mean'), ch4_mean=('ch4','mean')
        ).reset_index()
        frames.append(qk_g)

    # ── 4. Full sweep summary (final models, 6 scenarios) ─────────────────────
    sw_csv = ROOT / 'results/eval_planA/sweep_summary.csv'
    if sw_csv.exists():
        sw = pd.read_csv(sw_csv)
        sw_g = sw.groupby(['w_vfa','c_vfa','scenario']).agg(
            vr_mean=('vr_mean','mean'), ch4_mean=('ch4_mean','mean')
        ).reset_index()
        frames.append(sw_g)

    if not frames:
        raise FileNotFoundError("No eval CSV files found.")

    df = pd.concat(frames, ignore_index=True)
    # Deduplicate: keep last (highest-priority source wins for same w/c/scenario)
    df = df.drop_duplicates(subset=['w_vfa','c_vfa','scenario'], keep='last')
    return df


def make_scatter(df_all):
    """Main scatter: VR% (x) vs CH4 (y), colored by scenario, shaped by scenario,
    w value encoded as color intensity on a continuous scale."""

    # Only c=0 points for the w-axis plot
    df = df_all[df_all['c_vfa'] == 0.0].copy()
    w_vals = sorted(df['w_vfa'].unique())
    print(f"  w values with data: {[int(w) for w in w_vals]}")

    cmap   = plt.cm.plasma
    w_norm = plt.Normalize(vmin=0, vmax=max(w_vals) + 5)

    fig, ax = plt.subplots(figsize=(8.5, 5.8))

    leg_sc_handles = []
    leg_w_handles  = []
    plotted_sc     = set()

    for w in sorted(w_vals):
        sub     = df[df['w_vfa'] == w]
        w_color = cmap(w_norm(w))
        is_ref  = (w == 20)
        ms      = 120 if is_ref else 70
        lw_edge = 2.2 if is_ref else 0.7
        ec      = '#FFD600' if is_ref else 'white'
        zorder  = 5 if is_ref else 3

        for _, row in sub.iterrows():
            sc  = row['scenario']
            vr  = row['vr_mean'] * 100
            ch4 = row['ch4_mean']
            sty = SC_STYLE.get(sc, {'color':'#888','marker':'x','label': sc})

            ax.scatter(vr, ch4, s=ms, marker=sty['marker'],
                       color=w_color, edgecolors=ec, linewidths=lw_edge,
                       zorder=zorder)

            if sc not in plotted_sc:
                leg_sc_handles.append(
                    ax.scatter([], [], s=60, marker=sty['marker'],
                               color='#666', label=sty['label']))
                plotted_sc.add(sc)

        lbl = f'w = {int(w)}{"  ★ (ST-SAC)" if is_ref else ""}'
        leg_w_handles.append(
            ax.scatter([], [], s=65, color=w_color,
                       edgecolors='white', linewidths=0.7, label=lbl))

    # Annotate ST-SAC (w=20) reference points with scenario names
    ref = df[df['w_vfa'] == 20]
    for _, row in ref.iterrows():
        sc  = row['scenario']
        vr  = row['vr_mean'] * 100
        ch4 = row['ch4_mean']
        ax.annotate(SC_STYLE.get(sc,{}).get('label', sc), (vr, ch4),
                    xytext=(6, 3), textcoords='offset points',
                    fontsize=7.5, color='#222',
                    path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])

    # Safety target reference line
    ax.axvline(5, color='#388E3C', lw=1.0, ls='--', alpha=0.55, zorder=1)
    ylim = ax.get_ylim()
    ax.text(5.3, ylim[0] + (ylim[1]-ylim[0])*0.02,
            'VR = 5%', color='#388E3C', fontsize=7.5, va='bottom')

    # ── legends ───────────────────────────────────────────────────────────────
    leg1 = ax.legend(handles=leg_w_handles,
                     title=r'$w_{\mathrm{VFA}}$  (c = 0)',
                     loc='upper right', fontsize=8, title_fontsize=8.5,
                     framealpha=0.92, edgecolor='#ccc', ncol=2)
    ax.add_artist(leg1)
    ax.legend(handles=leg_sc_handles, title='Scenario',
              loc='lower right', fontsize=8, title_fontsize=8.5,
              framealpha=0.92, edgecolor='#ccc')

    # Colorbar for w
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=w_norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.01)
    cb.set_label(r'$w_{\mathrm{VFA}}$', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    ax.set_xlabel('VFA exceedance rate, VR (%)', fontsize=11, labelpad=6)
    ax.set_ylabel('CH₄ production (m³/day)',      fontsize=11, labelpad=6)
    ax.set_title(r'Safety–productivity trade-off vs. VFA penalty weight $w_{\mathrm{VFA}}$'
                 '\n' r'($c_{\mathrm{VFA}} = 0$, macro-average across scenarios)',
                 fontsize=11.5, pad=10)
    ax.grid(True, alpha=0.18, ls=':')
    ax.set_xlim(left=-1.0)

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        p = OUT / f'w_scatter.{ext}'
        fig.savefig(p, bbox_inches='tight', dpi=150)
        print(f"  → {p}")
    plt.close(fig)


if __name__ == '__main__':
    df_all = collect_w_data()
    print(f"Loaded {len(df_all)} data points across "
          f"{df_all['w_vfa'].nunique()} w values")
    make_scatter(df_all)

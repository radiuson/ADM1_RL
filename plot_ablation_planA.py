#!/usr/bin/env python3
"""
Ablation Figure — reward component decomposition
Produces:
  results/eval_planA/figures/ablation_2x2.pdf
  results/eval_planA/figures/ablation_bars.pdf
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150,
})

ROOT = Path(__file__).parent
SUMMARY = ROOT / 'results/eval_planA/ablation_summary.csv'
OUT     = ROOT / 'results/eval_planA/figures'
OUT.mkdir(parents=True, exist_ok=True)

# Config display order and mapping to (c_present, w_present)
CONFIG_META = {
    'Const-only (c, w=0)':     ('constant_only', True,  False),
    'Linear-only (w, c=0)':    ('linear_only',   False, True),
    'MS-SAC (w+c)':            ('ms_sac',         True,  True),
    'ST-SAC (large w, c=0)':   ('st_sac',         False, True),
}
SCENARIOS = ['nominal', 'high_load', 'cold_winter']
SC_LABELS = {'nominal':'Nominal', 'high_load':'High Load', 'cold_winter':'Cold Winter'}


def load():
    df = pd.read_csv(SUMMARY)
    # Normalize config names (remove newlines if any)
    df['config'] = df['config'].str.replace('\n', ' ')
    return df


def fig_grouped_bars(df):
    """4-config × 3-scenario grouped bars: VR% top, CH4 bottom."""
    configs   = list(CONFIG_META.keys())
    n_c, n_s  = len(configs), len(SCENARIOS)
    colors    = ['#4878CF','#5BB450','#C4AD30','#1E8FB5']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    width = 0.18
    x     = np.arange(n_s)

    for i, (cfg, color) in enumerate(zip(configs, colors)):
        sub = df[df['config'] == cfg].set_index('scenario')
        vr  = [sub.loc[s,'vr_mean']*100  if s in sub.index else np.nan for s in SCENARIOS]
        vr_e= [sub.loc[s,'vr_std']*100   if s in sub.index else 0      for s in SCENARIOS]
        ch4 = [sub.loc[s,'ch4_mean']     if s in sub.index else np.nan for s in SCENARIOS]
        ch4_e=[sub.loc[s,'ch4_std']      if s in sub.index else 0      for s in SCENARIOS]
        offset = (i - 1.5) * width

        lw   = 1.8 if 'ST-SAC' in cfg else 0.4
        ec   = color if 'ST-SAC' in cfg else 'black'
        alpha= 0.95  if 'ST-SAC' in cfg else 0.80

        ax1.bar(x+offset, vr,  width, color=color, alpha=alpha,
                edgecolor=ec, linewidth=lw, label=cfg.split('\n')[0],
                yerr=vr_e, error_kw={'elinewidth':0.8,'capsize':2})
        ax2.bar(x+offset, ch4, width, color=color, alpha=alpha,
                edgecolor=ec, linewidth=lw,
                yerr=ch4_e, error_kw={'elinewidth':0.8,'capsize':2})

    ax1.set_ylabel('Exceedance rate (%)')
    ax1.set_title('Reward component ablation — VR')
    ax1.axhline(0, color='black', lw=0.5)
    ax1.legend(loc='upper left', ncol=2, fontsize=8)

    ax2.set_ylabel('CH₄ flow rate (m³/day)')
    ax2.set_title('Reward component ablation — CH₄ production')
    ax2.set_xticks(x)
    ax2.set_xticklabels([SC_LABELS[s] for s in SCENARIOS])

    fig.tight_layout()
    path = OUT / 'ablation_bars.pdf'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path}")


def fig_2x2_matrix(df):
    """2×2 grid: rows=constant penalty (yes/no), cols=linear penalty (yes/no).
    Each cell shows macro-average VR and CH4."""
    # Map config → cell
    # (has_c, has_w) → config key
    cell_map = {
        (True,  False): 'Const-only (c, w=0)',
        (False, True):  'Linear-only (w, c=0)',
        (True,  True):  'MS-SAC (w+c)',
        (False, True):  'Linear-only (w, c=0)',  # overwritten below for ST-SAC
    }
    # ST-SAC: no c, large w — same cell as linear-only conceptually
    # Show both linear-only and ST-SAC side by side in that cell

    configs_2x2 = [
        ((True,  False), 'Const-only (c, w=0)',   'c only',     '#4878CF'),
        ((False, True),  'Linear-only (w, c=0)',   'w only',     '#5BB450'),
        ((True,  True),  'MS-SAC (w+c)',            'w + c',      '#C4AD30'),
        ((False, True),  'ST-SAC (large w, c=0)',   'w★ (large)', '#1E8FB5'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.8))
    fig.suptitle('Reward component ablation — macro-average (Nominal + High Load + Cold Winter)',
                 fontsize=11, y=1.02)

    for ax, (_, cfg, short, color) in zip(axes, configs_2x2):
        sub = df[df['config'] == cfg]
        if sub.empty:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(short)
            continue

        vr_m  = sub['vr_mean'].mean() * 100
        ch4_m = sub['ch4_mean'].mean()
        sc_m  = sub['score_mean'].mean()
        t_m   = sub['term_rate'].mean() * 100

        is_st = 'ST-SAC' in cfg
        ax.set_facecolor(color + '18')
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.0 if is_st else 0.8)

        ax.text(0.5, 0.72, f'{vr_m:.1f}%', ha='center', va='center',
                transform=ax.transAxes, fontsize=22,
                color=color, fontweight='bold')
        ax.text(0.5, 0.52, 'VR (exceedance)', ha='center', va='center',
                transform=ax.transAxes, fontsize=8, color='#666')

        ax.text(0.5, 0.36, f'{ch4_m:.0f} m³/d', ha='center', va='center',
                transform=ax.transAxes, fontsize=14, color='#333', fontweight='600')
        ax.text(0.5, 0.20, f'score {sc_m:.3f}  |  term {t_m:.0f}%',
                ha='center', va='center',
                transform=ax.transAxes, fontsize=8, color='#666')

        ax.set_title(cfg.replace(' (', '\n('), fontsize=9, color=color, pad=6)
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    path = OUT / 'ablation_2x2.pdf'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path}")


if __name__ == '__main__':
    df = load()
    print(f"Loaded {len(df)} rows, configs: {df['config'].unique().tolist()}")
    fig_grouped_bars(df)
    fig_2x2_matrix(df)

    # Print table
    macro = df.groupby('config').agg(
        VR=('vr_mean','mean'), CH4=('ch4_mean','mean'),
        Score=('score_mean','mean'), Term=('term_rate','mean')
    )
    print("\nMacro-average ablation table:")
    for cfg, row in macro.iterrows():
        print(f"  {cfg:<35s}  VR={row.VR*100:5.1f}%  CH4={row.CH4:6.0f}  "
              f"score={row.Score:.3f}  term={row.Term*100:.0f}%")

#!/usr/bin/env python3
"""
Plan A Results Figures
=======================
Generates paper-quality figures from eval_planA.py + baseline results.

Figures produced:
  1. main_table_heatmap.pdf   — VR% heatmap: agents × scenarios
  2. ch4_heatmap.pdf          — CH4 flow heatmap
  3. tradeoff_scatter.pdf     — VR vs CH4 scatter (all agents, color-coded)
  4. score_bar.pdf            — Overall score bar chart per agent
  5. scenario_comparison.pdf  — Side-by-side bar per scenario

Usage (from ADM1_RL/ directory):
    python plot_results_planA.py
    python plot_results_planA.py --eval-dir results/eval_planA \
                                  --baseline results/baselines_planA/full_evaluation_results.csv \
                                  --output-dir results/eval_planA/figures
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        10,
    'axes.titlesize':   11,
    'axes.labelsize':   10,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'legend.fontsize':  9,
    'figure.dpi':       150,
    'axes.spines.top':  False,
    'axes.spines.right': False,
})

SCENARIO_LABELS = {
    'nominal':          'Nominal',
    'high_load':        'High Load',
    'low_load':         'Low Load',
    'shock_load':       'Shock Load',
    'temperature_drop': 'Temp Drop',
    'cold_winter':      'Cold Winter',
}
SCENARIO_ORDER = list(SCENARIO_LABELS.keys())

AGENT_COLORS = {
    'Constant':     '#4878CF',
    'PID':          '#6ACC65',
    'CascadedPID':  '#D65F5F',
    'MPC':          '#B47CC7',
    'MS-SAC':       '#C4AD66',
    'ST-SAC':       '#77BEDB',
}
AGENT_ORDER = ['Constant', 'PID', 'CascadedPID', 'MPC', 'MS-SAC', 'ST-SAC']


def load_data(eval_dir: Path, baseline_csv: Path):
    """Load and merge RL + baseline results into a unified summary DataFrame."""
    # ── Baselines ──────────────────────────────────────────────────────────────
    bl = pd.read_csv(baseline_csv)
    bl_rows = []
    for _, row in bl.iterrows():
        bl_rows.append({
            'agent':    row['controller'],
            'scenario': row['scenario'],
            'vr_mean':  row['violation_rate'],
            'vr_std':   0.0,
            'ch4_mean': row['ch4_avg_flow'],
            'ch4_std':  0.0,
            'score_mean': row['overall_score'],
            'score_std':  0.0,
            'term_rate':  float(row['terminated']),
        })
    df_bl = pd.DataFrame(bl_rows)

    # ── RL agents ──────────────────────────────────────────────────────────────
    rl_summary = eval_dir / 'summary_table.csv'
    df_rl = pd.read_csv(rl_summary) if rl_summary.exists() else pd.DataFrame()

    df = pd.concat([df_bl, df_rl], ignore_index=True)
    return df


def build_pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot to agents × scenarios matrix, preserving order."""
    agents    = [a for a in AGENT_ORDER if a in df['agent'].unique()]
    scenarios = [s for s in SCENARIO_ORDER if s in df['scenario'].unique()]
    pivot = df.pivot_table(index='agent', columns='scenario', values=metric, aggfunc='mean')
    pivot = pivot.reindex(index=agents, columns=scenarios)
    return pivot


def fig_heatmap(pivot: pd.DataFrame, title: str, fmt: str, cmap, vmin, vmax,
                out_path: Path, annot_scale: float = 1.0):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    data = pivot.values.astype(float)

    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([SCENARIO_LABELS.get(c, c) for c in pivot.columns], rotation=30, ha='right')
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title, pad=8)

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                txt = 'N/A'
            else:
                txt = fmt.format(v * annot_scale)
            color = 'white' if (v * annot_scale > (vmax * annot_scale * 0.6)) else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=8, color=color)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out_path}")


def fig_tradeoff(df: pd.DataFrame, out_path: Path):
    agents = [a for a in AGENT_ORDER if a in df['agent'].unique()]
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Compute macro-average per agent
    macro = df.groupby('agent')[['vr_mean', 'ch4_mean', 'score_mean', 'term_rate']].mean().reset_index()

    for _, row in macro.iterrows():
        agent = row['agent']
        if agent not in AGENT_COLORS:
            continue
        vr  = row['vr_mean'] * 100
        ch4 = row['ch4_mean']
        color = AGENT_COLORS[agent]
        marker = '^' if agent in ('MS-SAC', 'ST-SAC') else 'o'
        size   = 120 if agent in ('MS-SAC', 'ST-SAC') else 80
        ax.scatter(vr, ch4, color=color, marker=marker, s=size, zorder=5,
                   edgecolors='black', linewidths=0.6)
        offset_x = 0.5
        offset_y = 20
        ax.annotate(agent, (vr, ch4),
                    xytext=(offset_x, offset_y), textcoords='offset points',
                    fontsize=8, color=color, fontweight='bold')

    ax.set_xlabel('Soft-constraint exceedance rate (%)')
    ax.set_ylabel('Average CH₄ flow rate (m³/day)')
    ax.set_title('Production–Safety Trade-off (macro-average over 6 scenarios)')
    ax.axvline(x=0, color='gray', lw=0.5, ls='--', alpha=0.5)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out_path}")


def fig_score_bar(df: pd.DataFrame, out_path: Path):
    agents = [a for a in AGENT_ORDER if a in df['agent'].unique()]
    macro  = df.groupby('agent')['score_mean'].mean().reindex(agents)
    macro_std = df.groupby('agent')['score_mean'].std().reindex(agents).fillna(0)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(agents))
    colors = [AGENT_COLORS.get(a, '#888') for a in agents]
    bars = ax.bar(x, macro.values, yerr=macro_std.values,
                  color=colors, edgecolor='black', linewidth=0.6,
                  error_kw={'elinewidth': 1, 'capsize': 3}, width=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=20, ha='right')
    ax.set_ylabel('Aggregated score (macro-average)')
    ax.set_title('Overall Performance Score — All Scenarios')
    ax.axhline(y=0, color='black', lw=0.5)
    ax.set_ylim(bottom=min(0, macro.min() - 0.1))

    for bar, val in zip(bars, macro.values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out_path}")


def fig_vr_per_scenario(df: pd.DataFrame, out_path: Path):
    scenarios = [s for s in SCENARIO_ORDER if s in df['scenario'].unique()]
    agents    = [a for a in AGENT_ORDER if a in df['agent'].unique()]
    n_s = len(scenarios)
    n_a = len(agents)
    width = 0.8 / n_a
    x = np.arange(n_s)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax_vr, ax_ch4 = axes

    for i, agent in enumerate(agents):
        sub  = df[df['agent'] == agent].set_index('scenario')
        vr   = [sub.loc[s, 'vr_mean'] * 100 if s in sub.index else np.nan for s in scenarios]
        ch4  = [sub.loc[s, 'ch4_mean']       if s in sub.index else np.nan for s in scenarios]
        vr_e = [sub.loc[s, 'vr_std']  * 100  if s in sub.index else 0      for s in scenarios]
        ch4_e= [sub.loc[s, 'ch4_std']        if s in sub.index else 0      for s in scenarios]
        offset = (i - n_a / 2 + 0.5) * width
        color  = AGENT_COLORS.get(agent, '#888')

        ax_vr.bar(x + offset, vr, width=width * 0.9, color=color, alpha=0.85,
                  edgecolor='black', linewidth=0.4, label=agent,
                  yerr=vr_e, error_kw={'elinewidth': 0.8, 'capsize': 2})
        ax_ch4.bar(x + offset, ch4, width=width * 0.9, color=color, alpha=0.85,
                   edgecolor='black', linewidth=0.4,
                   yerr=ch4_e, error_kw={'elinewidth': 0.8, 'capsize': 2})

    ax_vr.set_ylabel('VR — exceedance rate (%)')
    ax_vr.set_title('Soft-constraint exceedance rate per scenario')
    ax_vr.legend(loc='upper left', ncol=3, fontsize=8)
    ax_vr.axhline(0, color='black', lw=0.5)

    ax_ch4.set_ylabel('CH₄ flow rate (m³/day)')
    ax_ch4.set_title('Average methane production per scenario')
    ax_ch4.set_xticks(x)
    ax_ch4.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios], rotation=20, ha='right')

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-dir',  default='results/eval_planA')
    parser.add_argument('--baseline',  default='results/baselines_planA/full_evaluation_results.csv')
    parser.add_argument('--output-dir', default='results/eval_planA/figures')
    args = parser.parse_args()

    root    = Path(__file__).parent
    eval_dir = root / args.eval_dir
    out_dir  = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data(eval_dir, root / args.baseline)
    print(f"  {len(df)} rows  agents: {sorted(df['agent'].unique())}")

    # 1. VR heatmap
    print("Figure 1: VR heatmap")
    pivot_vr = build_pivot(df, 'vr_mean')
    cmap_vr = LinearSegmentedColormap.from_list('vr', ['#2ca02c', '#ffdd57', '#d62728'])
    fig_heatmap(pivot_vr, 'Soft-constraint exceedance rate (%)',
                '{:.1f}%', cmap_vr, 0.0, 0.8,
                out_dir / 'vr_heatmap.pdf', annot_scale=100)

    # 2. CH4 heatmap
    print("Figure 2: CH4 heatmap")
    pivot_ch4 = build_pivot(df, 'ch4_mean')
    cmap_ch4 = LinearSegmentedColormap.from_list('ch4', ['#f7f7f7', '#4393c3', '#053061'])
    fig_heatmap(pivot_ch4, 'Average CH₄ flow rate (m³/day)',
                '{:.0f}', cmap_ch4, 600, 2800,
                out_dir / 'ch4_heatmap.pdf', annot_scale=1.0)

    # 3. Trade-off scatter
    print("Figure 3: trade-off scatter")
    fig_tradeoff(df, out_dir / 'tradeoff_scatter.pdf')

    # 4. Score bar
    print("Figure 4: score bar")
    fig_score_bar(df, out_dir / 'score_bar.pdf')

    # 5. Per-scenario comparison
    print("Figure 5: per-scenario VR + CH4 bars")
    fig_vr_per_scenario(df, out_dir / 'scenario_comparison.pdf')

    # Print final summary table
    print("\n" + "="*65)
    print("  SUMMARY TABLE (macro-average)")
    print("="*65)
    macro = df.groupby('agent').agg(
        VR_pct=('vr_mean', lambda x: f"{x.mean()*100:.1f}%"),
        CH4=('ch4_mean', lambda x: f"{x.mean():.0f}"),
        Score=('score_mean', lambda x: f"{x.mean():.3f}"),
        Term=('term_rate', lambda x: f"{x.mean()*100:.0f}%"),
    ).reindex([a for a in AGENT_ORDER if a in df['agent'].unique()])
    print(macro.to_string())
    print(f"\nFigures saved to {out_dir}/")


if __name__ == '__main__':
    main()

"""
tau_a Sensitivity Analysis
===========================
Runs 30-day ADM1 simulations under Cold Winter and Nominal scenarios
with tau_a ∈ {7, 15, 30, 60} days using constant control.

Generates:
    results/tau_a_sensitivity/tau_a_sensitivity.png
    results/tau_a_sensitivity/tau_a_sensitivity.csv

Usage (from ADM1_RL/ directory):
    python scripts/tau_a_sensitivity.py
"""

import sys
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'env'))
sys.path.insert(0, str(ROOT / 'baselines'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from env.adm1_gym_env import ADM1Env_v2

# ── experiment configuration ──────────────────────────────────────────────────
TAU_A_VALUES  = [7, 15, 30, 60]          # days
SCENARIOS     = ['cold_winter', 'nominal']

# Constant-control actions per scenario  [q_ad m³/d, feed_mult, Q_HEX W]
CONSTANT_ACTION = {
    'cold_winter': np.array([178.0, 1.0, 2400.0], dtype=np.float32),
    'nominal':     np.array([178.0, 1.0,  500.0], dtype=np.float32),
}

COLORS = {7: '#E74C3C', 15: '#F39C12', 30: '#2ECC71', 60: '#3498DB'}
LABELS = {7: r'$\tau_a=7$ d', 15: r'$\tau_a=15$ d',
          30: r'$\tau_a=30$ d (paper)', 60: r'$\tau_a=60$ d'}
LINESTYLES = {7: '-', 15: '--', 30: '-', 60: ':'}
LINEWIDTHS = {7: 1.5, 15: 1.5, 30: 2.5, 60: 1.5}   # highlight the paper value


def run_episode(tau_a: float, scenario: str) -> pd.DataFrame:
    """Run one 30-day episode and return time-series DataFrame."""
    env = ADM1Env_v2(scenario_name=scenario, obs_mode='full')
    obs, _ = env.reset()

    # Override tau_a AFTER reset (so initial T_a is set correctly)
    env.solver.tau_a = float(tau_a)

    action = CONSTANT_ACTION[scenario]
    records = []

    while True:
        # Snapshot state before step
        T_L  = env.solver.T_L
        T_a  = env.solver.T_a
        F_K  = float(np.exp(-((T_L - T_a) ** 2) / (2.0 * env.solver.s_hg ** 2)))
        day  = env.current_step * env.step_size

        obs, reward, terminated, truncated, info = env.step(action)

        records.append({
            'day':     day,
            'T_L':     T_L - 273.15,           # K → °C
            'T_a':     T_a - 273.15,
            'F_K':     F_K,
            'q_ch4':   info.get('q_ch4', 0.0),
            'VFA':     info.get('total_vfa', 0.0),
            'pH':      info.get('pH', 7.0),
        })

        if terminated or truncated:
            break

    env.close()
    df = pd.DataFrame(records)
    df['tau_a']    = tau_a
    df['scenario'] = scenario
    return df


def run_all() -> pd.DataFrame:
    all_dfs = []
    for scenario in SCENARIOS:
        for tau in TAU_A_VALUES:
            print(f"  Running {scenario} | tau_a={tau:2d} d ...", end=' ', flush=True)
            df = run_episode(tau, scenario)
            all_dfs.append(df)
            print(f"done  ({len(df)} steps, "
                  f"q_ch4_mean={df['q_ch4'].mean():.0f} m³/d, "
                  f"F_K_min={df['F_K'].min():.3f})")
    return pd.concat(all_dfs, ignore_index=True)


def make_figure(data: pd.DataFrame, out_path: Path):
    """4-column × 2-row figure: [T_L/T_a | F_K | q_ch4 | VFA] × [Cold Winter | Nominal]"""
    scenario_labels = {'cold_winter': 'Cold Winter\n($T_{env}=5°C$, $T_{feed}=10°C$)',
                       'nominal':     'Nominal\n($T_{env}=25°C$, $T_{feed}=25°C$)'}

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(r'Sensitivity Analysis: Microbial Adaptation Time Constant $\tau_a$',
                 fontsize=14, fontweight='bold', y=1.00)

    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.40, wspace=0.32,
                           left=0.06, right=0.98,
                           top=0.91, bottom=0.10)

    col_titles = [r'Temperature $T_L$ and $T_a$ (°C)',
                  r'Methanogenic inhibition $F_K$',
                  r'Methane flow $q_{CH_4}$ (m³/d)',
                  r'Total VFA (kmol/m³)']

    axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(2)]

    for row, scenario in enumerate(SCENARIOS):
        df_sc = data[data['scenario'] == scenario]

        for col in range(4):
            ax = axes[row][col]

            # Row label (y-axis left side of col 0)
            if col == 0:
                ax.set_ylabel(scenario_labels[scenario], fontsize=10)

            # Column title (top of row 0)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10, pad=4)

            for tau in TAU_A_VALUES:
                df = df_sc[df_sc['tau_a'] == tau].sort_values('day')
                kw = dict(color=COLORS[tau], label=LABELS[tau],
                          ls=LINESTYLES[tau], lw=LINEWIDTHS[tau], alpha=0.9)

                if col == 0:
                    # Temperature: two lines (T_L solid, T_a dashed, same color)
                    ax.plot(df['day'], df['T_L'], **kw)
                    ax.plot(df['day'], df['T_a'],
                            color=COLORS[tau], ls=':', lw=1.2, alpha=0.6)
                    if tau == TAU_A_VALUES[0] and row == 0:
                        # Legend proxies for T_L / T_a style
                        ax.plot([], [], 'k-',  lw=1.5, label=r'$T_L$')
                        ax.plot([], [], 'k--', lw=1.2, label=r'$T_a$ (dashed)')
                elif col == 1:
                    ax.plot(df['day'], df['F_K'], **kw)
                    ax.set_ylim(-0.05, 1.05)
                    if row == 0:
                        ax.axhline(1.0, color='gray', lw=0.8, ls='--', alpha=0.5)
                elif col == 2:
                    ax.plot(df['day'], df['q_ch4'], **kw)
                elif col == 3:
                    ax.plot(df['day'], df['VFA'], **kw)
                    # hard threshold
                    ax.axhline(0.8, color='red', lw=0.8, ls='--', alpha=0.6)
                    if row == 0 and tau == TAU_A_VALUES[-1]:
                        ax.text(29, 0.81, 'hard\nlimit', color='red',
                                fontsize=7, va='bottom', ha='right')

            ax.set_xlabel('Day', fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3, lw=0.5)

    # Single legend at bottom
    handles, labels_leg = axes[0][1].get_legend_handles_labels()
    fig.legend(handles, labels_leg,
               loc='lower center', ncol=4,
               fontsize=9.5, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {out_path}")


def print_summary(data: pd.DataFrame):
    """Print key statistics for paper reporting."""
    print("\n" + "=" * 70)
    print("Key Statistics for Paper")
    print("=" * 70)
    for scenario in SCENARIOS:
        print(f"\n{scenario.upper().replace('_', ' ')}:")
        df_sc = data[data['scenario'] == scenario]
        print(f"  {'tau_a':>8} | {'F_K_min':>8} | {'F_K_t15':>8} | "
              f"{'q_ch4_mean':>10} | {'VFA_max':>8}")
        print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}")
        for tau in TAU_A_VALUES:
            df = df_sc[df_sc['tau_a'] == tau].sort_values('day')
            # F_K at ~day 15 (midpoint)
            mid = df[df['day'] >= 14.9].head(1)
            fk_t15 = mid['F_K'].values[0] if len(mid) > 0 else float('nan')
            marker = " ← paper" if tau == 30 else ""
            print(f"  {tau:>7}d | {df['F_K'].min():>8.4f} | {fk_t15:>8.4f} | "
                  f"{df['q_ch4'].mean():>10.1f} | {df['VFA'].max():>8.4f}{marker}")


def main():
    print("=" * 70)
    print(r"tau_a Sensitivity Analysis")
    print("=" * 70)

    out_dir = ROOT / 'results' / 'tau_a_sensitivity'
    out_dir.mkdir(parents=True, exist_ok=True)

    data = run_all()

    # Save raw data
    csv_path = out_dir / 'tau_a_sensitivity.csv'
    data.to_csv(csv_path, index=False)
    print(f"\n  Data saved: {csv_path}")

    make_figure(data, out_dir / 'tau_a_sensitivity.png')
    print_summary(data)

    print("\n✓ Done")


if __name__ == '__main__':
    main()

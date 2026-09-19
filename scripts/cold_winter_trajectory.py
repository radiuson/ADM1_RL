#!/usr/bin/env python3
"""
Cold Winter Trajectory Analysis
================================
Re-runs all Cold Winter baseline controllers + SAC + MPC and records
step-by-step trajectories.  Generates:

  results/cold_winter_trajectory/
      cold_winter_trajectories.png   — 3-panel figure (VFA / q_CH4 / T_L+T_a)
      cold_winter_survival.csv       — survival days per controller
      cold_winter_trajectory_data.csv

The figure directly addresses reviewer comment R1-5:
  "Explain the Cold Winter VFA termination mechanism."

Usage (from ADM1_RL/ directory):
    python scripts/cold_winter_trajectory.py
    python scripts/cold_winter_trajectory.py --no-rl   # skip SAC/MPC (fast)
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'env'))
sys.path.insert(0, str(ROOT / 'baselines'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from env.adm1_gym_env import ADM1Env_v2
from training.reward_configs import REWARD_CONFIGS

# ── Controller definitions ────────────────────────────────────────────────────

RULE_CONTROLLERS = [
    ('Constant',      'constant',      {}),
    ('PID',           'pid',           {'K_p': 0.5, 'K_i': 0.1, 'K_d': 0.05}),
    ('CascadedPID',   'cascaded_pid',  {}),
    ('FullPID',       'full_pid',      {'Q_HEX_bias': 2400.0}),
]

# Visual style per controller
STYLE = {
    'Constant':    dict(color='#E74C3C', ls='-',  lw=1.4, label='Constant'),
    'PID':         dict(color='#E67E22', ls='--', lw=1.4, label='PID (tuned)'),
    'CascadedPID': dict(color='#9B59B6', ls=':',  lw=1.4, label='Cascaded PID'),
    'FullPID':     dict(color='#F39C12', ls='-.', lw=1.4, label='Full PID'),
    'MPC':         dict(color='#2ECC71', ls='--', lw=1.8, label='MPC'),
    'SAC':         dict(color='#2980B9', ls='-',  lw=2.2, label='SAC (ours)'),
    'SAC-Curr':    dict(color='#1ABC9C', ls='-',  lw=2.2, label='SAC-Curriculum (ours)'),
    'PPO':         dict(color='#8E44AD', ls='--', lw=1.8, label='PPO'),
}

SCENARIO = 'cold_winter'
SEED = 42
STEP_SIZE = 1.0 / 96.0   # 15-min steps → days
MAX_STEPS = 2880           # 30 days


# ── Episode runner ────────────────────────────────────────────────────────────

def run_rule_controller(name: str, ctrl_type: str, ctrl_params: dict) -> pd.DataFrame:
    from baselines.baseline_controllers import get_controller

    env = ADM1Env_v2(
        scenario_name=SCENARIO,
        reward_config=REWARD_CONFIGS['safety_first'],
        obs_mode='full',
    )
    obs, _ = env.reset(seed=SEED)
    ctrl = get_controller(ctrl_type, **ctrl_params)
    ctrl.reset()

    records = []
    for step in range(MAX_STEPS):
        day = step * STEP_SIZE
        action = ctrl.get_action(obs)
        # Pad action to 3-dim if controller returns 2-dim
        if hasattr(action, '__len__') and len(action) == 2:
            action = np.append(action, 2400.0).astype(np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        records.append({
            'day':     day,
            'VFA':     info.get('total_vfa', 0.0),
            'pH':      info.get('pH', 7.0),
            'q_ch4':   info.get('q_ch4', 0.0),
            'T_L':     env.solver.T_L - 273.15,
            'T_a':     env.solver.T_a - 273.15,
            'F_K':     info.get('F_K', 1.0),
        })
        if terminated or truncated:
            break

    env.close()
    df = pd.DataFrame(records)
    df['controller'] = name
    df['terminated'] = terminated
    df['survival_days'] = df['day'].max()
    return df


def run_mpc_controller(name: str = 'MPC') -> pd.DataFrame:
    try:
        from baselines.mpc_controller import MPCController
    except ImportError:
        print(f"  [skip] {name} — import failed")
        return pd.DataFrame()

    env = ADM1Env_v2(
        scenario_name=SCENARIO,
        reward_config=REWARD_CONFIGS['safety_first'],
        obs_mode='full',
    )
    obs, _ = env.reset(seed=SEED)
    ctrl = MPCController(env=env, horizon=4, max_iter=10, verbose=0)

    records = []
    for step in range(MAX_STEPS):
        day = step * STEP_SIZE
        action = ctrl.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        records.append({
            'day':     day,
            'VFA':     info.get('total_vfa', 0.0),
            'pH':      info.get('pH', 7.0),
            'q_ch4':   info.get('q_ch4', 0.0),
            'T_L':     env.solver.T_L - 273.15,
            'T_a':     env.solver.T_a - 273.15,
            'F_K':     info.get('F_K', 1.0),
        })
        if terminated or truncated:
            break

    env.close()
    df = pd.DataFrame(records)
    df['controller'] = name
    df['terminated'] = False
    df['survival_days'] = df['day'].max()
    return df


def run_rl_controller(name: str, model_dir: Path, algo: str = 'sac') -> pd.DataFrame:
    """Load a trained SB3 model and run one Cold Winter episode."""
    from stable_baselines3 import SAC, PPO
    AlgoClass = {'sac': SAC, 'ppo': PPO}[algo.lower()]

    # Find best model
    best = model_dir / 'best_model' / 'best_model.zip'
    final = model_dir / 'final_model.zip'
    model_path = best if best.exists() else (final if final.exists() else None)
    if model_path is None:
        print(f"  [skip] {name} — no model found in {model_dir}")
        return pd.DataFrame()

    print(f"  Loading {name} from {model_path.name} ...")
    model = AlgoClass.load(model_path)

    env = ADM1Env_v2(
        scenario_name=SCENARIO,
        reward_config=REWARD_CONFIGS['safety_first'],
        obs_mode='full',
    )
    obs, _ = env.reset(seed=SEED)

    records = []
    terminated = False
    for step in range(MAX_STEPS):
        day = step * STEP_SIZE
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        records.append({
            'day':     day,
            'VFA':     info.get('total_vfa', 0.0),
            'pH':      info.get('pH', 7.0),
            'q_ch4':   info.get('q_ch4', 0.0),
            'T_L':     env.solver.T_L - 273.15,
            'T_a':     env.solver.T_a - 273.15,
            'F_K':     info.get('F_K', 1.0),
        })
        if terminated or truncated:
            break

    env.close()
    df = pd.DataFrame(records)
    df['controller'] = name
    df['terminated'] = terminated
    df['survival_days'] = df['day'].max()
    return df


def _find_model(models_dir: Path, pattern: str) -> Path | None:
    candidates = sorted(models_dir.glob(pattern))
    return candidates[0] if candidates else None


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(all_df: pd.DataFrame, out_path: Path):
    controllers = list(all_df['controller'].unique())

    # Draw RL controllers on top (higher z-order)
    RL_CONTROLLERS = {'SAC', 'SAC-Curr', 'PPO', 'MPC'}
    order = [c for c in controllers if c not in RL_CONTROLLERS] + \
            [c for c in controllers if c in RL_CONTROLLERS]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle('Cold Winter Scenario: Controller Trajectory Comparison\n'
                 r'($T_{env}=5°C$, $T_{feed}=10°C$, $\tau_a=30$ d)',
                 fontsize=12, fontweight='bold', y=1.01)

    ax_vfa, ax_ch4, ax_temp = axes

    # Track which controller to use as T_a representative (one clean line)
    ta_ref_shown = False

    for name in order:
        df = all_df[all_df['controller'] == name].sort_values('day')
        st = STYLE.get(name, dict(color='gray', ls='-', lw=1.2, label=name))
        zorder = 3 if name in RL_CONTROLLERS else 2

        # ── (a) VFA ──────────────────────────────────────────────────────────
        ax_vfa.plot(df['day'], df['VFA'], zorder=zorder, **st)
        if df['terminated'].any():
            t_day = df['day'].max()
            vfa_at_term = df.loc[df['day'] == t_day, 'VFA'].values[0]
            ax_vfa.axvline(t_day, color=st['color'], ls=':', lw=0.7, alpha=0.4,
                           zorder=1)
            ax_vfa.scatter([t_day], [vfa_at_term],
                           color=st['color'], marker='X', s=120,
                           edgecolors='black', linewidths=0.5, zorder=5)

        # ── (b) CH4 ──────────────────────────────────────────────────────────
        ax_ch4.plot(df['day'], df['q_ch4'], zorder=zorder, **st)

        # ── (c) Temperature ───────────────────────────────────────────────────
        ax_temp.plot(df['day'], df['T_L'], zorder=zorder, **st)
        # T_a: only draw one representative rule-based T_a to avoid clutter
        if name == 'Constant' and not ta_ref_shown:
            ax_temp.plot(df['day'], df['T_a'],
                         color=st['color'], ls=':', lw=1.1, alpha=0.6,
                         zorder=1, label='_nolegend_')
            ta_ref_shown = True
        elif name in RL_CONTROLLERS:
            # Always show T_a for RL/MPC controllers (they're different)
            ax_temp.plot(df['day'], df['T_a'],
                         color=st['color'], ls=':', lw=1.1, alpha=0.7,
                         zorder=zorder - 1, label='_nolegend_')

    # ── Threshold lines ───────────────────────────────────────────────────────
    ax_vfa.axhline(0.2, color='#F39C12', lw=1.3, ls='--', alpha=0.85, zorder=1)
    ax_vfa.axhline(0.8, color='#E74C3C', lw=1.8, ls='--', alpha=0.95, zorder=1)
    ax_vfa.text(30.1, 0.20, 'soft\n0.2', color='#F39C12', fontsize=7.5, va='center')
    ax_vfa.text(30.1, 0.80, 'hard\n0.8', color='#E74C3C', fontsize=7.5, va='center')
    ax_vfa.set_ylabel('Total VFA (kmol/m³)', fontsize=10)
    ax_vfa.set_ylim(-0.02, 0.92)
    ax_vfa.set_title('(a) VFA accumulation — ✕ marks episode termination (VFA > 0.8)',
                     loc='left', fontsize=9.5)

    # Annotate survival days for terminated controllers
    term_summary = (
        all_df[all_df['terminated'] == True]
        .groupby('controller')['survival_days'].max()
    )
    y_ann = 0.85
    for cname, sdays in term_summary.items():
        st = STYLE.get(cname, {})
        ax_vfa.annotate(f'{sdays:.0f}d', xy=(sdays, 0.80),
                        xytext=(sdays - 1.5, y_ann),
                        fontsize=7, color=st.get('color', 'gray'),
                        arrowprops=dict(arrowstyle='-', color=st.get('color','gray'),
                                        lw=0.6))
        y_ann -= 0.07

    # CH4 annotation for MPC conservative note
    if 'MPC' in all_df['controller'].values:
        ax_ch4.annotate('MPC: conservative\n(lower CH₄, higher safety)',
                        xy=(25, 900), fontsize=7.5, color='#2ECC71',
                        ha='center',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                  ec='#2ECC71', alpha=0.8))

    ax_ch4.set_ylabel('Methane flow q$_{CH_4}$ (m³/d)', fontsize=10)
    ax_ch4.set_title('(b) Methane production', loc='left', fontsize=9.5)

    ax_temp.axhline(35.0, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax_temp.text(30.1, 35.0, '35°C', color='gray', fontsize=7.5, va='center')
    ax_temp.set_ylabel('Temperature (°C)', fontsize=10)
    ax_temp.set_xlabel('Day', fontsize=10)
    ax_temp.set_title('(c) Reactor temperature T$_L$ — dotted: microbial adaptation '
                      'T$_a$ (τ$_a$ = 30 d)', loc='left', fontsize=9.5)

    for ax in axes:
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=9)
    ax_vfa.set_xlim(left=0, right=31)

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = []
    for name in order:
        st = STYLE.get(name, dict(color='gray', ls='-', lw=1.2, label=name))
        handles.append(plt.Line2D([0], [0],
                                  color=st['color'], ls=st['ls'], lw=st['lw'],
                                  label=st['label']))
    handles += [
        plt.Line2D([0], [0], color='gray', ls=':', lw=1.1, label='T$_a$ (dotted)'),
        plt.scatter([], [], marker='X', s=80, c='gray',
                    edgecolors='black', linewidths=0.5, label='Termination point'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               fontsize=8.5, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {out_path}")


def print_survival_table(all_df: pd.DataFrame):
    summary = (
        all_df.groupby('controller')
        .agg(survival_days=('survival_days', 'max'),
             terminated=('terminated', 'max'),
             vfa_max=('VFA', 'max'),
             ch4_mean=('q_ch4', 'mean'))
        .reset_index()
        .sort_values('survival_days')
    )

    print("\n" + "="*65)
    print("Cold Winter — Survival Analysis")
    print("="*65)
    print(f"  {'Controller':<18} {'Survived':>9} {'VFA_max':>9} {'CH4_mean':>10} {'Status'}")
    print(f"  {'-'*18}  {'-'*9}  {'-'*9}  {'-'*10}  {'-'*12}")
    for _, row in summary.iterrows():
        status = 'TERMINATED' if row['terminated'] else 'Completed'
        print(f"  {row['controller']:<18} {row['survival_days']:>8.1f}d "
              f"{row['vfa_max']:>9.3f}  {row['ch4_mean']:>10.0f}  {status}")
    print("="*65)
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Cold Winter trajectory analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--no-rl',  action='store_true',
                        help='Skip RL models (only run rule-based + MPC)')
    parser.add_argument('--no-mpc', action='store_true',
                        help='Skip MPC')
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'models',
                        help='Directory containing trained model subdirs')
    args = parser.parse_args()

    out_dir = ROOT / 'results' / 'cold_winter_trajectory'
    out_dir.mkdir(parents=True, exist_ok=True)

    print("="*65)
    print("Cold Winter Trajectory Analysis")
    print("="*65)

    all_dfs = []

    # ── Rule-based controllers ────────────────────────────────────────────────
    for name, ctype, params in RULE_CONTROLLERS:
        print(f"\n  Running {name} ...", end=' ', flush=True)
        df = run_rule_controller(name, ctype, params)
        all_dfs.append(df)
        status = f"terminated at day {df['day'].max():.1f}" if df['terminated'].any() else "completed"
        print(status)

    # ── MPC ───────────────────────────────────────────────────────────────────
    if not args.no_mpc:
        print(f"\n  Running MPC ...", end=' ', flush=True)
        df = run_mpc_controller('MPC')
        if not df.empty:
            all_dfs.append(df)
            print(f"completed (30d)")

    # ── RL models ─────────────────────────────────────────────────────────────
    if not args.no_rl:
        # SAC (baseline, trained on cold_winter)
        sac_dir = _find_model(args.models_dir, 'sac_cold_winter_safety_first_seed42')
        if sac_dir:
            print(f"\n  Running SAC ...", end=' ', flush=True)
            df = run_rl_controller('SAC', sac_dir, algo='sac')
            if not df.empty:
                all_dfs.append(df)
                print(f"completed (30d)")

        # SAC-Curriculum (experiment 3)
        cur_dir = _find_model(args.models_dir,
                              'sac_cold_winter_safety_first_curriculum_seed42_curriculum')
        if cur_dir:
            print(f"\n  Running SAC-Curriculum ...", end=' ', flush=True)
            df = run_rl_controller('SAC-Curr', cur_dir, algo='sac')
            if not df.empty:
                all_dfs.append(df)
                print(f"completed (30d)")

        # PPO
        ppo_dir = _find_model(args.models_dir, 'ppo_cold_winter_safety_first_seed42')
        if ppo_dir:
            print(f"\n  Running PPO ...", end=' ', flush=True)
            df = run_rl_controller('PPO', ppo_dir, algo='ppo')
            if not df.empty:
                all_dfs.append(df)
                print(f"completed (30d)")

    if not all_dfs:
        print("No data collected — exiting.")
        return

    all_df = pd.concat(all_dfs, ignore_index=True)

    # Save raw trajectory data
    csv_path = out_dir / 'cold_winter_trajectory_data.csv'
    all_df.to_csv(csv_path, index=False)
    print(f"\n  Data saved: {csv_path}")

    # Survival summary
    summary = print_survival_table(all_df)
    summary.to_csv(out_dir / 'cold_winter_survival.csv', index=False)

    # Figure
    make_figure(all_df, out_dir / 'cold_winter_trajectories.png')
    print("\n✓ Done")


if __name__ == '__main__':
    main()

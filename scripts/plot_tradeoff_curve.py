#!/usr/bin/env python3
"""
Production-Safety Tradeoff Curve for the VFA penalty sensitivity sweep.

Reads results/sweep_vfa/sweep_results.json and produces a two-panel figure:
  Panel A — Plan A: c_VFA = 0,   varying w_VFA in {5,10,15,20,25,30}
  Panel B — Plan B: w_VFA = 10,  varying c_VFA in {0,0.25,0.5,1.0,1.5}

Reference points from prior experiments (from cross_scenario_newvr.json):
  MS-SAC (safety_first):  VR = 0.5%,  CH4 = 1473  (w=10, c=1.0)
  ST-SAC (safety_target): VR = 16.1%, CH4 = 1629  (w=20, c=0.0)

Usage:
    cd /home/ihpc/code/biogas/ADM1_RL
    python scripts/plot_tradeoff_curve.py
"""

import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT      = pathlib.Path(__file__).resolve().parents[1]
SWEEP_DIR = ROOT / 'results' / 'sweep_vfa'
OUT_DIR   = ROOT / 'results' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Reference points from existing experiments ──────────────────────────────
MS_SAC = {'label': 'MS-SAC\n(c=1.0)', 'vr': 0.005, 'ch4': 1473, 'w': 10, 'c': 1.0}
ST_SAC = {'label': 'ST-SAC\n(w=20)', 'vr': 0.161, 'ch4': 1629, 'w': 20, 'c': 0.0}

# Reward config → (w_VFA, c_VFA)
CONFIG_PARAMS = {
    'sweep_w5':   (5,  0.0),
    'sweep_w10':  (10, 0.0),
    'sweep_w15':  (15, 0.0),
    'sweep_w25':  (25, 0.0),
    'sweep_w30':  (30, 0.0),
    'sweep_c025': (10, 0.25),
    'sweep_c050': (10, 0.50),
    'sweep_c150': (10, 1.50),
}


def load_results(path):
    with open(path) as f:
        records = json.load(f)
    # Aggregate: for each (reward_config, seed) pair, average over scenarios.
    # Then average over seeds.
    from collections import defaultdict
    by_config_seed = defaultdict(list)
    for r in records:
        key = (r['reward_config'], r['seed'])
        by_config_seed[key].append(r)

    config_stats = {}
    by_config = defaultdict(list)
    for (rc, seed), recs in by_config_seed.items():
        seed_vr  = np.mean([x['violation_rate'] for x in recs if x['violation_rate'] is not None])
        seed_ch4 = np.mean([x['ch4_avg'] for x in recs if x['ch4_avg'] is not None])
        by_config[rc].append({'vr': seed_vr, 'ch4': seed_ch4})

    for rc, seeds in by_config.items():
        vrs  = [s['vr']  for s in seeds]
        ch4s = [s['ch4'] for s in seeds]
        config_stats[rc] = {
            'vr':     np.mean(vrs),
            'vr_std': np.std(vrs),
            'ch4':    np.mean(ch4s),
            'ch4_std':np.std(ch4s),
        }
    return config_stats


def make_plan_a(stats):
    """c_VFA=0, varying w_VFA. Returns sorted list of (w, vr, ch4, label, is_ref)."""
    pts = []
    for rc, (w, c) in CONFIG_PARAMS.items():
        if c != 0.0:
            continue
        if rc not in stats:
            print(f'  [WARN] Plan A: {rc} not in results — skipping')
            continue
        pts.append((w, stats[rc]['vr'], stats[rc]['ch4'],
                    stats[rc]['vr_std'], stats[rc]['ch4_std'],
                    f'w={w}', False))
    # Add ST-SAC reference (w=20, c=0)
    pts.append((20, ST_SAC['vr'], ST_SAC['ch4'], 0.0, 0.0,
                ST_SAC['label'], True))
    return sorted(pts, key=lambda x: x[0])


def make_plan_b(stats):
    """w_VFA=10, varying c_VFA. Returns sorted list of (c, vr, ch4, label, is_ref)."""
    pts = []
    for rc, (w, c) in CONFIG_PARAMS.items():
        if w != 10:
            continue
        if rc not in stats:
            print(f'  [WARN] Plan B: {rc} not in results — skipping')
            continue
        pts.append((c, stats[rc]['vr'], stats[rc]['ch4'],
                    stats[rc]['vr_std'], stats[rc]['ch4_std'],
                    f'c={c}', False))
    # Add MS-SAC reference (w=10, c=1.0)
    pts.append((1.0, MS_SAC['vr'], MS_SAC['ch4'], 0.0, 0.0,
                MS_SAC['label'], True))
    return sorted(pts, key=lambda x: x[0])


def plot_curve(ax, pts, x_key_idx, xlabel, color, ref_color='#e55'):
    xs = [p[x_key_idx] for p in pts]
    vrs = [p[1] * 100 for p in pts]   # → %
    ch4s = [p[2] for p in pts]
    labels = [p[5] for p in pts]
    is_refs = [p[6] for p in pts]

    # Tradeoff: VR on x-axis, CH4 on y-axis (production vs safety)
    ax2 = ax.twinx()

    # VR line (left y-axis)
    for i in range(len(xs) - 1):
        c1 = ref_color if is_refs[i] else color
        c2 = ref_color if is_refs[i+1] else color
        ax.plot(xs[i:i+2], vrs[i:i+2], color=color, lw=1.5, zorder=1)

    # CH4 line (right y-axis, dashed)
    for i in range(len(xs) - 1):
        ax2.plot(xs[i:i+2], ch4s[i:i+2], color=color, lw=1.5, ls='--', zorder=1)

    # Points
    for x, vr, ch4, lbl, is_ref in zip(xs, vrs, ch4s, labels, is_refs):
        mk = '*' if is_ref else 'o'
        sz = 100 if is_ref else 60
        fc = ref_color if is_ref else color
        ax.scatter(x, vr,  c=fc, s=sz, marker=mk, zorder=3)
        ax2.scatter(x, ch4, c=fc, s=sz, marker=mk, zorder=3, facecolors='none',
                   edgecolors=fc, linewidths=1.5)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('VFA/pH Violation Rate (%)', color=color, fontsize=10)
    ax2.set_ylabel('Avg CH₄ Flow (m³/d)', color='#555', fontsize=10)
    ax.tick_params(axis='y', labelcolor=color)
    ax2.tick_params(axis='y', labelcolor='#555')
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(bottom=0)

    return ax2


def main():
    sweep_file = SWEEP_DIR / 'sweep_results.json'
    if not sweep_file.exists():
        print(f'ERROR: {sweep_file} not found. Run eval_sweep_results.py first.')
        return

    print(f'Loading {sweep_file}...')
    stats = load_results(sweep_file)

    print('\n=== Per-config summary ===')
    for rc, s in sorted(stats.items()):
        w, c = CONFIG_PARAMS.get(rc, ('?', '?'))
        print(f'  {rc:<12} w={w} c={c}  VR={s["vr"]*100:.1f}±{s["vr_std"]*100:.1f}%  '
              f'CH4={s["ch4"]:.0f}±{s["ch4_std"]:.0f}')

    plan_a = make_plan_a(stats)
    plan_b = make_plan_b(stats)

    print('\n=== Plan A (c=0, varying w) ===')
    for p in plan_a:
        print(f'  w={p[0]:4}  VR={p[1]*100:5.1f}%  CH4={p[2]:.0f}  {"[ref]" if p[6] else ""}')

    print('\n=== Plan B (w=10, varying c) ===')
    for p in plan_b:
        print(f'  c={p[0]:4}  VR={p[1]*100:5.1f}%  CH4={p[2]:.0f}  {"[ref]" if p[6] else ""}')

    # ── Figure: 1×2 panels ──────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle('Production–Safety Tradeoff: VFA Penalty Sensitivity',
                 fontsize=13, fontweight='bold', y=1.01)

    plot_curve(ax1, plan_a, x_key_idx=0, xlabel='VFA proportional weight w_VFA  (c_VFA = 0)',
               color='#2a6', ref_color='#c44')
    plot_curve(ax2, plan_b, x_key_idx=0, xlabel='VFA constant penalty c_VFA  (w_VFA = 10)',
               color='#47b', ref_color='#c44')

    ax1.set_title('Plan A: Linear coefficient sweep\n(no constant penalty)',
                  fontsize=10, pad=8)
    ax2.set_title('Plan B: Constant penalty sweep\n(fixed linear w=10)',
                  fontsize=10, pad=8)

    # Legends
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color='gray', lw=1.5, label='VFA/pH Violation Rate (%)'),
        Line2D([0], [0], color='gray', lw=1.5, ls='--', label='Avg CH₄ Flow (m³/d)'),
        Line2D([0], [0], marker='*', color='#c44', ms=10, lw=0, label='Reference (MS/ST-SAC)'),
        Line2D([0], [0], marker='o', color='gray', ms=7, lw=0, label='Sweep point'),
    ]
    ax1.legend(handles=legend_elems, fontsize=8, loc='upper right')

    plt.tight_layout()
    out_path = OUT_DIR / 'tradeoff_curve.pdf'
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    png_path = OUT_DIR / 'tradeoff_curve.png'
    plt.savefig(png_path, bbox_inches='tight', dpi=150)
    print(f'\nFigure saved → {out_path}')
    print(f'              → {png_path}')


if __name__ == '__main__':
    main()

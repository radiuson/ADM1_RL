#!/usr/bin/env python3
"""
Learning curves for the w and c sweep.
Two-panel figure:
  Left:  w-sweep (c=0), w = 2,5,10,20,40,60 — mean±std across 5 seeds
  Right: c-sweep (w=20), c = 0.25,0.5,1.0,1.5,2.0 — mean±std across 5 seeds
y-axis: mean per-step reward (episode_return / ep_length)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

BASE   = Path(__file__).parent / 'models_sweep'
FIGDIR = Path(__file__).parent / 'results/eval_planA/figures'
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

SEEDS = [1, 2, 3, 4, 5]
EP_LEN = 5760   # nominal episode length for normalisation


def load_curves(config_tag):
    """
    Load per-step reward curves for all seeds of a given config.
    Returns (timesteps, returns_array) where returns_array is
    shape (n_seeds, n_checkpoints).
    """
    all_returns = []
    timesteps = None
    for seed in SEEDS:
        name = f'sac_scenario_cur_{config_tag}_uniform_random_seed{seed}'
        p = BASE / name / 'eval' / 'evaluations.npz'
        if not p.exists():
            continue
        d = np.load(p)
        # mean over eval episodes, divide by ep_length → per-step reward
        ep_len_mean = d['ep_lengths'].mean(axis=1)           # (n_checkpoints,)
        ep_ret_mean = d['results'].mean(axis=1)               # (n_checkpoints,)
        per_step    = ep_ret_mean / ep_len_mean
        all_returns.append(per_step)
        if timesteps is None:
            timesteps = d['timesteps']
    if not all_returns:
        return None, None
    return timesteps, np.stack(all_returns)   # (n_seeds, n_checkpoints)


# ── colour maps ────────────────────────────────────────────────────────────────
W_CMAP = plt.cm.Blues
C_CMAP = plt.cm.Oranges

# selected w values for left panel
W_SELECT = [2, 5, 10, 20, 40, 60]
w_norm   = {w: i / (len(W_SELECT) - 1) for i, w in enumerate(W_SELECT)}

# c-sweep configs (w=20)
C_SELECT = ['0.25', '0.50', '1.00', '1.50', '2.00']
C_VALS   = [float(c) for c in C_SELECT]
c_norm   = {c: i / (len(C_SELECT) - 1) for i, c in enumerate(C_VALS)}

# ── figure ────────────────────────────────────────────────────────────────────
fig, (ax_w, ax_c) = plt.subplots(1, 2, figsize=(12, 4.8))
fig.suptitle('Training Reward Curves: Effect of Reward Parameters $w$ and $c$',
             fontsize=12, fontweight='bold', y=1.01)
fig.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.14, wspace=0.30)

# ── LEFT: w-sweep ─────────────────────────────────────────────────────────────
ax_w.set_title('(a)  $w$-sweep  ($c=0$)', fontsize=11, pad=6)

for w in W_SELECT:
    tag = f'sweep_w{w}'
    ts, arr = load_curves(tag)
    if ts is None:
        continue
    col = W_CMAP(0.25 + 0.65 * w_norm[w])
    mu  = arr.mean(axis=0)
    sig = arr.std(axis=0)
    ax_w.fill_between(ts / 1e3, mu - sig, mu + sig, alpha=0.18, color=col)
    ax_w.plot(ts / 1e3, mu, lw=2.0, color=col,
              label=f'$w$={w}')
    # mark last point
    ax_w.scatter(ts[-1] / 1e3, mu[-1], color=col, s=50, zorder=5, linewidths=0.5,
                 edgecolors='#333')

ax_w.set_xlabel('Training steps (×10³)', fontsize=10)
ax_w.set_ylabel('Mean per-step reward', fontsize=10)
ax_w.grid(True, alpha=0.22, lw=0.5)
ax_w.tick_params(labelsize=9)

# legend with colour gradient indication
handles_w = [
    Line2D([0], [0], color=W_CMAP(0.25 + 0.65 * w_norm[w]), lw=2.0,
           label=f'$w$={w}')
    for w in W_SELECT
]
ax_w.legend(handles=handles_w, fontsize=8.5, loc='lower right',
            framealpha=0.9, handlelength=1.8, labelspacing=0.3)

# annotation: higher w → different convergence target
ax_w.text(0.02, 0.97, 'Higher $w$ → stronger safety incentive\n→ lower reward scale but safer policy',
         transform=ax_w.transAxes, ha='left', va='top',
         fontsize=8, color='#546e7a',
         bbox=dict(boxstyle='round,pad=0.25', fc='#f5f5f5', ec='#bdbdbd', lw=0.7))

# ── RIGHT: c-sweep ────────────────────────────────────────────────────────────
ax_c.set_title('(b)  $c$-sweep  ($w=20$)', fontsize=11, pad=6)

for c_str, c_val in zip(C_SELECT, C_VALS):
    tag = f'sweep_w20c{c_str.replace(".", "")}'
    ts, arr = load_curves(tag)
    if ts is None:
        continue
    col = C_CMAP(0.30 + 0.60 * c_norm[c_val])
    mu  = arr.mean(axis=0)
    sig = arr.std(axis=0)
    ax_c.fill_between(ts / 1e3, mu - sig, mu + sig, alpha=0.18, color=col)
    ax_c.plot(ts / 1e3, mu, lw=2.0, color=col,
              label=f'$c$={c_val:.2g}')
    ax_c.scatter(ts[-1] / 1e3, mu[-1], color=col, s=50, zorder=5, linewidths=0.5,
                 edgecolors='#333')

ax_c.set_xlabel('Training steps (×10³)', fontsize=10)
ax_c.set_ylabel('Mean per-step reward', fontsize=10)
ax_c.grid(True, alpha=0.22, lw=0.5)
ax_c.tick_params(labelsize=9)

handles_c = [
    Line2D([0], [0], color=C_CMAP(0.30 + 0.60 * c_norm[c_val]), lw=2.0,
           label=f'$c$={c_val:.2g}')
    for c_val in C_VALS
]
ax_c.legend(handles=handles_c, fontsize=8.5, loc='lower right',
            framealpha=0.9, handlelength=1.8, labelspacing=0.3)

ax_c.text(0.02, 0.97, 'Higher $c$ → cliff at threshold\n→ more conservative behaviour',
         transform=ax_c.transAxes, ha='left', va='top',
         fontsize=8, color='#546e7a',
         bbox=dict(boxstyle='round,pad=0.25', fc='#f5f5f5', ec='#bdbdbd', lw=0.7))

# ── save ──────────────────────────────────────────────────────────────────────
for ext in ('pdf', 'png'):
    fig.savefig(FIGDIR / f'learning_curves_sweep.{ext}', bbox_inches='tight', dpi=200)
plt.close(fig)
print("Saved: learning_curves_sweep.pdf / .png")

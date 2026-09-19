#!/usr/bin/env python3
"""Figure 1: production-violation frontier.

Individual runs, not family means, so the reader can see the spread that the
median in Table 2 summarises.  On-policy and off-policy use different markers
because the classification is the paper's main claim.
"""
import json, sys
import numpy as np
sys.path.insert(0, __import__('pathlib').Path(__file__).resolve().parents[1].as_posix())
from paths import glob as dglob
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = [json.load(open(f)) for f in dglob('evbase')]
PI = [b for b in B if b['kind'] == 'pi']
def envelope(rows):
    pts = sorted((b['viol'], b['ch4']) for b in rows)
    out, best = [], -1
    for v, c in pts:
        if c > best:
            out.append((v, c)); best = c
    return np.array(out)
EP, EA = envelope(PI), envelope(B)

R = [json.load(open(f)) for f in dglob('evres')]
def fam(m):
    for k, g in (('_recurrentppo_', 'RecurrentPPO'), ('_crossq_', 'CrossQ'),
                 ('_a2c_', 'A2C'), ('_ars_', 'ARS'), ('_ddpg_', 'DDPG'),
                 ('_td3_', 'TD3'), ('_trpo_', 'TRPO'), ('_ppo_', 'PPO'),
                 ('_h5_tqc_', 'TQC+mem'), ('_h5_', 'SAC+mem')):
        if k in m:
            return g
    if m.startswith('models_conv'):  return 'SAC'
    if m.startswith(('models_v5', 'models_seed')): return 'SAC'
    return None
ON  = {'A2C', 'TRPO', 'PPO', 'RecurrentPPO'}
OFF = {'SAC', 'TQC+mem', 'TD3', 'DDPG', 'CrossQ', 'SAC+mem'}

on, off, ars = [], [], []
for r in R:
    g = fam(r['model'])
    if g in ON:    on.append((r['viol'], r['ch4']))
    elif g in OFF: off.append((r['viol'], r['ch4']))
    elif g == 'ARS': ars.append((r['viol'], r['ch4']))
on, off, ars = map(np.array, (on, off, ars))

fig, ax = plt.subplots(figsize=(7.0, 4.4))
ax.scatter(*np.array([(b['viol'], b['ch4']) for b in B]).T, s=14, c='0.78',
           marker='s', linewidths=0, label=f'conventional configs (n={len(B)})', zorder=1)
ax.plot(EA[:, 0], EA[:, 1], '-', c='0.35', lw=1.4, zorder=2,
        label=f'conventional Pareto envelope ({len(EA)} pts)')
ax.scatter(off[:, 0], off[:, 1], s=26, facecolors='none', edgecolors='#9a3a30',
           linewidths=1.0, marker='o', zorder=3, label=f'off-policy runs (n={len(off)})')
ax.scatter(on[:, 0], on[:, 1], s=30, c='#2f6f5e', marker='^', linewidths=0,
           zorder=4, label=f'on-policy runs (n={len(on)})')
if len(ars):
    ax.scatter(ars[:, 0], ars[:, 1], s=26, c='#a9691c', marker='x',
               linewidths=1.1, zorder=4, label=f'ARS, gradient-free (n={len(ars)})')
ax.axvline(23.8, color='0.55', ls=':', lw=1.1, zorder=1)
ax.text(24.4, 2200, 'reference plant,\nscale-matched\n(23.8 %)', fontsize=7.2,
        color='0.42', va='top', ha='left', linespacing=1.25)

ax.set_xlabel('soft-constraint violation rate (\\%)' if False else 'soft-constraint violation rate (%)')
ax.set_ylabel('mean methane flow (m$^3$/d)')
ax.set_xlim(-1, 52); ax.set_ylim(950, 2260)
ax.grid(alpha=0.25, lw=0.6)
ax.legend(fontsize=7.4, loc='lower right', framealpha=0.94, ncol=1,
          borderpad=0.5, labelspacing=0.35)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'figures/fig_frontier.{ext}', dpi=220)
print(f'conventional {len(B)} | on {len(on)} | off {len(off)} | ARS {len(ars)}')

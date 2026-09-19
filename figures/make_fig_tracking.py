#!/usr/bin/env python3
"""Figure 2: how closely each constrained algorithm attains the specified
violation rate.  The 45-degree line is exact specification; points below it are
conservative, above it are exceedances."""
import json, sys
import numpy as np
sys.path.insert(0, __import__('pathlib').Path(__file__).resolve().parents[1].as_posix())
from paths import glob as dglob
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

C = {}
for f in dglob('evcmdp'):
    r = json.load(open(f))
    if r.get('cost_limit'):
        C[(r['algo'], r['seed'], r['cost_limit'])] = r
G = defaultdict(lambda: defaultdict(list))
for r in C.values():
    G[r['algo']][r['cost_limit']].append(r['viol'])

ON = {'CPO', 'PCPO', 'CUP', 'FOCOPS', 'OnCRPO', 'P3O', 'PPOLag', 'CPPOPID',
      'TRPOLag', 'TRPOPID'}
HILITE = {'TRPOPID': '#2f6f5e', 'PCPO': '#3d8f7a', 'SACPID': '#9a3a30',
          'DDPGLag': '#b8564a'}

fig, ax = plt.subplots(figsize=(6.2, 4.6))
lim = [0, 26]
ax.plot(lim, lim, '-', c='0.45', lw=1.2, zorder=1, label='exact specification')
ax.fill_between(lim, lim, [26, 26], color='#9a3a30', alpha=0.055, zorder=0)
ax.text(6.2, 23.4, 'exceeds specification', fontsize=7.6, color='#9a3a30', alpha=.85)
ax.text(9.2, 2.2, 'conservative', fontsize=7.6, color='0.45', ha='right')

for a, byc in sorted(G.items()):
    cls = a in ON
    xs = sorted(byc)
    tx = [100 * c / 60 for c in xs]
    ty = [np.mean(byc[c]) for c in xs]
    col = HILITE.get(a, '#2f6f5e' if cls else '#9a3a30')
    named = a in HILITE
    ax.plot(tx, ty, '-' if cls else '--', c=col, lw=1.9 if named else 0.85,
            marker='o' if cls else 's', ms=5.0 if named else 3.0,
            alpha=1.0 if named else 0.42, zorder=4 if named else 2,
            label=a if named else None)
    if named:
        dy = {'TRPOPID': -9, 'PCPO': 6, 'SACPID': 2, 'DDPGLag': -2}.get(a, 0)
        ax.annotate(a, (tx[-1], ty[-1]), textcoords='offset points',
                    xytext=(7, dy), fontsize=8.0, color=col, weight='medium')

ax.set_xlabel('specified violation rate,  cost limit / episode length (%)')
ax.set_ylabel('attained violation rate (%)')
ax.set_xlim(0, 11.4); ax.set_ylim(0, 26)
ax.grid(alpha=0.25, lw=0.6)
h, l = ax.get_legend_handles_labels()
ax.legend(h, l, fontsize=7.4, loc='center left', bbox_to_anchor=(0.015, 0.72),
          framealpha=0.94, labelspacing=0.32)
fig.tight_layout()
for e in ('pdf', 'png'):
    fig.savefig(f'figures/fig_tracking.{e}', dpi=220)
print('algorithms plotted:', len(G))

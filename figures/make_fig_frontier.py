#!/usr/bin/env python3
"""Figure 1: production-violation frontier.

Individual runs, not family means, so the reader can see the spread that the
median in Table 2 summarises.  On-policy and off-policy use different markers
because the classification is the paper's main claim.
"""
import json, sys
import numpy as np
sys.path.insert(0, __import__('pathlib').Path(__file__).resolve().parents[1].as_posix())
import glob as _g, os
SP = os.environ.get('ADM1_S7', '/tmp/claude-1000/-home-ihpc-code/'
                    'd8e23994-23f1-4be7-bbbc-30cb1d90de5a/scratchpad')
def dglob(which):
    if which == 'evbase':
        return (_g.glob('results_frozen_20260915/evbase/*.json')
                + _g.glob(f'{SP}/evbase_ext/*.json'))
    return _g.glob(f'{SP}/s7/evres/*.json')
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
                 ('_h5_tqc_', 'TQC+mem'), ('_h5_', 'SAC+mem'),
                 ('_tqc_', 'TQC')):
        if k in m:
            return g
    if m.startswith('models_conv'):  return 'SAC'
    if m.startswith(('models_v5', 'models_seed')): return 'SAC'
    return None
ON  = {'A2C', 'TRPO', 'PPO', 'RecurrentPPO'}
OFF = {'SAC', 'TQC', 'TD3', 'DDPG', 'CrossQ'}

WEIGHTS = ('lw0p5', 'lw1', 'lw2', 'lw5')
CORRECTED = {'A2C': 'models_align', 'DDPG': 'models_noise',
             'TD3': 'models_noise', 'CrossQ': 'models_native'}
on, off, ars = [], [], []
seen = set()
for r in sorted(R, key=lambda x: x.get('model', '')):
    if r.get('steps') != 150000 or r.get('w') not in WEIGHTS:
        continue
    g = fam(r['model'])
    want = CORRECTED.get(g)
    if want and not r['model'].startswith(want):
        continue
    key = (g, r.get('w'), r.get('seed'))
    if key in seen:
        continue
    seen.add(key)
    if g in ON:    on.append((r['viol'], r['ch4']))
    elif g in OFF: off.append((r['viol'], r['ch4']))
    elif g == 'ARS': ars.append((r['viol'], r['ch4']))
on, off, ars = map(np.array, (on, off, ars))

fig, ax = plt.subplots(figsize=(7.0, 4.5))
ax.scatter(*np.array([(b['viol'], b['ch4']) for b in B]).T, s=14, c='0.78',
           marker='s', linewidths=0, zorder=1,
           label=f'conventional controllers, all tunings (n={len(B)})')
ax.plot(EA[:, 0], EA[:, 1], '-', c='0.35', lw=1.4, zorder=2,
        label=f'best conventional controller at each rate ({len(EA)} pts)')
ax.scatter(off[:, 0], off[:, 1], s=26, facecolors='none', edgecolors='#9a3a30',
           linewidths=1.0, marker='o', zorder=3,
           label=f'RL, off-policy update (n={len(off)})')
ax.scatter(on[:, 0], on[:, 1], s=30, c='#2f6f5e', marker='^', linewidths=0,
           zorder=4, label=f'RL, on-policy update (n={len(on)})')
if len(ars):
    ax.scatter(ars[:, 0], ars[:, 1], s=26, c='#a9691c', marker='x',
               linewidths=1.1, zorder=4,
               label=f'RL, gradient-free search (ARS, n={len(ars)})')
# The mapped plant range, not a single line: the ratio behind it is not
# constant, so its endpoints (1579 and 3750 mg/L) bracket the comparison.
ax.axvspan(0.1, 23.8, color='0.55', alpha=0.10, lw=0, zorder=0,
           label='reference plant, mapped range')
# Which way is which: the axes trade the same quantity against each other, so
# say out loud what each end of the trade-off means.
# A point above the grey line produces more methane than the best conventional
# controller operating at the same excursion rate; that is the comparison.
ax.set_xlabel('control steps above the soft VFA limit (%)', labelpad=7)
ax.set_ylabel('mean methane flow (m$^3$/d)', labelpad=7)
ax.set_xlim(-1, 52); ax.set_ylim(680, 2230)
ax.grid(alpha=0.25, lw=0.6)
ax.legend(fontsize=7.4, loc='lower right', framealpha=0.94, ncol=1,
          borderpad=0.5, labelspacing=0.35)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'figures/fig_frontier.{ext}', dpi=220)
print(f'conventional {len(B)} | on {len(on)} | off {len(off)} | ARS {len(ars)}')

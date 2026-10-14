#!/usr/bin/env python3
"""Regenerate the paper's two result tables from the evaluation records.

Both come from one pass over the data, so they cannot drift apart:

  table_results.tex    the reward-penalty table
  table_cmdp.tex       the constrained table

The figures the running text quotes are printed to the terminal rather than
written as macros, so a change in the data shows up as a mismatch to be
reconciled rather than silently rewriting the prose.

    python3 scripts/build_tables.py <outdir>
"""
from __future__ import annotations

import glob
import json
import os
import re
import random
import statistics as st
import sys
from collections import defaultdict

import numpy as np

# The per-run evaluation records every table and figure is computed from.
# paper_data/ ships with the repository, so the tables rebuild from a clean
# clone with no other inputs; ADM1_PAPER_DATA overrides it to re-run against a
# freshly evaluated set.
DATA = os.environ.get(
    'ADM1_PAPER_DATA',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 'paper_data'))
EPISODE_STEPS = 60

REWARD_ROWS = [('PPO', 'ppo', 'on-policy'), ('A2C', 'a2c', 'on-policy'),
               ('TRPO', 'trpo', 'on-policy'), ('ARS', 'ars', 'gradient-free'),
               ('RecurrentPPO', 'recurrentppo', 'on-policy'),
               ('SAC', 'sac', 'off-policy'), ('TQC', 'tqc', 'off-policy'),
               ('DDPG', 'ddpg', 'off-policy'), ('TD3', 'td3', 'off-policy'),
               ('CrossQ', 'crossq', 'off-policy')]
CMDP_ON = ['CUP', 'PCPO', 'TRPOPID', 'FOCOPS', 'CPPOPID',
           'PPOLag', 'CPO', 'TRPOLag', 'OnCRPO', 'P3O']
CMDP_OFF = ['TD3Lag', 'DDPGLag', 'SACPID', 'SACLag']

# A2C, DDPG, TD3 and CrossQ are read from the directories holding the runs with
# the corrected settings; the earlier models_algo2 runs for those four used
# action noise, n_steps and warm-up values that are not on the same scale as
# the other families and are excluded.
CORRECTED = {'a2c': 'models_align', 'ddpg': 'models_noise',
             'td3': 'models_noise', 'crossq': 'models_native'}
# The swept grid. Runs at other penalty weights exist for SAC and TRPO from
# exploratory sweeps; they are not part of this design and are left out so that
# every family enters the comparison on the same four weights.
WEIGHTS = ('lw0p5', 'lw1', 'lw2', 'lw5')


# The learned policies are clipped to a feed-strength multiplier of 1.3, so a
# conventional configuration above that commands an input the policies cannot.
# The reported envelope holds both to the same actuator range; ADM1_FULL_RANGE
# restores the wider sweep for the sensitivity check in the text.
MULT_MAX = 1.3


def _mult(b):
    """Feed-strength multiplier of a configuration; constant feed runs at 1.3."""
    m = re.search(r'_m([0-9.]+)', b['name'])
    return float(m.group(1)) if m else 1.3


# The dual-loop PI family lives outside paper_data/ because it was swept
# separately; it is part of the conventional set, so the envelope includes it
# unless ADM1_STATIC_ONLY asks for the static-multiplier families alone.
DUAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'results_dualloop')


def envelope():
    B = [json.load(open(f)) for f in glob.glob(f'{DATA}/evbase/*.json')]
    if not os.environ.get('ADM1_FULL_RANGE'):
        B = [b for b in B if _mult(b) <= MULT_MAX]
    if not os.environ.get('ADM1_STATIC_ONLY'):
        B += [json.load(open(f)) for f in glob.glob(f'{DUAL}/*.json')]
    pts = sorted((b['viol'], b['ch4'], b['name']) for b in B)
    E, best = [], -1.0
    for v, c, n in pts:
        if c > best:
            E.append((v, c, n))
            best = c
    return B, E


def boot(vals, rng, B=4000):
    s = sorted(st.median([rng.choice(vals) for _ in vals]) for _ in range(B))
    return s[int(.025 * B)], s[int(.975 * B)]


def boot_by_seed(runs, value, rng, B=4000):
    """Per-family interval that resamples seeds, not individual runs.

    A family's runs are one per (weight, seed) cell, so the four runs that
    share a seed are not independent draws.  Resampling runs flat treats them
    as if they were and reports an interval narrower than the design carries.
    This draws seeds with replacement and takes each drawn seed's runs across
    all weights together.
    """
    by = defaultdict(list)
    for r in runs:
        v = value(r)
        if v is not None:
            by[r.get('seed')].append(v)
    seeds = [k for k in by if by[k]]
    if not seeds:
        return float('nan'), float('nan')
    out = []
    for _ in range(B):
        pick = [rng.choice(seeds) for _ in seeds]
        vals = [v for sd in pick for v in by[sd]]
        out.append(st.median(vals))
    out.sort()
    return out[int(.025 * B)], out[int(.975 * B)]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '.'
    B, E = envelope()
    xs = [a for a, _, _ in E]
    ys = [b for _, b, _ in E]

    def delta(d):
        if not (xs[0] <= d['viol'] <= xs[-1]):
            return None
        return d['ch4'] - float(np.interp(d['viol'], xs, ys))

    rng = random.Random(20260919)
    M = {}                                   # macro name -> rendered value

    # ---- reward-penalty -------------------------------------------------
    R = defaultdict(list)
    for f in glob.glob(f'{DATA}/evres/*.json'):
        d = json.load(open(f))
        if d.get('steps') != 150000 or '_h5_' in d.get('model', ''):
            continue
        if d.get('w') not in WEIGHTS:
            continue
        a = (d.get('algo') or 'sac').lower()
        want = CORRECTED.get(a)
        if want and not d.get('model', '').startswith(want):
            continue
        R[a].append(d)

    # Some (family, weight, seed) cells were trained more than once across
    # sweeps. Counting every training of a cell would weight those families
    # more heavily in the class comparison, so one run per cell is kept --
    # the first by model path, which is deterministic and independent of the
    # result -- leaving the balanced 4 weights x 10 seeds design.
    for a in list(R):
        seen = {}
        for d in sorted(R[a], key=lambda x: x.get('model', '')):
            seen.setdefault((d.get('w'), d.get('seed')), d)
        R[a] = list(seen.values())

    lines = []
    prev = None
    for label, key, upd in REWARD_ROWS:
        runs = R.get(key, [])
        if not runs:
            continue
        # The rule between the classes carries the table's main comparison.
        if prev is not None and upd == 'off-policy' and prev != 'off-policy':
            lines.append('\\midrule')
        prev = upd
        ds = [x for x in (delta(r) for r in runs) if x is not None]
        lo, hi = boot_by_seed(runs, delta, rng)
        lines.append(f'{label:<12} & {upd:<13} & {len(runs)} & '
                     f'${st.median(ds):+.0f}$ & $[{lo:+.0f},{hi:+.0f}]$ & '
                     f'{sum(1 for v in ds if v > 0)}/{len(ds)} \\\\')
        # LaTeX control sequences are letters only, so digits are spelled out.
        name = (label.replace('A2C', 'AtwoC').replace('TD3', 'TDthree')
                     .replace('TQC', 'TQC').replace('P3O', 'PthreeO'))
        M['Rew' + ''.join(c for c in name if c.isalpha())] = \
            f'{st.median(ds):+.0f}'
    # The file carries the whole tabular, not just the rows: \input of a
    # fragment ending in \\ leaves the last row unterminated where LaTeX
    # expects \bottomrule.
    open(os.path.join(out, 'table_results.tex'), 'w').write(
        '\\begin{tabular}{llrrrr}\n\\toprule\n'
        'Algorithm & Update & Runs & Median & 95\\,\\% CI & Runs above \\\\\n'
        '\\midrule\n' + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n')

    # ---- constrained ----------------------------------------------------
    C = defaultdict(list)
    for f in glob.glob(f'{DATA}/evcmdp/*.json'):
        d = json.load(open(f))
        C[d['algo']].append(d)

    def absorbing_viol(r):
        """Violation rate on the fixed 420-step horizon.

        delta is defined at the full T=60 (eq:rate), so scoring viol over
        completed steps would compare the realised and specified rates over
        different exposures.  A breach is therefore absorbing: every step
        after it counts as a violation.  Only two of the 560 constrained
        runs terminate early, so this leaves all but two unchanged.
        """
        ps = r.get('per_scenario') or {}
        if not ps:
            return r['viol']
        n = sum(v['steps'] for v in ps.values())
        full = len(ps) * EPISODE_STEPS
        if n >= full:
            return r['viol']
        viols = sum(v['steps'] * v['viol'] / 100.0 for v in ps.values())
        return 100.0 * (viols + (full - n)) / full

    def cmdp_row(a):
        runs = C[a]
        ds = [x for x in (delta(r) for r in runs) if x is not None]
        lo, hi = boot_by_seed(runs, delta, rng)
        sv = [absorbing_viol(r) - 100.0 * r['cost_limit'] / EPISODE_STEPS
              for r in runs if r.get('cost_limit')]
        av = [abs(x) for x in sv]
        safe = 100.0 * sum(1 for x in sv if x <= 0) / len(sv)
        return (f'{a:<8} & ${st.median(ds):+.0f}$ & $[{lo:+.0f},{hi:+.0f}]$ & '
                f'{sum(1 for v in ds if v > 0)}/{len(ds)} & '
                f'${st.mean(sv):+.2f}$ & {safe:.0f}\\,\\% & '
                f'{st.mean(av):.2f} \\\\'), st.median(ds), st.mean(av)

    # Both classes carry a heading: labelling only the second leaves the first
    # group unnamed, and the split between them is the table's main comparison.
    rows, meds = [], {}
    rows.append('\\multicolumn{7}{l}{\\emph{On-policy}} \\\\')
    for a in CMDP_ON:
        r, m, ab = cmdp_row(a)
        rows.append(r)
        meds[a] = (m, ab)
    rows.append('\\midrule')
    rows.append('\\multicolumn{7}{l}{\\emph{Off-policy}} \\\\')
    for a in CMDP_OFF:
        r, m, ab = cmdp_row(a)
        rows.append(r)
        meds[a] = (m, ab)
    open(os.path.join(out, 'table_cmdp.tex'), 'w').write(
        '\\begin{tabular}{lrrrrrr}\n\\toprule\n'
        'Algorithm & Median & 95\\,\\% CI & Above & $\\Delta_C$ & Safe & $A_C$ '
        '\\\\\n\\midrule\n' + '\n'.join(rows) +
        '\n\\bottomrule\n\\end{tabular}\n')

    # ---- summary --------------------------------------------------------
    # The manuscript spells these figures out in the text rather than through
    # macros, so they are printed for checking against it rather than written
    # to a file for it to include.
    best = max(CMDP_ON, key=lambda a: meds[a][0])
    spec = min(CMDP_ON, key=lambda a: meds[a][1])
    nsig = sum(1 for a in CMDP_ON
               if boot([x for x in (delta(r) for r in C[a]) if x is not None],
                       rng)[0] > 0)
    zero = max(c for v, c, _ in E if v == 0)

    # len(B) counts evaluations; four constant-feed configurations were
    # evaluated twice, so the distinct-configuration count is four lower.
    print(f'envelope {len(B)} evaluations -> {len(E)} points, '
          f'{xs[0]:.2f}-{xs[-1]:.1f}%, best at zero violation {zero:.0f}')
    print(f'wrote table_results.tex and table_cmdp.tex to {out}')
    print(f'{nsig}/{len(CMDP_ON)} on-policy constrained families above the '
          f'envelope with intervals excluding zero')
    print(f'best constrained median {best} {meds[best][0]:+.0f} m3/d; '
          f'closest to specification {spec} at {meds[spec][1]:.2f} pp')
    print('reward-penalty family medians: '
          + ', '.join(f'{k[3:]} {v}' for k, v in sorted(M.items())
                      if k.startswith('Rew')))


if __name__ == '__main__':
    main()

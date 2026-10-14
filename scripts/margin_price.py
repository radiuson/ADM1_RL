#!/usr/bin/env python3
"""What a percentage point of unspent allowance is worth, measured on the
constrained runs themselves.

Section 8 reports the absolute deviation $A_C$ as a tracking quantity and
says plainly that the production it costs cannot be read off it.  The open
question is whether an unspent allowance costs anything at all: if it does
not, counting conservatism as a deviation has no economic content.

The conventional envelope answers this for conventional controllers, whose
slope falls from 113 m3/d per point below 2 % to 15 between 2 and 10 %.  That
is a statement about PI loops, not about what a learned policy forgoes.  This
script asks the same question of the constrained runs themselves: across the
runs of one family, does reaching a lower violation rate come with less
methane, and at what rate?

Two estimates are reported, because they answer slightly different questions:

    within-family   runs of one family across its four cost limits, which
                    holds the algorithm fixed and varies what it was asked
                    for.  This is the quantity the paper needs.
    pooled          all constrained runs together, which mixes families and
                    therefore confounds the price of margin with differences
                    in how well each family produces.

A family enters the within-family estimate only if its runs span enough
violation rates to define a slope.  The pooled figure is reported for
contrast, not as the headline.

    python3 scripts/margin_price.py [out.json]
"""
from __future__ import annotations

import glob
import json
import os
import random
import statistics as st
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.environ.get('ADM1_PAPER_DATA', os.path.join(ROOT, 'paper_data'))

# The dense sweep is kept in results_margin/ and is deliberately not merged
# into paper_data/evcmdp/.  Table 4 reads every record in that directory
# without filtering on the cost limit, so folding in four further limits for
# three of the fourteen families would unbalance the design and move the
# reported figures -- TRPOPID's tracking deviation from 1.22 to 1.24, PPOLag's
# from 2.70 to 2.58 -- while the other eleven families stayed at forty runs.
MARGIN = os.path.join(ROOT, 'results_margin', 'evres')

# The band the constrained families actually work in.  Slopes are also
# reported over the full range, but the paper's claim concerns the runs near
# the limits they were given.
BANDS = [(0.0, 2.0), (0.0, 5.0), (0.0, 10.0), (0.0, 1e9)]
MIN_RUNS = 6          # a slope on fewer points than this is not reported
MIN_SPREAD = 1.0      # percentage points of violation rate


def load():
    runs = []
    for f in glob.glob(os.path.join(DATA, 'evcmdp', '*.json')):
        d = json.load(open(f))
        if d.get('viol') is None or d.get('ch4') is None:
            continue
        runs.append({'algo': d['algo'], 'viol': float(d['viol']),
                     'ch4': float(d['ch4']), 'seed': d.get('seed'),
                     'limit': d.get('cost_limit')})
    return runs


def ols(xs, ys):
    """Slope and intercept, or None when x has no spread."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return b, my - b * mx


def boot_ci(xs, ys, reps=4000, seed=0):
    """Percentile interval for the slope, resampling runs."""
    rng = random.Random(seed)
    n = len(xs)
    out = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        r = ols([xs[i] for i in idx], [ys[i] for i in idx])
        if r:
            out.append(r[0])
    if len(out) < reps // 2:
        return None
    out.sort()
    return out[int(.025 * len(out))], out[int(.975 * len(out))]


def in_band(runs, lo, hi):
    return [r for r in runs if lo <= r['viol'] < hi]


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else None
    runs = load()
    print(f'{len(runs)} constrained runs from {DATA}/evcmdp')
    fams = defaultdict(list)
    for r in runs:
        fams[r['algo']].append(r)
    print(f'{len(fams)} families\n')

    results = {}
    for lo, hi in BANDS:
        label = f'{lo:g}-{hi:g}%' if hi < 1e8 else 'all rates'
        sel = in_band(runs, lo, hi)
        print(f'=== violation rate {label} — {len(sel)} runs ===')

        # Within-family slopes.
        rows = []
        for a in sorted(fams):
            fr = in_band(fams[a], lo, hi)
            if len(fr) < MIN_RUNS:
                continue
            xs = [r['viol'] for r in fr]
            ys = [r['ch4'] for r in fr]
            if max(xs) - min(xs) < MIN_SPREAD:
                continue
            r = ols(xs, ys)
            if not r:
                continue
            rows.append((a, r[0], len(fr), max(xs) - min(xs)))

        if rows:
            print('%-12s %10s %6s %8s' % ('family', 'm3/d/pt', 'runs', 'spread'))
            for a, b, n, sp in sorted(rows, key=lambda t: -t[1]):
                print('%-12s %10.1f %6d %8.1f' % (a, b, n, sp))
            slopes = [b for _, b, _, _ in rows]
            med = st.median(slopes)
            pos = sum(1 for s in slopes if s > 0)
            print(f'\n  median within-family slope: {med:+.1f} m3/d per point'
                  f'  ({pos}/{len(slopes)} families positive)')
            results[f'within_{label}'] = {
                'median_slope': med, 'n_families': len(slopes),
                'n_positive': pos,
                'per_family': {a: b for a, b, _, _ in rows}}
        else:
            print('  no family has enough spread in this band')

        # Pooled, for contrast.
        if len(sel) >= MIN_RUNS:
            xs = [r['viol'] for r in sel]
            ys = [r['ch4'] for r in sel]
            r = ols(xs, ys)
            if r:
                ci = boot_ci(xs, ys)
                s = f'  pooled slope: {r[0]:+.1f} m3/d per point'
                if ci:
                    s += f'  [{ci[0]:+.1f}, {ci[1]:+.1f}]'
                print(s + f'   (confounds family with margin)')
                results[f'pooled_{label}'] = {
                    'slope': r[0], 'ci': ci, 'n': len(sel)}
        print()

    if out_path:
        json.dump(results, open(out_path, 'w'), indent=1)
        print(f'wrote {out_path}')


if __name__ == '__main__':
    main()

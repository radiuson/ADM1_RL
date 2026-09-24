#!/usr/bin/env python3
"""Algorithm family against penalty weight, on one response.

The two design choices the paper compares are normally reported in different
units -- an algorithm in methane, a penalty weight in violation rate -- which
leaves the comparison between them unmade.  This refers both to the same
response, the difference from the conventional envelope at matched violation
rate, and reports how much of its variance each accounts for.

    python3 scripts/effect_sizes.py
"""
from __future__ import annotations

import collections
import glob
import json
import statistics as st

import numpy as np

from build_tables import CORRECTED, DATA, WEIGHTS, envelope

ON = ('ppo', 'a2c', 'trpo', 'recurrentppo')
OFF = ('sac', 'tqc', 'ddpg', 'td3', 'crossq')


def load():
    """The reward-penalty runs, filtered as the main table filters them."""
    _, E = envelope()
    xs = [p[0] for p in E]
    ys = [p[1] for p in E]

    def delta(d):
        if not (xs[0] <= d['viol'] <= xs[-1]):
            return None
        return d['ch4'] - float(np.interp(d['viol'], xs, ys))

    R = collections.defaultdict(list)
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
    for a in list(R):
        seen = {}
        for d in sorted(R[a], key=lambda x: x.get('model', '')):
            seen.setdefault((d.get('w'), d.get('seed')), d)
        R[a] = list(seen.values())

    rows = []
    for a, runs in R.items():
        for r in runs:
            v = delta(r)
            if v is not None:
                rows.append((a, r['w'], v))
    return rows


def main():
    rows = load()
    grand = st.mean(x for _, _, x in rows)
    total = sum((x - grand) ** 2 for _, _, x in rows)

    def between(key):
        """Sum of squares between the levels of one factor."""
        g = collections.defaultdict(list)
        for a, w, x in rows:
            g[a if key == 'algo' else w].append(x)
        return sum(len(v) * (st.mean(v) - grand) ** 2 for v in g.values())

    ss_a, ss_w = between('algo'), between('weight')
    print(f'{len(rows)} runs, response = difference from the envelope (m3/d)')
    print(f'  algorithm family {100 * ss_a / total:5.1f} % of the variance')
    print(f'  penalty weight   {100 * ss_w / total:5.1f} %')

    med = collections.defaultdict(list)
    for a, _, x in rows:
        med[a].append(x)
    fam = {a: st.median(v) for a, v in med.items()}
    lo, hi = min(fam, key=fam.get), max(fam, key=fam.get)
    print(f'\nfamily medians span {fam[hi] - fam[lo]:.0f} m3/d '
          f'({hi} {fam[hi]:+.0f} to {lo} {fam[lo]:+.0f})')

    # How far the weight moves each family, for contrast with that span.
    spans = {}
    for a in med:
        by = collections.defaultdict(list)
        for f_, w, x in rows:
            if f_ == a:
                by[w].append(x)
        ms = [st.median(v) for v in by.values() if v]
        if len(ms) > 1:
            spans[a] = max(ms) - min(ms)
    print(f'weight span within a family: median {st.median(spans.values()):.0f}'
          f', {min(spans.values()):.0f} to {max(spans.values()):.0f} m3/d')
    for a in sorted(spans, key=spans.get, reverse=True):
        print(f'    {a:14s} {spans[a]:6.0f}')


if __name__ == '__main__':
    main()

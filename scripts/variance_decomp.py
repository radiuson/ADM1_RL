#!/usr/bin/env python3
"""Variance decomposition of the envelope difference over family and weight.

Section 6 refers both design choices to one response and reports the share of
the variance each accounts for.  Those shares depend on choices the text has
to state: the order the factors enter a sequential decomposition, the sum-of-
squares definition, and what becomes of the runs that fall outside the
envelope's support and so have no response.

This computes all of them: Type I with either factor first, and Type II and
Type III, each with the runs outside the support dropped and again with the
design restricted to the cells that are complete.  The point is not to pick
one number but to show how far the conclusion moves across the choices.

    python3 scripts/variance_decomp.py [--source DIR]
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_tables import DATA, WEIGHTS, CORRECTED

ON = ('ppo', 'a2c', 'trpo', 'recurrentppo')
OFF = ('sac', 'tqc', 'ddpg', 'td3', 'crossq')
# ARS is in the decomposition: the reported shares are over all ten families
# of Table 2, not the nine gradient-based ones.
FAMS = ON + OFF + ('ars',)


def envelope_pts():
    # One definition of the conventional envelope, in build_tables, so the
    # dual-loop family reaches every script that scores against it.
    from build_tables import envelope as _env
    _, E = _env()
    return [a for a, _, _ in E], [b for _, b, _ in E]


def load(source=None):
    """-> rows of (family, weight, delta), one per run of the main design."""
    xs, ys = envelope_pts()
    src = source or f'{DATA}/evres'
    seen = {}
    for f in glob.glob(f'{src}/*.json'):
        d = json.load(open(f))
        if source is None:
            if d.get('steps') != 150000 or '_h5_' in d.get('model', ''):
                continue
            a = (d.get('algo') or 'sac').lower()
            if a in CORRECTED and CORRECTED[a] not in d.get('model', ''):
                continue
        else:
            a = (d.get('algo') or 'sac').lower()
        if a not in FAMS or d.get('w') not in WEIGHTS:
            continue
        seen.setdefault((a, d['w'], d.get('seed')), d)
    rows, dropped = [], 0
    for (a, w, _), d in seen.items():
        v = d['viol']
        if not (xs[0] <= v <= xs[-1]):
            dropped += 1
            continue
        rows.append((a, w, d['ch4'] - float(np.interp(v, xs, ys))))
    return rows, dropped, len(seen)


def design_matrix(rows, terms):
    """Dummy-coded model matrix for the requested terms, with intercept."""
    fams = sorted({r[0] for r in rows})
    wts = sorted({r[1] for r in rows})
    cols = [np.ones(len(rows))]
    if 'family' in terms:
        for a in fams[1:]:
            cols.append(np.array([1.0 if r[0] == a else 0.0 for r in rows]))
    if 'weight' in terms:
        for w in wts[1:]:
            cols.append(np.array([1.0 if r[1] == w else 0.0 for r in rows]))
    if 'inter' in terms:
        for a in fams[1:]:
            for w in wts[1:]:
                cols.append(np.array([1.0 if (r[0] == a and r[1] == w) else 0.0
                                      for r in rows]))
    return np.column_stack(cols)


def rss(rows, terms):
    y = np.array([r[2] for r in rows])
    X = design_matrix(rows, terms)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(((y - X @ beta) ** 2).sum())


def decompose(rows, order):
    """Sequential (Type I) shares for the given factor order."""
    y = np.array([r[2] for r in rows])
    sst = float(((y - y.mean()) ** 2).sum())
    out, have = {}, []
    base = rss(rows, have)
    for t in order:
        have = have + [t]
        r = rss(rows, have)
        out[t] = 100.0 * (base - r) / sst
        base = r
    out['residual'] = 100.0 * base / sst
    return out


def typeiii(rows):
    """Each term's share given every other term, as a share of total SS."""
    y = np.array([r[2] for r in rows])
    sst = float(((y - y.mean()) ** 2).sum())
    full = ['family', 'weight', 'inter']
    rfull = rss(rows, full)
    out = {}
    for t in full:
        out[t] = 100.0 * (rss(rows, [x for x in full if x != t]) - rfull) / sst
    out['residual'] = 100.0 * rfull / sst
    return out


def balanced(rows):
    """Restrict to the (family, weight) cells that kept every seed."""
    cnt = defaultdict(int)
    for a, w, _ in rows:
        cnt[(a, w)] += 1
    n = max(cnt.values())
    keep = {k for k, v in cnt.items() if v == n}
    return [r for r in rows if (r[0], r[1]) in keep], n, len(keep)


def report(rows, label):
    print(f'--- {label}: {len(rows)} runs ---')
    for order in (['family', 'weight', 'inter'], ['weight', 'family', 'inter']):
        d = decompose(rows, order)
        print('  Type I, %-14s family %5.1f%%  weight %5.1f%%  '
              'interaction %5.1f%%  residual %5.1f%%'
              % (order[0] + ' first', d['family'], d['weight'],
                 d['inter'], d['residual']))
    d = typeiii(rows)
    print('  Type III           family %5.1f%%  weight %5.1f%%  '
          'interaction %5.1f%%  residual %5.1f%%'
          % (d['family'], d['weight'], d['inter'], d['residual']))


ON_CLS = ON
OFF_CLS = OFF


def as_class(rows, drop_free=True):
    """Relabel the family factor as the two-level update-rule factor.

    The share reported in the text for `family` is a ten-level factor and so
    measures the choice of algorithm.  The class claim is a separate thing,
    and this is the factor that carries it.  ARS belongs to neither class and
    is dropped, which is what makes the factor two-level.
    """
    out = []
    for f, w, d in rows:
        if f in ON_CLS:
            out.append(('on', w, d))
        elif f in OFF_CLS:
            out.append(('off', w, d))
        elif not drop_free:
            out.append(('free', w, d))
    return out


def main():
    source = None
    if '--source' in sys.argv:
        source = sys.argv[sys.argv.index('--source') + 1]
    rows, dropped, total = load(source)
    print(f'{total} runs in the design, {dropped} outside the envelope\'s '
          f'support and without a response, {len(rows)} scored')
    if dropped:
        miss = defaultdict(int)
        for f in glob.glob(f'{source or DATA + "/evres"}/*.json'):
            pass
    report(rows, 'all scored runs')
    br, n, ncell = balanced(rows)
    report(br, f'complete cells only ({ncell} cells at {n} seeds)')
    grad = [r for r in rows if r[0] != 'ars']
    report(grad, 'nine-level family factor, ARS dropped')
    report(as_class(rows), 'two-level update-rule factor, ARS dropped')


if __name__ == '__main__':
    main()

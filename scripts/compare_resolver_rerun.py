#!/usr/bin/env python3
"""What the solver re-run changes, measured against the same envelope.

The 240 runs that predate the 17 September solver change were retrained on
the current dynamics.  Every number in the paper that rests on those runs has
to be recomputed, and the recomputation is only meaningful if the archived and
the retrained runs are scored the same way: one conventional envelope, one
interpolation rule, one set of family definitions.

This reports the quantities the manuscript states, computed first from the
archived records and then with the retrained ones substituted for the runs
they replace, so the two columns can be read side by side.

    python3 scripts/compare_resolver_rerun.py [--json out.json]

The conventional envelope and the constrained families are untouched by the
re-run -- it covers the reward-penalty families only -- so the zero-violation
comparison is carried through unchanged as a control: if it moves, something
other than the re-run has changed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from build_tables import CORRECTED, WEIGHTS, boot, envelope
from plot_figures import CMDP_OFF, OFF_POLICY, ON_POLICY, load_cmdp

RERUN = os.environ.get('RERUN_DIR') or os.path.join(ROOT, 'results_resolver', 'evres')
ARCHIVE = os.path.join(ROOT, 'paper_data', 'evres')


def load_runs(source, replace_with=None):
    """The main reward-penalty design: 150k steps, four weights, no stacking.

    ``replace_with`` supplies retrained records keyed by (algo, weight, seed);
    where one exists it stands in for the archived run of the same cell, which
    is what the re-run is for.
    """
    rows = {}
    for f in glob.glob(f'{source}/*.json'):
        d = json.load(open(f))
        if d.get('steps') != 150000 or '_h5_' in d.get('model', ''):
            continue
        a = (d.get('algo') or 'sac').lower()
        if a not in ON_POLICY + OFF_POLICY or d.get('w') not in WEIGHTS:
            continue
        # Four families have runs from a superseded batch whose action noise
        # and warm-up are not on this design's scale; build_tables names the
        # directory each is read from, and the same choice applies here.
        if a in CORRECTED and CORRECTED[a] not in d.get('model', ''):
            continue
        rows.setdefault((a, d.get('w'), d.get('seed')), d)
    if replace_with:
        swapped = 0
        for k, d in replace_with.items():
            if k in rows:
                swapped += 1
            rows[k] = d
        return rows, swapped
    return rows, 0


def load_rerun():
    """Retrained records, keyed the way the archived ones are."""
    out = {}
    for f in glob.glob(f'{RERUN}/*.json'):
        d = json.load(open(f))
        a = (d.get('algo') or 'sac').lower()
        # The same design filter the archived side uses: a retrained run at a
        # weight outside the swept grid would otherwise enter as a new cell
        # rather than replacing one, and change the counts it is meant to
        # leave alone.
        if a not in ON_POLICY + OFF_POLICY or d.get('w') not in WEIGHTS:
            continue
        out[(a, d.get('w'), d.get('seed'))] = d
    return out


def summarise(rows, xs, ys):
    """The manuscript's reward-penalty quantities for one set of runs."""
    fam = {}
    scored = above = 0
    for (a, _, _), d in rows.items():
        v = d['viol']
        if not (xs[0] <= v <= xs[-1]):
            continue
        scored += 1
        delta = d['ch4'] - float(np.interp(v, xs, ys))
        if delta > 0:
            above += 1
        fam.setdefault(a, []).append(delta)

    med = {a: st.median(v) for a, v in fam.items()}
    on = [d for a in ON_POLICY if a in fam for d in fam[a]]
    off = [d for a in OFF_POLICY if a in fam for d in fam[a]]
    return {
        'n_runs': len(rows),
        'n_scored': scored,
        'n_above': above,
        'family_median': med,
        'on_median': st.median(on) if on else None,
        'off_median': st.median(off) if off else None,
        'class_gap': (st.median(on) - st.median(off)) if on and off else None,
        'zero_violation': sum(1 for d in rows.values() if d['viol'] < 0.01),
    }


def control_zero_violation(xs, ys):
    """The constrained families at zero violations -- untouched by the re-run."""
    best = max(c for v, c in zip(xs, ys) if v < 0.01)
    Rc = load_cmdp()
    on = [c for f, v in Rc.items() if f not in CMDP_OFF
          for c in [d['ch4'] for d in v if d['viol'] < 0.01]]
    los, his = [], []
    for s in range(20):
        lo, hi = boot([c - best for c in on], random.Random(s), B=20000)
        los.append(lo)
        his.append(hi)
    return {'conventional_best': best, 'n': len(on),
            'median_gain': st.median(on) - best,
            'ci': [min(los), max(his)],
            'above': sum(1 for c in on if c > best)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', help='also write the comparison here')
    args = ap.parse_args()

    B, E = envelope()
    xs = [e[0] for e in E]
    ys = [e[1] for e in E]
    print(f'envelope {len(B)} evaluations -> {len(E)} points, '
          f'{xs[0]:.2f}-{xs[-1]:.1f}%')

    rerun = load_rerun()
    if not rerun:
        print(f'\nno retrained records in {RERUN} yet -- nothing to compare')
        return
    print(f'{len(rerun)} retrained records')

    before, _ = load_runs(ARCHIVE)
    after, swapped = load_runs(ARCHIVE, replace_with=rerun)
    print(f'{swapped} of {len(before)} design cells replaced\n')

    b = summarise(before, xs, ys)
    a = summarise(after, xs, ys)

    print('%-26s %>12s %>12s' .replace('>', '') % ('quantity', 'archived', 'retrained'))
    print('-' * 52)
    rows = [('scored runs', 'n_scored', '%d'),
            ('runs above the envelope', 'n_above', '%d'),
            ('zero-violation runs', 'zero_violation', '%d'),
            ('on-policy median', 'on_median', '%+.0f'),
            ('off-policy median', 'off_median', '%+.0f'),
            ('class gap (on - off)', 'class_gap', '%+.0f')]
    for label, key, fmt in rows:
        bv, av = b[key], a[key]
        sb = fmt % bv if bv is not None else '-'
        sa = fmt % av if av is not None else '-'
        flag = '' if sb == sa else '   <-'
        print('%-26s %12s %12s%s' % (label, sb, sa, flag))

    print('\nfamily medians (m3/d from the envelope)')
    for fam in ON_POLICY + OFF_POLICY:
        bv = b['family_median'].get(fam)
        av = a['family_median'].get(fam)
        if bv is None and av is None:
            continue
        sb = '%+.0f' % bv if bv is not None else '-'
        sa = '%+.0f' % av if av is not None else '-'
        flag = '' if sb == sa else '   <-'
        print('  %-14s %10s %12s%s' % (fam, sb, sa, flag))

    ctl = control_zero_violation(xs, ys)
    print('\ncontrol -- constrained families at zero violations '
          '(the re-run does not touch these)')
    print('  conventional best      %d m3/d' % ctl['conventional_best'])
    print('  on-policy gain         %+.0f m3/d, bootstrap [%+.0f, %+.0f]'
          % (ctl['median_gain'], ctl['ci'][0], ctl['ci'][1]))
    print('  runs above it          %d/%d' % (ctl['above'], ctl['n']))

    if args.json:
        json.dump({'archived': b, 'retrained': a, 'control': ctl,
                   'n_rerun': len(rerun), 'cells_replaced': swapped},
                  open(args.json, 'w'), indent=1, default=float)
        print(f'\nwrote {args.json}')


if __name__ == '__main__':
    main()

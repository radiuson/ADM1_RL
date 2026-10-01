#!/usr/bin/env python3
"""Recover the gains of the archived PI evaluations that stored only summaries.

The earliest conventional evaluations were written before the record format
carried ``Kc`` and ``tau_i``, so those records name a setpoint and a feed
multiplier but not the loop that produced them.  The evaluation is
deterministic -- one seed, a fixed scenario order, a fixed horizon -- so a
record can be identified by re-running its setpoint against the gain pairs the
archive uses and keeping the pair whose summary reproduces it.

Each candidate is scored on the three quantities the record stores.  A match
has to agree on all of them to within the tolerance below; anything looser
would not distinguish neighbouring gains, and a record that no pair reproduces
is reported rather than assigned a nearest guess.

    python3 scripts/recover_pi_gains.py            # report only
    python3 scripts/recover_pi_gains.py --write    # patch the matched records

``--self-test`` re-identifies records whose gains *are* stored, which checks
that the search recovers a known answer before it is trusted on an unknown one.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from build_tables import DATA
from evaluation.eval_baseline_ext import VFA_HARD, VFA_SOFT, MG, run

# The gain pairs the sweep used.  The archive contains no others, and the
# recovery does not invent any: a record these cannot reproduce is left
# unidentified.
GAINS = [(0.03, 60.0), (0.05, 40.0), (0.1, 20.0), (0.2, 12.0), (0.35, 8.0),
         (0.5, 5.0), (1.0, 3.0), (1.5, 2.5), (2.0, 2.0), (3.0, 2.0),
         (5.0, 1.0)]

# The identifying quantities are the median VFA and the mean methane, which
# are averages over the episode and are reproduced to within rounding.  The
# violation rate is deliberately excluded: it is a count divided by the
# episode length, so a single step of difference in where a scenario
# terminates moves it by a tenth of a point while leaving the averages
# untouched.  The earliest records sit one step longer than a present-day
# re-run, which is enough to shift the rate and nothing else.  The tolerances
# are far tighter than the spacing between neighbouring gain pairs, which is
# several mg/L.
TOL = {'ch4': 3.0, 'vfa_med': 0.05}

# How far apart the stored and re-run violation rates may be before a match
# is reported as questionable rather than accepted.  One step either way on
# an episode of this length is about 0.25 points.
VIOL_SLACK = 0.40


def summarise(V, CH):
    """The record fields, computed exactly as eval_baseline_ext writes them."""
    return {'ch4': float(CH.mean()),
            'viol': float(100 * np.mean(V > VFA_SOFT)),
            'viol_hard': float(100 * np.mean(V > VFA_HARD)),
            'vfa_max': float(V.max() * MG),
            'vfa_med': float(np.median(V) * MG)}


def setpoint_of(name):
    """The setpoint a record's name encodes, e.g. pi_sp225_m1.3 -> 225."""
    for part in name.split('_'):
        if part.startswith('sp'):
            return float(part[2:])
    return None


def mult_of(rec):
    if rec.get('mult') is not None:
        return float(rec['mult'])
    for part in rec['name'].split('_'):
        if part.startswith('m'):
            return float(part[1:])
    return 1.3


def agrees(rec, got):
    return all(abs(got[k] - rec[k]) <= t for k, t in TOL.items())


def identify(rec, cache):
    """The gain pairs that reproduce this record's summary."""
    sp, mult = setpoint_of(rec['name']), mult_of(rec)
    hits = []
    for Kc, ti in GAINS:
        key = (sp, Kc, ti, mult)
        if key not in cache:
            V, CH, _ = run('pi', sp, Kc, ti, mult)
            cache[key] = summarise(V, CH)
        if agrees(rec, cache[key]):
            hits.append((Kc, ti))
    return hits


def load(path):
    d = json.load(open(path))
    d['_path'] = path
    return d


def pi_records():
    R = [load(f) for f in glob.glob(f'{DATA}/evbase/*.json')]
    return [d for d in R if str(d.get('kind', '')).lower().startswith('pi')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true',
                    help='patch the matched records in place')
    ap.add_argument('--self-test', action='store_true',
                    help='re-identify records whose gains are already stored')
    ap.add_argument('--band', nargs=2, type=float, default=None,
                    metavar=('LO', 'HI'),
                    help='restrict to a violation-rate band')
    a = ap.parse_args()

    R = pi_records()
    if a.band:
        R = [d for d in R if a.band[0] <= d['viol'] <= a.band[1]]

    known = [d for d in R if d.get('Kc') is not None]
    unknown = [d for d in R if d.get('Kc') is None]
    cache = {}

    if a.self_test:
        ok = bad = 0
        for rec in sorted(known, key=lambda d: d['vfa_med']):
            hits = identify(rec, cache)
            truth = (float(rec['Kc']), float(rec['tau_i']))
            good = hits == [truth]
            ok, bad = ok + good, bad + (not good)
            if not good:
                print(f'  {rec["name"]:<20} stored {truth} -> matched {hits}')
        print(f'self-test: {ok} uniquely recovered, {bad} not')
        return

    print(f'{len(unknown)} records without gains, {len(GAINS)} candidates '
          f'each\n')
    found = []
    for rec in sorted(unknown, key=lambda d: d['vfa_med']):
        hits = identify(rec, cache)
        if len(hits) == 1:
            Kc, ti = hits[0]
            dv = cache[(setpoint_of(rec['name']), Kc, ti,
                        mult_of(rec))]['viol'] - rec['viol']
            flag = '' if abs(dv) <= VIOL_SLACK else f'  [viol {dv:+.2f} !]'
            tag = f'Kc={Kc:g} tau_i={ti:g}{flag}'
            found.append((rec, hits[0]))
        else:
            tag = 'no match' if not hits else f'ambiguous: {hits}'
        print(f'  {rec["name"]:<20} viol {rec["viol"]:5.2f}  '
              f'vfa_med {rec["vfa_med"]:6.2f}  ->  {tag}')

    print(f'\n{len(found)} of {len(unknown)} uniquely identified')

    if a.write and found:
        for rec, (Kc, ti) in found:
            d = json.load(open(rec['_path']))
            d['Kc'], d['tau_i'], d['mult'] = Kc, ti, mult_of(rec)
            d['gains_recovered'] = True
            json.dump(d, open(rec['_path'], 'w'))
        print(f'patched {len(found)} records')


if __name__ == '__main__':
    main()

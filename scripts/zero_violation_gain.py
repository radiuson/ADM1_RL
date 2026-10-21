#!/usr/bin/env python3
"""The zero-violation comparison: constrained families against the best tuning.

The envelope stops at a bound where no excursion is permitted, and the
comparison the manuscript opens with is between that bound and the
constrained families that reach the same rate.  This reports the quantities
Section 5.1 and the abstract state, so they rest on a script rather than on
an ad-hoc calculation.

The estimand is the difference between the pooled-run class median and the
highest conventional methane flow at zero violation.  Pooling weights each
family by the number of runs it lands there, which is the same convention
the class comparisons elsewhere use; the interval resamples families and
then runs within a family, so a family's runs stay together and the
interval is not narrowed by treating them as independent draws.

    python3 scripts/zero_violation_gain.py
"""
from __future__ import annotations

import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_tables import envelope
from plot_figures import CMDP_OFF, load_cmdp

SEED = 20260919
B = 4000


def main() -> None:
    _, E = envelope()
    xs = [e[0] for e in E]
    ys = [e[1] for e in E]
    # The envelope's own zero-violation endpoint: the best tuning that holds
    # the limit over the whole scenario set.
    best = max(c for v, c in zip(xs, ys) if v < 0.01)

    Rc = load_cmdp()
    fams = {}
    for f, runs in Rc.items():
        if f in CMDP_OFF:
            continue
        z = [d['ch4'] for d in runs if d['viol'] < 0.01]
        if z:
            fams[f] = z
    off = {}
    for f, runs in Rc.items():
        if f not in CMDP_OFF:
            continue
        z = [d['ch4'] for d in runs if d['viol'] < 0.01]
        if z:
            off[f] = z

    on = [c for v in fams.values() for c in v]
    gain = st.median(on) - best

    rng = random.Random(SEED)
    keys = list(fams)
    draws = []
    for _ in range(B):
        fs = [rng.choice(keys) for _ in keys]
        vals = []
        for f in fs:
            r = fams[f]
            vals += [rng.choice(r) for _ in r]
        draws.append(st.median(vals) - best)
    draws.sort()

    print(f'conventional best at zero violation   {best:.0f} m3/d')
    print(f'on-policy class median                {st.median(on):.0f} m3/d '
          f'over {len(on)} runs in {len(fams)} families')
    print(f'gain over the conventional best       {gain:+.0f} m3/d  '
          f'[{draws[int(.025 * B)]:+.0f}, {draws[int(.975 * B)]:+.0f}]')
    print(f'runs above the conventional best      {sum(1 for c in on if c > best)}'
          f'/{len(on)}')
    print(f'families whose median is above it     '
          f'{sum(1 for v in fams.values() if st.median(v) > best)}/{len(fams)}'
          f' reaching zero, of 10 on-policy families')
    print()
    for f, v in sorted(fams.items(), key=lambda kv: st.median(kv[1])):
        print(f'  {f:10s} n={len(v):2d}  median {st.median(v):6.0f}  '
              f'{st.median(v) - best:+5.0f}')
    if off:
        print()
        print('off-policy families at zero violation')
        for f, v in sorted(off.items(), key=lambda kv: st.median(kv[1])):
            print(f'  {f:10s} n={len(v):2d}  median {st.median(v):6.0f}  '
                  f'{st.median(v) - best:+5.0f}')


if __name__ == '__main__':
    main()

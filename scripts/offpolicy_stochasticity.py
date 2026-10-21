#!/usr/bin/env python3
"""Policy stochasticity among the off-policy constrained families.

All four off-policy constrained families keep a replay buffer, so buffer use
cannot be what separates the two that follow the specified rate from the two
that hold almost the same rate whatever is asked of them.  What distinguishes
them is whether the policy is stochastic.  Section 7.2 reports that contrast
twice: as the span of realised rates across the four cost limits, which is a
tracking statement, and as a production difference from the envelope, which
is not resolved.

The production interval resamples runs within each pair, which is the
weakest claim the data supports here: with two families on each side there
is no family level to resample, so the interval cannot also carry the
variation between families.  It crosses zero either way, which is why the
subsection reports the contrast as a tracking result.

    python3 scripts/offpolicy_stochasticity.py
"""
from __future__ import annotations

import os
import random
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_tables import envelope
from plot_figures import load_cmdp

# The two pairs: a stochastic policy against a deterministic one.
STOCHASTIC = ('SACLag', 'SACPID')
DETERMINISTIC = ('DDPGLag', 'TD3Lag')
LIMITS = ('0.6', '1.8', '3.0', '6.0')
SEED = 20260919
B = 4000


def main() -> None:
    _, E = envelope()
    xs = [e[0] for e in E]
    ys = [e[1] for e in E]
    Rc = load_cmdp()

    def deltas(fam):
        return [d['ch4'] - float(np.interp(d['viol'], xs, ys))
                for d in Rc[fam] if xs[0] <= d['viol'] <= xs[-1]]

    print('realised violation rate by cost limit (family medians, points)')
    for fam in DETERMINISTIC + STOCHASTIC:
        by = {}
        for d in Rc[fam]:
            by.setdefault(str(d.get('cost_limit')), []).append(d['viol'])
        med = [st.median(by[k]) for k in LIMITS if k in by]
        kind = 'stochastic' if fam in STOCHASTIC else 'deterministic'
        print(f'  {fam:9s} {kind:13s} '
              f'{"  ".join(f"{v:5.2f}" for v in med)}   '
              f'span {max(med) - min(med):.2f} pp')

    sto = [x for f in STOCHASTIC for x in deltas(f)]
    det = [x for f in DETERMINISTIC for x in deltas(f)]
    gap = st.median(sto) - st.median(det)

    rng = random.Random(SEED)
    draws = []
    for _ in range(B):
        a = [rng.choice(sto) for _ in sto]
        b = [rng.choice(det) for _ in det]
        draws.append(st.median(a) - st.median(b))
    draws.sort()
    lo = draws[int(.025 * B)]
    hi = draws[int(.975 * B)]

    print()
    print(f'production relative to the envelope, m3/d')
    print(f'  stochastic     median {st.median(sto):+5.0f} over {len(sto)} runs')
    print(f'  deterministic  median {st.median(det):+5.0f} over {len(det)} runs')
    print(f'  difference     {gap:+.0f}  [{lo:+.0f}, {hi:+.0f}]  '
          f'{"includes" if lo <= 0 <= hi else "excludes"} zero')


if __name__ == '__main__':
    main()

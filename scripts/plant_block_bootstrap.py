#!/usr/bin/env python3
"""Correlations in the plant record, with a block bootstrap for serial dependence.

The daily plant series are strongly autocorrelated, so the number of
independent observations is far below the number of rows.  A p-value computed
as if the 1103 daily records were independent is therefore not interpretable,
and for a correlation as small as the feed-VFA one it is the p-value that the
argument rests on.  A moving-block bootstrap resamples contiguous blocks
instead of individual days, which preserves the within-block dependence; the
block length is set from the integrated autocorrelation time of the two series.

    python3 scripts/plant_block_bootstrap.py
"""
from __future__ import annotations

import csv
import math
import os
import random

import numpy as np

# The raw laboratory file, not the interpolated one: interpolation fills the
# gaps between sampling days, which inflates the effective sample size and
# further inflates the serial dependence this script is written to account for.
CSV = os.path.expanduser(
    '~/code/biogas/datasets/muscatine/'
    'hunter-schroer-municipal-biogas-forecasting-199f913/'
    'LAB/LABS-raw.csv')

FEED = ['V-TWAS_gal', 'V-PS_gal', 'V-HSW_gal', 'FOG_gal']
PAIRS = [('feed', 'Biogas'), ('feed', 'Dig1-VFA_mgL'), ('feed', 'Dig2-VFA_mgL')]


def load():
    rows = list(csv.DictReader(open(CSV)))
    out = {}
    feed = []
    for r in rows:
        try:
            feed.append(sum(float(r[c]) for c in FEED))
        except (ValueError, KeyError):
            feed.append(float('nan'))
    out['feed'] = np.array(feed)
    for c in ('Biogas', 'Dig1-VFA_mgL', 'Dig2-VFA_mgL'):
        out[c] = np.array([float(r[c]) if r.get(c) not in (None, '') else np.nan
                           for r in rows])
    return out


def ranks(x):
    order = np.argsort(x, kind='mergesort')
    r = np.empty(len(x), float)
    r[order] = np.arange(1, len(x) + 1)
    return r


def spearman(x, y):
    a, b = ranks(x), ranks(y)
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d else 0.0


def act(x, max_lag=60):
    """Integrated autocorrelation time of a mean-centred series."""
    z = x - x.mean()
    v = (z * z).mean()
    if v == 0:
        return 1.0
    tau = 1.0
    for k in range(1, min(max_lag, len(z) // 4)):
        c = float((z[:-k] * z[k:]).mean() / v)
        if c <= 0.05:
            break
        tau += 2 * c
    return max(1.0, tau)


def block_boot(x, y, L, B=4000, seed=20260919):
    """Moving-block bootstrap of Spearman rho; returns the resample spread."""
    rng = random.Random(seed)
    n = len(x)
    nb = int(math.ceil(n / L))
    out = []
    for _ in range(B):
        idx = []
        for _ in range(nb):
            s = rng.randrange(0, n - L + 1)
            idx.extend(range(s, s + L))
        idx = np.array(idx[:n])
        out.append(spearman(x[idx], y[idx]))
    out.sort()
    return out


def main():
    D = load()
    print(f'{"pair":<26}{"n":>6}{"rho":>8}{"block":>7}'
          f'{"naive p":>10}{"block 95% CI":>20}')
    print('-' * 78)
    for a, b in PAIRS:
        x, y = D[a], D[b]
        m = ~(np.isnan(x) | np.isnan(y))
        x, y = x[m], y[m]
        n = len(x)
        r = spearman(x, y)
        L = int(round(max(act(x), act(y))))
        L = max(2, min(L, n // 10))
        t = r * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
        # two-sided normal approximation, adequate at this n
        p = math.erfc(abs(t) / math.sqrt(2))
        bs = block_boot(x, y, L)
        lo, hi = bs[int(.025 * len(bs))], bs[int(.975 * len(bs))]
        mark = '' if lo <= 0 <= hi else '  excludes 0'
        print(f'{a + " vs " + b:<26}{n:>6}{r:>+8.3f}{L:>7}'
              f'{p:>10.1e}   [{lo:+.3f}, {hi:+.3f}]{mark}')
    print('\nBlock length from the integrated autocorrelation time of the two '
          'series.\nThe naive p-value treats each day as independent and is '
          'reported only\nfor comparison.')


if __name__ == '__main__':
    main()

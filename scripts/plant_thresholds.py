#!/usr/bin/env python3
"""Exceedance counts in the plant VFA record, with their sample definition.

Section "Scope of the plant anchoring" maps the model's 300 mg/L soft limit
onto the plant scale and reports how often the plant record exceeds the
mapped endpoints.  Those rates depend on which rows count as an observation,
so this script prints the numerator and the denominator of every one of them
rather than the percentages alone.

The record carries two VFA columns, one per digester.  The published series
does not say how feed is divided between them, so a simulated day cannot be
paired with a recorded day; the sample is therefore the set of digester-day
observations -- up to two per calendar day -- and every rate below is a share
of those, not of calendar days.

    python3 scripts/plant_thresholds.py
"""
from __future__ import annotations

import csv
import os

# The raw laboratory file, not the interpolated one: interpolation invents
# readings on days that were not sampled.
CSV = os.path.expanduser(
    '~/code/biogas/datasets/muscatine/'
    'hunter-schroer-municipal-biogas-forecasting-199f913/'
    'LAB/LABS-raw.csv')

COLS = ('Dig1-VFA_mgL', 'Dig2-VFA_mgL')

# The two mapped endpoints come from dividing the model's limit by each end of
# the simulated/recorded VFA ratio; the other two are the levels quoted from
# the facility's own record.
LEVELS = (300, 1579, 2000, 3000, 3750)


def main() -> None:
    rows = list(csv.DictReader(open(CSV)))

    def value(row, col):
        v = row.get(col)
        return float(v) if v not in (None, '') else None

    per_day = [[value(r, c) for c in COLS] for r in rows]
    obs = [v for day in per_day for v in day if v is not None]
    any_day = sum(1 for day in per_day if any(v is not None for v in day))
    both_day = sum(1 for day in per_day if all(v is not None for v in day))

    print(f'calendar days in file          {len(rows)}')
    print(f'days with at least one reading {any_day}')
    print(f'days with both digesters       {both_day}')
    print(f'digester-day observations      {len(obs)}')
    print()
    print('threshold   exceeding   share of observations')
    for lv in LEVELS:
        n = sum(1 for v in obs if v > lv)
        print(f'{lv:>7} mg/L {n:>10}   {100.0 * n / len(obs):>5.1f}%')


if __name__ == '__main__':
    main()

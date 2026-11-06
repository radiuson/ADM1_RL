#!/usr/bin/env python3
"""How close does any evaluation episode come to the VFA observation bound?

Section 3.3 clips an observation outside its declared bounds, so a run that
drove total VFA past 2.5 kg COD/m3 would be invisible to the policy at that
step.  ``scripts/observation_bounds.py`` answers the same question from a
corner sweep of the action space; this one answers it from the evaluation
records themselves, which cover the policies that were actually trained.

Every record that logs a peak total VFA is read, whichever round produced it,
and the largest value is reported with the record it came from.  The records
store mg HAc-eq/L; the observation bound is in kg COD/m3.

    ADM1_PAPER_DATA=paper_data_merged python3 scripts/evaluation_vfa_peak.py
"""
from __future__ import annotations

import glob
import json
import math
import os

MG_TO_KGCOD = 1.067 / 1000.0          # 1 mg HAc-eq = 1.067 mg COD
VFA_OBS_BOUND = 2.5                   # kg COD/m3, Section 3.3


def main() -> None:
    root = os.environ.get('ADM1_PAPER_DATA', 'paper_data_merged')
    peaks = []
    for path in sorted(glob.glob(os.path.join(root, '**', '*.json'),
                                 recursive=True)):
        with open(path) as fh:
            try:
                blob = json.load(fh)
            except json.JSONDecodeError:
                continue
        for rec in (blob if isinstance(blob, list) else [blob]):
            if not isinstance(rec, dict):
                continue
            v = rec.get('vfa_max')
            if isinstance(v, (int, float)) and not math.isnan(v):
                peaks.append((v, os.path.relpath(path, root)))

    if not peaks:
        print(f'no records with a logged VFA peak under {root}')
        return

    peaks.sort(reverse=True)
    by_round = {}
    for v, rel in peaks:
        by_round.setdefault(rel.split(os.sep)[0], []).append(v)

    print(f'{len(peaks)} evaluation records log a peak total VFA\n')
    for name in sorted(by_round):
        vs = by_round[name]
        print(f'  {name:12s} n={len(vs):4d}  '
              f'peak {max(vs) * MG_TO_KGCOD:.3f} kg COD/m3')

    top, rel = peaks[0]
    kg = top * MG_TO_KGCOD
    print(f'\nhighest overall  {kg:.3f} kg COD/m3  ({top:.1f} mg HAc-eq/L)'
          f'  in {rel}')
    print(f'observation bound {VFA_OBS_BOUND} kg COD/m3  ->  '
          f'{"CLIPPED" if kg > VFA_OBS_BOUND else "not reached"}')


if __name__ == '__main__':
    main()

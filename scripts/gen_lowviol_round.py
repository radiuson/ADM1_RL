#!/usr/bin/env python3
"""Queue the training runs that fill the low-violation corner of the sweep.

Below about 5 % violation the reward-penalty comparison rests on almost no
on-policy runs: of the sixteen in the 0--5 % band, thirteen are A2C at a
heavy penalty weight, where the policy collapses to near-zero feed rather
than learning a tighter distribution.  The region is the one a digester
would actually be operated in, so the class comparison should not be blank
there.

The swept weights stop at 5.  Weights of 10 and 20 already exist as reward
configurations and were run for TRPO and SAC at three seeds each, which is
enough to show both families stay above the envelope down to about 2 %
violation but not enough to compare classes.  This extends both weights to
the ten seeds the main sweep uses, across all nine gradient-based families.

    python3 scripts/gen_lowviol_round.py <out.txt>      # write the job list
    python3 scripts/gen_lowviol_round.py <out.txt> --dry # list, do not write

Each line is a shell command; feed the file to xargs as the other campaign
scripts do.  Runs that already exist are skipped, so the script is safe to
re-run after an interruption.
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from build_tables import DATA

PY = '/home/ihpc/anaconda3/envs/biosim/bin/python3'
OUTDIR = os.path.join(ROOT, 'models_lowviol')

# The nine gradient-based families; ARS has no penalty-weight response worth
# extending here and is reported separately throughout.
FAMILIES = ['ppo', 'trpo', 'recurrentppo', 'a2c',
            'sac', 'tqc', 'ddpg', 'td3', 'crossq']
WEIGHTS = ['lw10', 'lw20']
SEEDS = list(range(42, 52))          # the ten the main sweep uses
STEPS = 150000                        # matches the reported budget


def existing():
    """(algo, weight, seed) already evaluated, from the archive."""
    done = set()
    for f in glob.glob(f'{DATA}/evres/*.json'):
        d = json.load(open(f))
        if d.get('w') in WEIGHTS and d.get('steps') == STEPS:
            done.add((d['algo'].lower(), d['w'], d.get('seed')))
    return done


def main():
    out_path = sys.argv[1]
    dry = '--dry' in sys.argv
    done = existing()

    jobs = []
    for w in WEIGHTS:
        for algo in FAMILIES:
            for seed in SEEDS:
                if (algo, w, seed) in done:
                    continue
                # The run directory also tells eval_sb3 what it is looking
                # at, so it follows the naming the archive already uses.
                jobs.append(
                    f"{PY} training/train_sac_std_cur.py"
                    f" --reward-config {w} --algo {algo} --seed {seed}"
                    f" --timesteps {STEPS} --output-dir {OUTDIR}"
                    f" --obs-mode scada --verbose 0"
                    f" >> {ROOT}/logs_campaign/lowviol_{w}_{algo}_{seed}.log 2>&1"
                )

    print(f'{len(done)} runs already present, {len(jobs)} to run')
    by_w = {w: sum(1 for j in jobs if f'--reward-config {w} ' in j)
            for w in WEIGHTS}
    for w, n in by_w.items():
        print(f'   {w}: {n}')

    if dry:
        for j in jobs[:3]:
            print('  ', j[:120], '...')
        return

    os.makedirs(os.path.join(ROOT, 'logs_campaign'), exist_ok=True)
    with open(out_path, 'w') as fh:
        fh.write('\n'.join(jobs) + ('\n' if jobs else ''))
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()

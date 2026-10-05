#!/usr/bin/env python3
"""Queue the 600k-step runs that test why memory does not help.

Section 7 reports that memory widens the class gap, and the archive gives
two signs that the memory variants are short of data rather than badly
optimised: doubling the budget to 300k moves RecurrentPPO from -26 to -12
while moving PPO the other way, from +27 to +13, and the recurrent runs are
the more consistent across seeds, 21 against 40 m3/d, which is what a model
that has not used its capacity looks like rather than one that is unstable.

Doubling again separates the two readings.  If the shortfall is capacity
that 150k steps could not fill, the gap keeps closing; if it stops at a
negative value, memory costs something the budget cannot buy back.  PPO is
run at the same budget because without it a change in RecurrentPPO cannot be
told from a change the budget would produce in any family.

The design mirrors the 300k ablation already reported: four penalty weights,
three seeds, two families, 24 runs.

    python3 scripts/gen_budget600k_round.py <out.txt> [--dry]
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
OUTDIR = os.path.join(ROOT, 'models_budget600k')

FAMILIES = ['recurrentppo', 'ppo']
WEIGHTS = ['lw0p5', 'lw1', 'lw2', 'lw5']   # as in the 300k ablation
SEEDS = [42, 43, 44]
STEPS = 600000


def existing():
    done = set()
    for f in glob.glob(f'{DATA}/evres/*.json'):
        d = json.load(open(f))
        if d.get('steps') == STEPS:
            done.add((d['algo'].lower(), d.get('w'), d.get('seed')))
    return done


def main():
    out_path = sys.argv[1]
    dry = '--dry' in sys.argv
    done = existing()

    jobs = []
    for algo in FAMILIES:
        for w in WEIGHTS:
            for seed in SEEDS:
                if (algo, w, seed) in done:
                    continue
                jobs.append(
                    f"{PY} training/train_sac_std_cur.py"
                    f" --reward-config {w} --algo {algo} --seed {seed}"
                    f" --timesteps {STEPS} --output-dir {OUTDIR}"
                    f" --obs-mode scada --verbose 0"
                    f" >> {ROOT}/logs_campaign/b600k_{w}_{algo}_{seed}.log 2>&1"
                )

    print(f'{len(done)} runs already at {STEPS} steps, {len(jobs)} to run')
    if dry:
        for j in jobs[:2]:
            print('  ', j[:110], '...')
        return

    os.makedirs(os.path.join(ROOT, 'logs_campaign'), exist_ok=True)
    with open(out_path, 'w') as fh:
        fh.write('\n'.join(jobs) + ('\n' if jobs else ''))
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()

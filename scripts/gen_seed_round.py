#!/usr/bin/env python3
"""Emit the training commands needed to bring one seed up to full coverage.

Usage: gen_seed_round.py <seed> <outfile>

Every algorithm family is covered at all four sweep points, so a round that
completes leaves the seed count equal across families.  Commands already
represented on disk are skipped, which makes the generator safe to re-run after
an interrupted round.
"""
import sys, glob, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SB3 = ['sac', 'tqc', 'ppo', 'recurrentppo', 'trpo', 'ddpg', 'td3', 'a2c', 'crossq', 'ars']
OM_ON = ['CPO', 'CPPOPID', 'CUP', 'FOCOPS', 'OnCRPO', 'P3O', 'PCPO', 'PPOLag',
         'TRPOLag', 'TRPOPID']
OM_OFF = ['SACLag', 'DDPGLag', 'TD3Lag', 'SACPID']
W = ['lw0p5', 'lw1', 'lw2', 'lw5']
CL = ['0.6', '1.8', '3.0', '6.0']
PY_BIO = '/home/ihpc/anaconda3/envs/biosim/bin/python3'
PY_SAFE = '/home/ihpc/anaconda3/envs/adm1_safe/bin/python'

def existing():
    have = set()
    for f in glob.glob(str(ROOT / 'models_*/*/run_meta.json')):
        try:
            m = json.load(open(f))
            a = m.get('algo', 'SAC').lower()
            if '_h5_' in f:
                a += '+mem'
            have.add((a, m['reward_config'], m['seed']))
        except Exception:
            pass
    for d in glob.glob(str(ROOT / 'models_cmdp/*/seed-*/')):
        try:
            c = json.load(open(os.path.join(d, 'config.json')))
            cl = (c.get('lagrange_cfgs', {}).get('cost_limit')
                  or c.get('algo_cfgs', {}).get('cost_limit'))
            have.add((c['algo'], str(cl), c['seed']))
        except Exception:
            pass
    return have

def main(seed, out):
    have = existing()
    jobs, stagger = [], 0
    for a in SB3:
        for w in W:
            if (a, w, seed) in have:
                continue
            jobs.append(
                f"{PY_BIO} training/train_sac_std_cur.py --reward-config {w} "
                f"--seed {seed} --normalize --step-size 1.0 --timesteps 150000 "
                f"--algo {a} --output-dir models_seed --device cpu --verbose 0 "
                f"> logs_campaign/s{seed}_{a}_{w}.log 2>&1")
    for a in OM_ON + OM_OFF:
        for cl in CL:
            if (a, cl, seed) in have:
                continue
            # OmniSafe names its run directory by the wall-clock second, so
            # simultaneous launches with the same seed collide and overwrite.
            jobs.append(
                f"sleep {stagger * 6}; {PY_SAFE} training/train_omnisafe.py "
                f"--algo {a} --cost-limit {cl} --seed {seed} --timesteps 150000 "
                f"--output-dir models_cmdp --device cpu "
                f"> logs_campaign/s{seed}_{a}_c{cl}.log 2>&1")
            stagger += 1
    Path(out).write_text('\n'.join(jobs) + ('\n' if jobs else ''))
    print(f"seed {seed}: {len(jobs)} jobs")

if __name__ == '__main__':
    main(int(sys.argv[1]), sys.argv[2])

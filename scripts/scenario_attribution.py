#!/usr/bin/env python3
"""Where does the zero-violation advantage come from, scenario by scenario?

Section 6.1 reports that the constrained on-policy runs reaching zero
violations produce more methane than the best conventional tuning, and offers
a mechanism: a fixed-gain loop holds one standing margin for the whole
scenario set, so it gives up production in the scenarios where the limit is
out of reach, while a learned policy can set its operating point per scenario.
Testing that requires methane per scenario, which the evaluation records do not
store -- they keep the pooled mean and a per-scenario VFA summary.

This replays both sides and records methane for each scenario separately:
the zero-violation constrained policies, and the conventional tunings that
hold the limit over the whole set.  Both replays use the evaluation protocol
of Section 5.3 (the seven scenarios, 60 steps, reset seed 7, greedy actions).

The mechanism predicts the advantage concentrates in low_load and nominal,
the two scenarios where the soft limit cannot be reached at all.

    ADM1_PAPER_DATA=paper_data_merged \
        /path/to/adm1_safe/bin/python scripts/scenario_attribution.py out.json
"""
from __future__ import annotations

import glob
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

SC = ['low_load', 'nominal', 'plant_load', 'high_load', 'peak_load',
      'fog_surge', 'elevated_start']
UNREACHABLE = ('low_load', 'nominal')
VFA_SOFT = 0.320
MG = 1000 / 1.0667
Q_LO, Q_HI = 41.0, 159.0
VFA_SPECIES = ('S_va', 'S_bu', 'S_pro', 'S_ac')


def _vfa(state) -> float:
    return sum(float(state[k]) for k in VFA_SPECIES)


def conventional(name: str) -> dict:
    """Replay one dual-loop tuning, as scripts/sweep_dualloop.py evaluates it."""
    from env.adm1_gym_env_std import ADM1Env_Std as Env
    from training.baselines import DualLoopPI

    # dual_sp140_Kc1_ti20_Km0.008_b40
    bits = name.split('_')
    sp = float(bits[1][2:])
    kc = float(bits[2][2:])
    ti = float(bits[3][2:])
    km = float(bits[4][2:])
    band = float(bits[5][1:])

    per = {}
    for s in SC:
        env = Env(s, obs_mode='scada', step_size=1.0)
        env.reset(seed=7)
        ctrl = DualLoopPI(sp, Q_LO, Q_HI, Kc=kc, tau_i=ti, Km=km, band=band)
        ctrl.reset()
        q, m = Q_HI, 1.0
        V, C = [], []
        for _ in range(60):
            st = env.solver.state
            q, m = ctrl.act(_vfa(st) * MG, st['S_IC'],
                            -np.log10(max(st['S_H_ion'], 1e-14)),
                            env.solver.q_ch4, q, m)
            _, _, te, tr, _ = env.step(np.array([q, m], dtype=np.float32))
            st = env.solver.state
            V.append(_vfa(st))
            C.append(env.solver.q_ch4)
            if te or tr:
                break
        per[s] = {'ch4': float(np.mean(C)), 'steps': len(V),
                  'viol': float(100 * np.mean(np.array(V) > VFA_SOFT))}
    return {'name': name, 'kind': 'dual', 'per_scenario': per}


def policy(run_dir: str) -> dict:
    """Replay one constrained policy, as scripts/treatment_metrics.py loads it."""
    import torch
    from omnisafe.common.normalizer import Normalizer
    from omnisafe.models.actor import ActorBuilder

    from scripts.treatment_metrics import E, NormalizedADM1Env
    from training.omnisafe_env import CMDP_REWARD

    torch.set_num_threads(1)
    run_dir = run_dir.rstrip('/')
    cfg = json.load(open(os.path.join(run_dir, 'config.json')))
    ck = sorted(glob.glob(os.path.join(run_dir, 'torch_save', 'epoch-*.pt')),
                key=lambda q: int(q.split('epoch-')[1].split('.pt')[0]))
    params = torch.load(ck[-1], map_location='cpu')

    probe = NormalizedADM1Env(E('nominal', reward_config=CMDP_REWARD,
                                obs_mode='scada', step_size=1.0))
    mc = cfg['model_cfgs']
    actor = ActorBuilder(
        obs_space=probe.observation_space, act_space=probe.action_space,
        hidden_sizes=mc['actor']['hidden_sizes'],
        activation=mc['actor']['activation'],
        weight_initialization_mode=mc['weight_initialization_mode'],
    ).build_actor(mc['actor_type'])
    actor.load_state_dict(params['pi'])
    actor.eval()
    norm = None
    if 'obs_normalizer' in params:
        norm = Normalizer(shape=probe.observation_space.shape, clip=5)
        norm.load_state_dict(params['obs_normalizer'])

    per = {}
    for s in SC:
        env = NormalizedADM1Env(E(s, reward_config=CMDP_REWARD,
                                  obs_mode='scada', step_size=1.0))
        obs, _ = env.reset(seed=7)
        base = env.env
        V, C = [], []
        for _ in range(60):
            x = torch.as_tensor(obs, dtype=torch.float32)
            if norm is not None:
                x = norm.normalize(x)
            with torch.no_grad():
                action = actor.predict(x, deterministic=True).numpy()
            obs, _, te, tr, _ = env.step(action)
            st = base.solver.state
            V.append(_vfa(st))
            C.append(base.solver.q_ch4)
            if te or tr:
                break
        per[s] = {'ch4': float(np.mean(C)), 'steps': len(V),
                  'viol': float(100 * np.mean(np.array(V) > VFA_SOFT))}
    return {'name': run_dir, 'kind': 'policy', 'algo': cfg['algo'],
            'seed': cfg.get('seed'),
            'cost_limit': cfg['algo_cfgs'].get('cost_limit'),
            'per_scenario': per}


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'scenario_attribution.json'
    root = os.environ.get('ADM1_PAPER_DATA', 'paper_data_merged')

    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__))))
    from plot_figures import CMDP_OFF

    off = {a.lower() for a in CMDP_OFF}

    # The zero-violation constrained on-policy runs, and the conventional
    # tunings that hold the limit over the whole scenario set.
    pols = []
    for f in sorted(glob.glob(os.path.join(root, 'evcmdp', '*.json'))):
        r = json.load(open(f))
        if r.get('viol', 99) >= 0.01:
            continue
        if (r.get('algo') or '').lower() in off:
            continue
        rd = r.get('run_dir')
        if rd and os.path.isdir(rd.rstrip('/')):
            pols.append(rd)

    # The conventional side comes from the same loader that builds the
    # envelope, so the set scored here is the one the envelope rests on.
    from build_tables import envelope

    base, _ = envelope()
    convs = [b['name'] for b in base
             if b['viol'] < 0.01 and b.get('kind') == 'dual']

    print(f'{len(pols)} zero-violation constrained policies, '
          f'{len(convs)} zero-violation dual-loop tunings', flush=True)

    # The conventional replays are independent and each holds one solver, so
    # they run in a pool; the policy replays each load torch and stay serial.
    rows = []
    workers = int(os.environ.get('ATTRIB_WORKERS', '12'))
    if workers > 1 and convs:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(conventional, convs), 1):
                rows.append(r)
                if i % 25 == 0 or i == len(convs):
                    print(f'  conventional {i}/{len(convs)}', flush=True)
    else:
        for i, name in enumerate(convs, 1):
            rows.append(conventional(name))
            print(f'  conventional {i}/{len(convs)}', flush=True)
    for i, rd in enumerate(pols, 1):
        rows.append(policy(rd))
        if i % 10 == 0 or i == len(pols):
            print(f'  policy {i}/{len(pols)}', flush=True)

    json.dump(rows, open(out_path, 'w'), indent=1)
    print(f'wrote {out_path}')
    report(rows)


def report(rows: list) -> None:
    """The attribution Section 6.1 reports."""
    import statistics as st

    conv = [r for r in rows if r['kind'] == 'dual']
    pol = [r for r in rows if r['kind'] == 'policy']
    if not conv or not pol:
        return

    def pooled(r):
        ps = r['per_scenario']
        return (sum(ps[s]['ch4'] * ps[s]['steps'] for s in SC)
                / sum(ps[s]['steps'] for s in SC))

    best = max(conv, key=pooled)
    pm = st.median([pooled(p) for p in pol])
    print(f'\nbest conventional tuning {best["name"]}: {pooled(best):.1f} m3/d')
    print(f'policy pooled median {pm:.1f}  ->  {pm - pooled(best):+.1f} m3/d\n')

    for label, ref in (('best tuning', lambda s: best['per_scenario'][s]['ch4']),
                       ('median tuning',
                        lambda s: st.median([r['per_scenario'][s]['ch4']
                                             for r in conv]))):
        d = {s: st.median([r['per_scenario'][s]['ch4'] for r in pol]) - ref(s)
             for s in SC}
        tot = sum(d.values())
        print(f'against the {label}:')
        for s in SC:
            flag = ' unreachable' if s in UNREACHABLE else ''
            print(f'  {s:16s} {d[s]:+8.1f} m3/d  {100 * d[s] / tot:+6.1f}%{flag}')
        u = sum(d[s] for s in UNREACHABLE)
        print(f'  unreachable pair: {100 * u / tot:+.1f}% of the total '
              f'({len(UNREACHABLE)}/{len(SC)} = '
              f'{100 * len(UNREACHABLE) / len(SC):.0f}% by count)\n')

    print('policies above the best tuning, by scenario:')
    for s in SC:
        c = best['per_scenario'][s]['ch4']
        n = sum(1 for p in pol if p['per_scenario'][s]['ch4'] > c)
        flag = ' unreachable' if s in UNREACHABLE else ''
        print(f'  {s:16s} {n:2d}/{len(pol)}{flag}')


if __name__ == '__main__':
    main()

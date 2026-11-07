#!/usr/bin/env python3
"""Wastewater-treatment metrics for the evaluated controllers.

The benchmark scores a controller by methane flow at a matched excursion
rate.  That is a production axis, and a digester is also a treatment unit:
the quantity an operator is held to is how much organic load leaves the
reactor, not only how much gas it makes.  The two need not agree, because
both are driven by how hard the reactor is fed.

This re-evaluates trained policies and records, per run:

  cod_removal   100 * (1 - COD out / COD in), flow-weighted over the episode
  cod_in/out    the flow-weighted influent and effluent COD, kg COD/m3
  yield_ch4     m3 CH4 per kg COD removed, the conversion efficiency
  olr           organic loading rate, kg COD per m3 of reactor per day
  hrt           hydraulic retention time, d
  q_ad          mean commanded feed rate, m3/d

The per-run archive keeps summary statistics only, so these cannot be
recovered from it; the policies are re-run instead.  Training is untouched
and the environment is the one the paper evaluates on, so the methane and
violation figures a run reports here reproduce its archived record.

    python3 scripts/treatment_metrics.py --conventional      # PI / constant feed
    python3 scripts/treatment_metrics.py --models <glob>     # trained policies
    python3 scripts/treatment_metrics.py --sweep             # feed-rate sweep
    python3 scripts/treatment_metrics.py --manifest <json>   # the paper's set

The two policy families were trained under different libraries and are read
back the same way here: the reward-penalty runs are Stable-Baselines3
archives, the constrained runs are OmniSafe checkpoint directories.  Neither
library imports in the other's environment, so --manifest selects the entries
it can load and reports the rest as pending, and the two passes are run from
their own interpreters into one output directory.

Writes one JSON per run into treatment/ and prints a summary.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from env.adm1_gym_env_std import ADM1Env_Std as E
from env.normalized_wrapper import NormalizedADM1Env
from env.history_wrapper import HistoryWrapper

SC = ['low_load', 'nominal', 'plant_load', 'high_load', 'peak_load',
      'fog_surge', 'elevated_start']

# The COD-bearing ADM1 species.  S_IC and S_IN are inorganic carbon and
# nitrogen and carry no COD; S_h2 and S_ch4 are dissolved gas and do, so they
# leave with the liquid and count against removal.  The ion states are the
# dissociated forms already counted in their parent acid.
COD_SPECIES = ('S_su', 'S_aa', 'S_fa', 'S_va', 'S_bu', 'S_pro', 'S_ac',
               'S_h2', 'S_ch4', 'S_I',
               'X_xc', 'X_ch', 'X_pr', 'X_li', 'X_su', 'X_aa', 'X_fa',
               'X_c4', 'X_pro', 'X_ac', 'X_h2', 'X_I')

VFA_SOFT = 0.320
MG = 1000 / 1.0667


def cod_of(d):
    return sum(float(d.get(k, 0.0)) for k in COD_SPECIES)


def run_episode(env, base, act_fn, steps=60):
    """One scenario.  Returns the flow-weighted COD balance and the rest."""
    qin = cod_in = cod_out = ch4 = 0.0
    vfa, n = [], 0
    for k in range(steps):
        inf = base._build_influent_dict(k)
        obs_act = act_fn(k)
        _, _, term, trunc, _ = env.step(obs_act)
        s = base.solver.state
        q = float(base.solver.q_ad)
        # Influent and effluent leave at the same flow in this reactor, so the
        # balance is over the load each step carries rather than the
        # concentration, which would weight a starved step like a full one.
        qin += q
        cod_in += q * cod_of(inf)
        cod_out += q * cod_of(s)
        ch4 += float(base.solver.q_ch4)
        vfa.append(sum(float(s[x]) for x in ('S_va', 'S_bu', 'S_pro', 'S_ac')))
        n += 1
        if term or trunc:
            break
    return dict(qin=qin, cod_in=cod_in, cod_out=cod_out, ch4=ch4,
                vfa=vfa, n=n)


def summarise(parts, v_liq):
    qin = sum(p['qin'] for p in parts)
    cod_in = sum(p['cod_in'] for p in parts)
    cod_out = sum(p['cod_out'] for p in parts)
    ch4 = sum(p['ch4'] for p in parts)
    n = sum(p['n'] for p in parts)
    vfa = np.array([x for p in parts for x in p['vfa']])
    removed = cod_in - cod_out
    return {
        'steps': n,
        'ch4': ch4 / n,
        'viol': float(100 * np.mean(vfa > VFA_SOFT)),
        'vfa_med': float(np.median(vfa) * MG),
        'q_ad': qin / n,
        'cod_in': cod_in / qin,
        'cod_out': cod_out / qin,
        'cod_removal': 100.0 * removed / cod_in,
        # m3 CH4 per kg COD removed: the conversion efficiency, which a rate
        # of production alone does not show.
        'yield_ch4': ch4 / removed if removed > 0 else float('nan'),
        # Load applied per unit reactor volume per day, the axis the
        # scenarios are laid out on.
        'olr': cod_in / n / v_liq,
        'hrt': v_liq / (qin / n),
    }


def eval_policy(model_path):
    import warnings
    warnings.filterwarnings('ignore')
    import torch
    torch.set_num_threads(1)
    import stable_baselines3 as sb3
    import sb3_contrib as sbc

    meta = json.load(open(os.path.join(os.path.dirname(model_path),
                                       'run_meta.json')))
    algo = meta.get('algo', 'SAC').lower()
    MAP = {'sac': sb3.SAC, 'ppo': sb3.PPO, 'a2c': sb3.A2C, 'ddpg': sb3.DDPG,
           'td3': sb3.TD3, 'tqc': sbc.TQC, 'trpo': sbc.TRPO,
           'recurrentppo': sbc.RecurrentPPO, 'crossq': sbc.CrossQ,
           'ars': sbc.ARS}
    model = MAP[algo].load(model_path, device='cpu')
    hist = 5 if '_h5_' in model_path else 1

    parts, v_liq = [], None
    for s in SC:
        env = NormalizedADM1Env(E(s, obs_mode='scada', step_size=1.0))
        if hist > 1:
            env = HistoryWrapper(env, hist)
        obs, _ = env.reset(seed=7)
        base = env.env.env if hist > 1 else env.env
        v_liq = base.V_liq
        state = {'h': None, 'obs': obs}

        def act(_k):
            if algo == 'recurrentppo':
                a, state['h'] = model.predict(state['obs'], state=state['h'],
                                              deterministic=True)
            else:
                a, _ = model.predict(state['obs'], deterministic=True)
            return a

        # env.step returns the next observation, which predict needs, so the
        # episode is driven here rather than through run_episode's callback.
        qin = cod_in = cod_out = ch4 = 0.0
        vfa, n = [], 0
        for k in range(60):
            inf = base._build_influent_dict(k)
            a = act(k)
            state['obs'], _, term, trunc, _ = env.step(a)
            sta = base.solver.state
            q = float(base.solver.q_ad)
            qin += q
            cod_in += q * cod_of(inf)
            cod_out += q * cod_of(sta)
            ch4 += float(base.solver.q_ch4)
            vfa.append(sum(float(sta[x]) for x in
                           ('S_va', 'S_bu', 'S_pro', 'S_ac')))
            n += 1
            if term or trunc:
                break
        parts.append(dict(qin=qin, cod_in=cod_in, cod_out=cod_out,
                          ch4=ch4, vfa=vfa, n=n))

    out = summarise(parts, v_liq)
    out.update(model=model_path, algo=algo, seed=meta.get('seed'),
               w=meta.get('reward_config'),
               total_timesteps=meta.get('total_timesteps'))
    return out


def eval_cmdp(run_dir):
    """A constrained policy, replayed as eval_cmdp.py loads it."""
    import warnings
    warnings.filterwarnings('ignore')
    import torch
    torch.set_num_threads(1)
    from omnisafe.common.normalizer import Normalizer
    from omnisafe.models.actor import ActorBuilder

    from training.omnisafe_env import CMDP_REWARD

    run_dir = run_dir.rstrip('/')
    cfg = json.load(open(os.path.join(run_dir, 'config.json')))
    ck = sorted(glob.glob(os.path.join(run_dir, 'torch_save', 'epoch-*.pt')),
                key=lambda q: int(q.split('epoch-')[1].split('.pt')[0]))
    params = torch.load(ck[-1], map_location='cpu')

    # The actor is rebuilt from the run's own config, so a family that was
    # swept at a different width or activation is restored at its own.
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

    parts, v_liq = [], None
    for sc in SC:
        # The constrained runs were trained against CMDP_REWARD, so the
        # episode is replayed under the same reward the policy saw.
        env = NormalizedADM1Env(E(sc, reward_config=CMDP_REWARD,
                                  obs_mode='scada', step_size=1.0))
        obs, _ = env.reset(seed=7)
        base = env.env
        v_liq = base.V_liq
        state = {'obs': obs}

        def act(_k):
            x = torch.as_tensor(state['obs'], dtype=torch.float32)
            if norm is not None:
                x = norm.normalize(x)
            with torch.no_grad():
                return actor.predict(x, deterministic=True).numpy()

        qin = cod_in = cod_out = ch4 = 0.0
        vfa, n = [], 0
        for k in range(60):
            inf = base._build_influent_dict(k)
            a = act(k)
            state['obs'], _, term, trunc, _ = env.step(a)
            sta = base.solver.state
            q = float(base.solver.q_ad)
            qin += q
            cod_in += q * cod_of(inf)
            cod_out += q * cod_of(sta)
            ch4 += float(base.solver.q_ch4)
            vfa.append(sum(float(sta[x]) for x in
                           ('S_va', 'S_bu', 'S_pro', 'S_ac')))
            n += 1
            if term or trunc:
                break
        parts.append(dict(qin=qin, cod_in=cod_in, cod_out=cod_out,
                          ch4=ch4, vfa=vfa, n=n))

    out = summarise(parts, v_liq)
    out.update(model=run_dir, algo=cfg['algo'],
               seed=cfg.get('seed'),
               cost_limit=cfg['algo_cfgs'].get('cost_limit'))
    return out


def eval_constant(q, mult):
    """A constant-feed controller, as a reference the policies are read against."""
    parts, v_liq = [], None
    for s in SC:
        env = E(s, obs_mode='scada', step_size=1.0)
        env.reset(seed=7)
        v_liq = env.V_liq
        a = np.array([q, mult], dtype=np.float32)
        parts.append(run_episode(env, env, lambda _k: a))
    out = summarise(parts, v_liq)
    out.update(kind='constant', q=q, mult=mult)
    return out


def _record_name(entry):
    """A stable output name for a run, independent of where it was loaded."""
    return (entry['path'].rstrip('/').replace(ROOT + '/', '')
            .replace('/', '__').replace('.zip', ''))


def run_manifest(path, out, limit=0):
    """Re-evaluate the manifest's runs that this interpreter can load.

    The two policy libraries do not coexist in one environment, so each pass
    takes the entries it can import for and leaves the others; running both
    passes into the same output directory completes the set.
    """
    entries = json.load(open(path))
    if limit:
        entries = entries[:limit]
    have_sb3 = have_omni = True
    try:
        import stable_baselines3                    # noqa: F401
        import sb3_contrib                          # noqa: F401
    except ImportError:
        have_sb3 = False
    try:
        import omnisafe                             # noqa: F401
    except ImportError:
        have_omni = False
    print(f'{len(entries)} runs in manifest; this interpreter can read '
          f'{"SB3" if have_sb3 else ""}{" and " if have_sb3 and have_omni else ""}'
          f'{"OmniSafe" if have_omni else ""}' or 'neither library')

    done = skipped = pending = failed = 0
    for i, e in enumerate(entries, 1):
        dest = os.path.join(out, _record_name(e) + '.json')
        if os.path.exists(dest):
            skipped += 1
            continue
        kind = e.get('set')
        if kind == 'cmdp' and not have_omni:
            pending += 1
            continue
        if kind != 'cmdp' and not have_sb3:
            pending += 1
            continue
        try:
            r = eval_cmdp(e['path']) if kind == 'cmdp' else eval_policy(e['path'])
        except Exception as exc:                    # noqa: BLE001
            print(f'  FAIL {e["path"]}: {type(exc).__name__}: {exc}')
            failed += 1
            continue
        # The archived figures come from the same environment, so carrying
        # them here lets the reproduction be checked rather than assumed.
        r.update(family=e.get('family'), cls=e.get('cls'),
                 ch4_archived=e.get('ch4'), viol_archived=e.get('viol'))
        json.dump(r, open(dest, 'w'))
        done += 1
        if done % 20 == 0:
            print(f'  {done} written ({i}/{len(entries)} seen)')
    print(f'done {done}  already had {skipped}  other library {pending}  '
          f'failed {failed}  -> {out}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', help='glob of model .zip files')
    ap.add_argument('--sweep', action='store_true',
                    help='constant-feed sweep over the actuator range')
    ap.add_argument('--manifest',
                    help='JSON list of the runs the paper reports')
    ap.add_argument('--out', default=os.path.join(ROOT, 'treatment'))
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.sweep:
        print(f'{"q":>5} {"mult":>5} {"CODin":>7} {"CODout":>7} {"removal":>8} '
              f'{"CH4":>8} {"yield":>7} {"OLR":>6} {"HRT":>6} {"viol":>6}')
        rows = []
        for mult in (0.7, 1.0, 1.3):
            for q in (41, 70, 100, 130, 159):
                r = eval_constant(float(q), float(mult))
                rows.append(r)
                print(f'{q:5d} {mult:5.1f} {r["cod_in"]:7.2f} {r["cod_out"]:7.2f} '
                      f'{r["cod_removal"]:7.1f}% {r["ch4"]:8.1f} '
                      f'{r["yield_ch4"]:7.3f} {r["olr"]:6.2f} {r["hrt"]:6.1f} '
                      f'{r["viol"]:5.1f}%')
        json.dump(rows, open(os.path.join(a.out, 'sweep_constant.json'), 'w'),
                  indent=1)
        print(f'-> {os.path.join(a.out, "sweep_constant.json")}')
        return

    if a.manifest:
        run_manifest(a.manifest, a.out, a.limit)
        return

    if not a.models:
        ap.error('pass --models, --manifest or --sweep')
    paths = sorted(glob.glob(a.models))
    if a.limit:
        paths = paths[:a.limit]
    print(f'{len(paths)} models')
    done = 0
    for p in paths:
        name = p.replace(ROOT + '/', '').replace('/', '__').replace('.zip', '')
        dest = os.path.join(a.out, name + '.json')
        if os.path.exists(dest):
            done += 1
            continue
        try:
            r = eval_policy(p)
        except Exception as e:                      # noqa: BLE001
            print(f'  skip {p}: {type(e).__name__}: {e}')
            continue
        json.dump(r, open(dest, 'w'))
        done += 1
        if done % 20 == 0:
            print(f'  {done}/{len(paths)}')
    print(f'{done} records -> {a.out}')


if __name__ == '__main__':
    main()

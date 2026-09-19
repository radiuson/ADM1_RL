#!/usr/bin/env python3
"""Figure 4: where each controller sits inside the constraint margin.

Two controllers matched on violation rate are compared by their VFA
distribution.  The mechanism by which a learned policy converts equal risk into
higher production is a tighter distribution placed closer to the limit, which
is visible here as mass in the band just below the constraint rather than a
long tail.
"""
import os, sys, json, glob
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
import numpy as np, torch, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
torch.set_num_threads(1)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from env.adm1_gym_env_std import ADM1Env_Std as E
from env.normalized_wrapper import NormalizedADM1Env
from training.baselines import PIFeed
from omnisafe.models.actor import ActorBuilder
from omnisafe.common.normalizer import Normalizer
from training.omnisafe_env import CMDP_REWARD

SC = ['low_load', 'nominal', 'plant_load', 'high_load', 'peak_load',
      'fog_surge', 'elevated_start', 'acidified_recovery']
LIM = 0.320
MG = 1000 / 1.0667


def run_policy(run_dir):
    cfg = json.load(open(os.path.join(run_dir, 'config.json')))
    ck = sorted(glob.glob(os.path.join(run_dir, 'torch_save', 'epoch-*.pt')),
                key=lambda p: int(p.split('epoch-')[1].split('.pt')[0]))
    prm = torch.load(ck[-1], map_location='cpu')
    probe = NormalizedADM1Env(E('nominal', reward_config=CMDP_REWARD,
                                obs_mode='scada', step_size=1.0))
    mc = cfg['model_cfgs']
    ac = ActorBuilder(obs_space=probe.observation_space,
                      act_space=probe.action_space,
                      hidden_sizes=mc['actor']['hidden_sizes'],
                      activation=mc['actor']['activation'],
                      weight_initialization_mode=mc['weight_initialization_mode']
                      ).build_actor(mc['actor_type'])
    ac.load_state_dict(prm['pi']); ac.eval()
    nm = None
    if 'obs_normalizer' in prm:
        nm = Normalizer(shape=probe.observation_space.shape, clip=5)
        nm.load_state_dict(prm['obs_normalizer'])
    V = []
    for s in SC:
        env = NormalizedADM1Env(E(s, reward_config=CMDP_REWARD, obs_mode='scada',
                                  step_size=1.0))
        o, _ = env.reset(seed=7)
        for _ in range(200):
            x = torch.as_tensor(o, dtype=torch.float32)
            if nm is not None:
                x = nm.normalize(x)
            with torch.no_grad():
                a = ac.predict(x, deterministic=True).numpy()
            o, r, te, tr, _ = env.step(a)
            st = env.env.solver.state
            V.append(sum(st[k] for k in ('S_va', 'S_bu', 'S_pro', 'S_ac')))
            if te or tr:
                break
    return np.array(V)


def run_pi(sp, Kc, ti):
    C = PIFeed(sp, 41, 159, Kc=Kc, tau_i=ti, mult=1.3)
    V = []
    for s in SC:
        e = E(s, obs_mode='scada', step_size=1.0); e.reset(seed=7); C.reset()
        q, mult = 159.0, 1.3
        for _ in range(200):
            st = e.solver.state
            vfa = sum(st[k] for k in ('S_va', 'S_bu', 'S_pro', 'S_ac'))
            q, mult = C.act(vfa * MG, st['S_IC'],
                            -np.log10(max(st['S_H_ion'], 1e-14)),
                            e.solver.q_ch4, q, mult)
            o, r, te, tr, _ = e.step(np.array([q, mult], dtype=np.float32))
            st = e.solver.state
            V.append(sum(st[k] for k in ('S_va', 'S_bu', 'S_pro', 'S_ac')))
            if te or tr:
                break
    return np.array(V)


d = sorted(glob.glob('models_cmdp/TRPOPID-{ADM1-v0}/seed-042-*/'))
rl = run_policy(d[2])
pi = run_pi(225, 0.35, 8)
print(f'TRPOPID n={len(rl)} viol={100*(rl>LIM).mean():.1f}%  '
      f'PI n={len(pi)} viol={100*(pi>LIM).mean():.1f}%')

fig, ax = plt.subplots(figsize=(6.4, 4.0))
bins = np.linspace(0, 600, 61)
for v, lab, col, a in ((pi, 'PI, setpoint 225 mg/L', '#9a7a2f', 0.55),
                       (rl, 'TRPOPID, cost limit 3.0', '#2f6f5e', 0.55)):
    ax.hist(v * MG, bins=bins, density=True, color=col, alpha=a,
            label=f'{lab}   ({100*(v>LIM).mean():.1f} % above limit)')
ax.axvline(300, color='#9a3a30', lw=1.4, zorder=5)
ax.text(305, ax.get_ylim()[1]*0.93, 'soft limit\n300 mg/L', fontsize=7.6,
        color='#9a3a30', va='top')
ax.axvspan(210, 300, color='0.5', alpha=0.10, zorder=0)
ax.text(255, ax.get_ylim()[1]*0.55, 'margin\nband', fontsize=7.2, color='0.45',
        ha='center')
ax.set_xlabel('total VFA (mg/L as acetic acid)')
ax.set_ylabel('density')
ax.set_xlim(0, 600)
ax.grid(alpha=0.22, lw=0.6)
ax.legend(fontsize=7.6, loc='upper right', framealpha=0.94)
fig.tight_layout()
for e in ('pdf', 'png'):
    fig.savefig(f'figures/fig_mechanism.{e}', dpi=220)

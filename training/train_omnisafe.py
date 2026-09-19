#!/usr/bin/env python3
"""Train a constrained RL agent on the ADM1 digester.

The frontier is swept over ``--cost-limit`` rather than over a reward penalty
weight: the limit is the number of soft-VFA excursions allowed per 60-day
episode, so ``--cost-limit 3`` asks for a 5 % violation rate directly.

Run inside the ``adm1_safe`` conda environment, which pins the gymnasium 0.28
/ numpy 1.26 stack OmniSafe requires:

    /home/ihpc/anaconda3/envs/adm1_safe/bin/python training/train_omnisafe.py \
        --algo CPO --cost-limit 3.0 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import omnisafe

import training.omnisafe_env  # noqa: F401  (registers ADM1-v0)

# 60 one-day control steps per episode, so a cost limit of 3.0 is a 5 % rate.
EPISODE_STEPS = 60


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--algo', default='CPO',
                   help='OmniSafe algorithm, e.g. CPO, CPPOPID, PPOLag, TRPOLag, SACLag')
    p.add_argument('--cost-limit', type=float, default=3.0,
                   help='allowed VFA excursions per episode (3.0 = 5%% of 60 steps)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--timesteps', type=int, default=150_000)
    p.add_argument('--steps-per-epoch', type=int, default=2000)
    p.add_argument('--output-dir', default='models_cmdp')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    rate = 100.0 * args.cost_limit / EPISODE_STEPS
    tag = f'{args.algo}_c{args.cost_limit:g}_seed{args.seed}'
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfgs = {
        'seed': args.seed,
        'train_cfgs': {
            'total_steps': args.timesteps,
            'device': args.device,
            'vector_env_nums': 1,
            'torch_threads': 1,
        },
        'algo_cfgs': {
            'steps_per_epoch': args.steps_per_epoch,
        },
        'lagrange_cfgs': {
            'cost_limit': args.cost_limit,
        },
        'logger_cfgs': {
            'log_dir': str(out),
            'use_tensorboard': True,
            'use_wandb': False,
        },
    }

    # CPO and the other projection methods take the limit in algo_cfgs; the
    # Lagrangian family reads lagrange_cfgs.  Passing an unknown key is an
    # error in OmniSafe, so send only the one the algorithm declares.
    from omnisafe.algorithms import ALGORITHMS
    from omnisafe.utils.config import get_default_kwargs_yaml
    algo_type = next(k for k, v in ALGORITHMS.items()
                     if k != 'all' and args.algo in v)
    default = get_default_kwargs_yaml(args.algo, 'ADM1-v0', algo_type)
    if 'lagrange_cfgs' in default:
        pass                                   # Lagrangian family: keep it there
    elif 'cost_limit' in default.get('algo_cfgs', {}):
        cfgs.pop('lagrange_cfgs')
        cfgs['algo_cfgs']['cost_limit'] = args.cost_limit
    else:
        # Unconstrained algorithm (e.g. NaturalPG, PolicyGradient): it has no
        # constraint knob at all, so run it as the unconstrained reference.
        cfgs.pop('lagrange_cfgs')

    n_epochs = args.timesteps // args.steps_per_epoch
    if algo_type == 'off-policy':
        # OmniSafe's off-policy constrained defaults are not on the same scale
        # as their on-policy counterparts: lambda_lr is 1e-5 against 0.035, the
        # PID gains are 1e-6/1e-7 against 0.1/0.01, and warmup_epochs is 100
        # where the on-policy configs have no warmup at all.  Left as shipped,
        # the multiplier barely moves within this study's 75-epoch budget and
        # the cost limit has almost no effect on the learned policy, so the
        # comparison would report the defaults rather than the algorithms.  The
        # dual-update settings are matched to the on-policy values and the
        # warmup is scaled to the budget.
        if 'warmup_epochs' in default.get('algo_cfgs', {}):
            cfgs['algo_cfgs']['warmup_epochs'] = max(1, n_epochs // 15)
        lag = default.get('lagrange_cfgs', {})
        matched = {}
        if 'lambda_lr' in lag:
            matched['lambda_lr'] = 0.035
        for k, v in (('pid_kp', 0.1), ('pid_ki', 0.01), ('pid_kd', 0.01)):
            if k in lag:
                matched[k] = v
        if matched:
            cfgs.setdefault('lagrange_cfgs', {}).update(matched)
        # The off-policy loop saves only on (epoch+1) % save_model_freq == 0 and
        # has no final-epoch clause, so the default 100 leaves epoch-0 as the
        # only checkpoint.
        freq = next((f for f in (25, 15, 5, 1) if n_epochs % f == 0), 1)
        cfgs['logger_cfgs']['save_model_freq'] = freq
        print(f'[{tag}] off-policy: warmup -> {cfgs["algo_cfgs"].get("warmup_epochs")}, '
              f'dual {matched}, save_model_freq -> {freq}', flush=True)

    print(f'[{tag}] cost_limit {args.cost_limit:g} = {rate:.1f}% 目标越限率, '
          f'{args.timesteps} 步', flush=True)

    t0 = time.time()
    agent = omnisafe.Agent(args.algo, 'ADM1-v0', custom_cfgs=cfgs)
    agent.learn()
    elapsed = (time.time() - t0) / 60

    meta = {
        'algo': args.algo,
        'formulation': 'CMDP',
        'cost_limit': args.cost_limit,
        'target_violation_pct': rate,
        'episode_steps': EPISODE_STEPS,
        'seed': args.seed,
        'total_timesteps': args.timesteps,
        'elapsed_min': round(elapsed, 1),
        'log_dir': str(agent.agent.logger.log_dir) if hasattr(agent, 'agent') else str(out),
    }
    with open(out / f'{tag}_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'[{tag}] 完成, {elapsed:.1f} 分钟', flush=True)


if __name__ == '__main__':
    main()

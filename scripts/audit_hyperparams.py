#!/usr/bin/env python3
"""Compare every algorithm's effective hyperparameters against its library default.

Four separate configuration faults reached results in this study before being
caught one at a time; this enumerates the whole surface instead.  A divergence
is not necessarily wrong -- matching the on-policy dual gains was deliberate --
but every one should be a decision on record rather than an accident.
"""
import inspect, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WATCH = ['learning_rate', 'n_steps', 'batch_size', 'gamma', 'gae_lambda',
         'buffer_size', 'learning_starts', 'train_freq', 'gradient_steps',
         'tau', 'ent_coef', 'action_noise', 'target_kl', 'clip_range',
         'n_epochs', 'policy_delay', 'target_policy_noise']

def sb3_defaults(cls):
    return {k: v.default for k, v in inspect.signature(cls.__init__).parameters.items()
            if k in WATCH and v.default is not inspect.Parameter.empty}

def main():
    import stable_baselines3 as sb3, sb3_contrib as sbc
    from training.train_sac_std_cur import (SAC_HYPERPARAMS, PPO_HYPERPARAMS,
                                            TRPO_HYPERPARAMS)
    CLS = {'sac': sb3.SAC, 'ppo': sb3.PPO, 'a2c': sb3.A2C, 'ddpg': sb3.DDPG,
           'td3': sb3.TD3, 'tqc': sbc.TQC, 'trpo': sbc.TRPO,
           'recurrentppo': sbc.RecurrentPPO, 'crossq': sbc.CrossQ, 'ars': sbc.ARS}
    # mirror the dispatch in train_sac_std_cur.py
    USED = {
        'sac': SAC_HYPERPARAMS, 'tqc': SAC_HYPERPARAMS,
        'ppo': PPO_HYPERPARAMS, 'recurrentppo': PPO_HYPERPARAMS,
        'trpo': TRPO_HYPERPARAMS,
        'a2c': {**{k: v for k, v in PPO_HYPERPARAMS.items()
                   if k not in ('n_epochs', 'clip_range', 'batch_size')},
                'n_steps': 32},
        'crossq': {k: v for k, v in SAC_HYPERPARAMS.items() if k != 'tau'},
        'ddpg': {k: v for k, v in SAC_HYPERPARAMS.items() if k != 'ent_coef'},
        'td3': {k: v for k, v in SAC_HYPERPARAMS.items() if k != 'ent_coef'},
        'ars': {'n_delta': 8, 'n_top': 4, 'learning_rate': 0.02, 'delta_std': 0.05},
    }
    print(f"{'algorithm':<14}{'parameter':<20}{'library default':>20}{'used':>20}")
    print('-' * 74)
    flagged = 0
    for a in sorted(USED):
        dflt = sb3_defaults(CLS[a])
        used = USED[a]
        for k in WATCH:
            if k not in dflt:
                continue
            u = used.get(k, dflt[k])
            if repr(u) != repr(dflt[k]):
                flagged += 1
                print(f"{a:<14}{k:<20}{repr(dflt[k]):>20}{repr(u):>20}")
    print(f"\n{flagged} parameters differ from the library default.")
    print("Deterministic-policy methods (DDPG, TD3) explore only through "
          "action_noise; a None there means no exploration after warm-up.")

if __name__ == '__main__':
    main()

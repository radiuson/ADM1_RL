#!/usr/bin/env python3
"""Read the effective hyperparameters back from the trained models.

run_meta.json records only the values passed explicitly, so a library default
-- TRPO's line search, TD3's target-policy noise, TQC's dropped quantiles --
is absent from it even though it governed the run.  Loading one model per
family and reading the attributes off the object gives the values that were
actually in force.

    python3 scripts/probe_hyper.py > table_hyper_probed.tex
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

# Attributes worth reporting, in the order they appear in the appendix.
KEYS = ['learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma',
        'gae_lambda', 'clip_range', 'ent_coef', 'vf_coef', 'max_grad_norm',
        'normalize_advantage', 'target_kl', 'cg_max_steps', 'cg_damping',
        'line_search_shrinking_factor', 'line_search_max_iter',
        'sub_sampling_factor', 'tau', 'buffer_size', 'learning_starts',
        'gradient_steps', 'target_policy_noise', 'target_noise_clip',
        'policy_delay', 'top_quantiles_to_drop_per_net', 'n_delta', 'n_top']

LABEL = {'a2c': 'A2C', 'ars': 'ARS', 'crossq': 'CrossQ', 'ddpg': 'DDPG',
         'ppo': 'PPO', 'recurrentppo': 'RecurrentPPO', 'sac': 'SAC',
         'td3': 'TD3', 'tqc': 'TQC', 'trpo': 'TRPO'}


def probe(algo, path):
    import stable_baselines3 as sb3
    import sb3_contrib as sbc
    MAP = {'sac': sb3.SAC, 'ppo': sb3.PPO, 'a2c': sb3.A2C, 'ddpg': sb3.DDPG,
           'td3': sb3.TD3, 'tqc': sbc.TQC, 'trpo': sbc.TRPO,
           'recurrentppo': sbc.RecurrentPPO, 'crossq': sbc.CrossQ,
           'ars': sbc.ARS}
    m = MAP[algo].load(path, device='cpu')
    out = {}
    for k in KEYS:
        if not hasattr(m, k):
            continue
        v = getattr(m, k)
        if callable(v):
            try:
                v = v(1.0)
            except Exception:
                continue
        if v is None:
            continue
        if not isinstance(v, (int, float, str, bool)):
            v = str(v)
        out[k] = v
    noise = getattr(m, 'action_noise', None)
    if noise is not None:
        out['action_noise'] = str(noise)
    return out


def main():
    from plot_figures import load_reward
    R = load_reward()
    res = {}
    for a in sorted(R):
        if not R[a]:
            continue
        res[a] = probe(a, R[a][0]['model'])
    json.dump(res, open(os.path.join(os.path.dirname(HERE),
                                     'hyper_probed.json'), 'w'), indent=1)
    print(f'probed {len(res)} families -> hyper_probed.json', file=sys.stderr)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
SAC Training on Standard ADM1 — Multi-Scenario Curriculum
===========================================================

Trains SAC on ADM1Env_Std with uniform-random scenario sampling over all
six paper scenarios.  This is the standard training recipe for the paper's
SF-SAC, ST-SAC, and Balanced-RL agents.

Scenarios (paper group):
    nominal, high_load, low_load, shock_load, high_load_real, pre_stressed

Usage:
    # SF-SAC  (Safety-First, 3 seeds)
    python training/train_sac_std_cur.py --reward-config safety_first   --seed 42
    python training/train_sac_std_cur.py --reward-config safety_first   --seed 123
    python training/train_sac_std_cur.py --reward-config safety_first   --seed 456

    # ST-SAC  (Safety-Target)
    python training/train_sac_std_cur.py --reward-config safety_target  --seed 42
    ...

    # Balanced-RL (no safety reward, weak quadratic penalty)
    python training/train_sac_std_cur.py --reward-config balanced       --seed 42
    ...

Output layout:
    models_std_paper/
      sac_std_cur_<reward>_seed<seed>/
        best_model/best_model.zip
        final_model.zip
        run_meta.json
"""

import argparse
import os
import json
import random
import time
from pathlib import Path

# One thread per process. These runs are launched many at a time, and the
# linear-algebra backends otherwise each open a pool sized to the whole
# machine: eleven concurrent runs then contend for hundreds of threads on
# twenty-four cores and every one of them slows down. Set before torch and
# numpy are imported, which is when the pools are sized.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
import torch

torch.set_num_threads(1)

import gymnasium as gym
from gymnasium import spaces

from env.adm1_gym_env_std import ADM1Env_Std, STD_SCENARIOS
from env.normalized_wrapper import NormalizedADM1Env
from env.history_wrapper import HistoryWrapper
from training.reward_configs import REWARD_CONFIGS

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback
)


# ── Paper scenario set ────────────────────────────────────────────────────────

PAPER_SCENARIOS = ['low_load', 'nominal', 'plant_load', 'high_load', 'peak_load',
                   'fog_surge', 'elevated_start', 'acidified_recovery']

# ── Hyperparameters ───────────────────────────────────────────────────────────

SAC_HYPERPARAMS = {
    'learning_rate':   3e-4,
    'buffer_size':     1_000_000,
    'batch_size':      256,
    'tau':             0.005,
    'gamma':           0.99,
    'learning_starts': 10_000,
    'train_freq':      1,
    'gradient_steps':  1,
    'ent_coef':        'auto',
    'policy_kwargs':   dict(net_arch=[256, 256]),
}

# On-policy hyperparameters.  PPO-family methods learn from short on-policy
# rollouts rather than a replay buffer, so they need their own settings.
# n_steps=2048 spans several 200-day episodes' worth of control decisions.
# Reward normalisation (VecNormalize below) is applied for these algorithms
# only: the PPO literature reports it as important for stable advantage
# estimates, while SAC/TQC scale their own critic targets.
PPO_HYPERPARAMS = {
    'learning_rate': 3e-4,
    'n_steps':       2048,
    'batch_size':    256,
    'n_epochs':      10,
    'gamma':         0.99,
    'gae_lambda':    0.95,
    'clip_range':    0.2,
    'ent_coef':      0.0,
    'policy_kwargs': dict(net_arch=[256, 256]),
}

# TRPO takes a trust-region step instead of a clipped surrogate, so it has no
# clip_range, n_epochs or entropy bonus.
TRPO_HYPERPARAMS = {
    'learning_rate': 3e-4,
    'n_steps':       2048,
    'batch_size':    256,
    'gamma':         0.99,
    'gae_lambda':    0.95,
    'policy_kwargs': dict(net_arch=[256, 256]),
}

ON_POLICY_ALGOS = ('ppo', 'recurrentppo', 'trpo')

ENT_COEF_MIN      = 0.01
ENT_COEF_MAX      = 0.5    # reduced from 5.0 — allows policy to converge
DEFAULT_TIMESTEPS = 1_000_000


# ── Multi-scenario training wrapper ──────────────────────────────────────────

class MultiScenarioEnv(gym.Env):
    """
    Gymnasium wrapper that samples a new scenario uniformly at random on
    every episode reset.  The action and observation spaces match
    ADM1Env_Std (2-dim action, 12-dim or 5-dim obs).
    """

    def __init__(self, scenarios, reward_config, obs_mode='scada', seed=None, normalize=False,
                 step_size=0.01041667, vfa_soft=None, vfa_hard=None,
                 history=1, algo='sac'):
        super().__init__()
        self.scenarios     = scenarios
        self.reward_config = reward_config
        self.obs_mode      = obs_mode
        self.normalize     = normalize
        self.step_size     = step_size
        self.vfa_soft      = vfa_soft
        self.vfa_hard      = vfa_hard
        self.history       = int(history)
        self._rng          = np.random.default_rng(seed)

        # Build a reference env to define spaces
        _ref = ADM1Env_Std(scenarios[0], reward_config=reward_config, obs_mode=obs_mode,
                           step_size=step_size, vfa_soft=vfa_soft, vfa_hard=vfa_hard)
        if normalize:
            _ref = NormalizedADM1Env(_ref)
        if self.history > 1:
            _ref = HistoryWrapper(_ref, self.history)
        self.observation_space = _ref.observation_space
        self.action_space      = _ref.action_space
        _ref.close()

        self._env = None

    def _new_env(self):
        if self._env is not None:
            self._env.close()
        scenario = self._rng.choice(self.scenarios)
        self._env = ADM1Env_Std(
            scenario_name=scenario,
            reward_config=self.reward_config,
            obs_mode=self.obs_mode,
            step_size=self.step_size,
            vfa_soft=self.vfa_soft,
            vfa_hard=self.vfa_hard,
        )
        if self.normalize:
            self._env = NormalizedADM1Env(self._env)
        if self.history > 1:
            self._env = HistoryWrapper(self._env, self.history)

    def reset(self, seed=None, options=None):
        self._new_env()
        obs, info = self._env.reset(seed=seed)
        return obs, info

    def step(self, action):
        return self._env.step(action)

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None


# ── Entropy clamp callback ────────────────────────────────────────────────────

def _make_ent_clamp_callback(max_ent=ENT_COEF_MAX, min_ent=ENT_COEF_MIN):
    _max_log = torch.log(torch.tensor(max_ent, dtype=torch.float32)).item()
    _min_log = torch.log(torch.tensor(min_ent, dtype=torch.float32)).item()

    class _Clamp(BaseCallback):
        def _on_step(self):
            if hasattr(self.model, 'log_ent_coef') and self.model.log_ent_coef is not None:
                with torch.no_grad():
                    self.model.log_ent_coef.data.clamp_(_min_log, _max_log)
            return True

    return _Clamp(verbose=0)


# ── Main training function ────────────────────────────────────────────────────

def train(
    reward_config_name: str = 'safety_first',
    normalize: bool         = False,
    step_size: float        = 0.01041667,
    seed: int               = 42,
    obs_mode: str           = 'scada',
    total_timesteps: int    = DEFAULT_TIMESTEPS,
    output_dir: str         = 'models_std_paper',
    eval_scenario: str      = 'nominal',
    eval_freq: int          = 10_000,
    n_eval_episodes: int    = 5,
    device: str             = 'auto',
    verbose: int            = 1,
    vfa_soft: float         = None,
    vfa_hard: float         = None,
    history: int            = 1,
    algo: str               = 'sac',
) -> Path:
    """
    Train SAC on multi-scenario ADM1Env_Std curriculum.

    Returns:
        Path to the run directory.
    """
    if reward_config_name not in REWARD_CONFIGS:
        raise ValueError(f"Unknown reward config '{reward_config_name}'.")

    obs_tag  = f'_{obs_mode}' if obs_mode != 'scada' else ''
    obs_tag += f'_h{history}' if history > 1 else ''
    obs_tag += f'_{algo}' if algo != 'sac' else ''
    run_name = f'sac_std_cur_{reward_config_name}_seed{seed}{obs_tag}' + ('_norm' if normalize else '') + ('_daily' if step_size >= 0.99 else '')
    run_dir  = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'best_model').mkdir(exist_ok=True)
    (run_dir / 'checkpoints').mkdir(exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  SAC — Standard ADM1, Multi-Scenario Curriculum")
    print(f"  Reward:     {reward_config_name}")
    print(f"  Obs mode:   {obs_mode}")
    print(f"  Scenarios:  {PAPER_SCENARIOS}")
    print(f"  Seed:       {seed}")
    print(f"  Timesteps:  {total_timesteps:,}")
    print(f"  Output:     {run_dir}")
    print(f"{'='*65}")

    reward_config = REWARD_CONFIGS[reward_config_name]

    train_env = Monitor(MultiScenarioEnv(
        PAPER_SCENARIOS, reward_config=reward_config,
        obs_mode=obs_mode, seed=seed, normalize=normalize,
        step_size=step_size, vfa_soft=vfa_soft, vfa_hard=vfa_hard,
        history=history,
    ))

    _ev = ADM1Env_Std(
        scenario_name=eval_scenario,
        reward_config=reward_config,
        obs_mode=obs_mode,
        step_size=step_size,
        vfa_soft=vfa_soft,
        vfa_hard=vfa_hard,
    )
    _ee = NormalizedADM1Env(_ev) if normalize else _ev
    if history > 1:
        _ee = HistoryWrapper(_ee, history)
    eval_env = Monitor(_ee)
    eval_env.reset(seed=seed + 1)

    policy_name = 'MlpPolicy'
    if algo == 'ars':
        policy_name = 'LinearPolicy'
    if algo == 'sac':
        Algo, hp = SAC, SAC_HYPERPARAMS
    elif algo == 'tqc':
        from sb3_contrib import TQC as Algo
        hp = SAC_HYPERPARAMS
    elif algo == 'ppo':
        from stable_baselines3 import PPO as Algo
        hp = PPO_HYPERPARAMS
    elif algo == 'recurrentppo':
        from sb3_contrib import RecurrentPPO as Algo
        hp, policy_name = PPO_HYPERPARAMS, 'MlpLstmPolicy'
    elif algo == 'trpo':
        from sb3_contrib import TRPO as Algo
        hp = TRPO_HYPERPARAMS
    elif algo == 'a2c':
        # On-policy WITHOUT a trust region or clipped surrogate: the control
        # that separates "on-policy" from "trust region" as the explanation
        # for the on-policy methods' advantage here.
        from stable_baselines3 import A2C as Algo
        hp = {k: v for k, v in PPO_HYPERPARAMS.items()
              if k not in ('n_epochs', 'clip_range', 'batch_size')}
        # A2C updates far more often than PPO or TRPO at a shared n_steps, so
        # the rollout length is the axis on which the on-policy family is not
        # directly comparable.  ADM1_ALIGN_NSTEPS matches it to the others.
        hp['n_steps'] = 2048 if os.environ.get('ADM1_ALIGN_NSTEPS') else 32
    elif algo == 'crossq':
        # Recent off-policy method; tests whether the off-policy shortfall is
        # specific to SAC or general to the family.
        from sb3_contrib import CrossQ as Algo
        # CrossQ removes the target network and relies on batch normalisation
        # to learn early, so a long warm-up works against the mechanism it is
        # proposed for.  ADM1_NATIVE_HP restores its own short warm-up.
        hp = {k: v for k, v in SAC_HYPERPARAMS.items() if k != 'tau'}
        if os.environ.get('ADM1_NATIVE_HP'):
            hp['learning_starts'] = 100
    elif algo == 'ars':
        # Derivative-free direct policy search.  If it matches the gradient
        # methods, the problem does not need value-function learning at all.
        from sb3_contrib import ARS as Algo
        hp = {'n_delta': 8, 'n_top': 4, 'learning_rate': 0.02, 'delta_std': 0.05}
    elif algo in ('ddpg', 'td3'):
        # The deterministic off-policy pair. They share SAC's replay-buffer
        # settings but have no entropy temperature, and being deterministic
        # they have no exploration of their own: without action_noise the
        # policy is greedy the moment the random warm-up ends, which is not a
        # property of the algorithm but of leaving the argument unset. The
        # noise scale is the SB3 documented starting point for these methods.
        from stable_baselines3 import DDPG, TD3
        from stable_baselines3.common.noise import NormalActionNoise
        Algo = DDPG if algo == 'ddpg' else TD3
        hp = {k: v for k, v in SAC_HYPERPARAMS.items() if k != 'ent_coef'}
        if os.environ.get('ADM1_NATIVE_HP'):
            hp['learning_rate'] = 1e-3          # library default for DDPG/TD3
        hp['action_noise'] = NormalActionNoise(
            mean=np.zeros(2), sigma=0.1 * np.ones(2))
    else:
        raise ValueError(f"Unknown algo '{algo}'.")

    # Normalise the reward stream for the on-policy methods.  Observations are
    # already scaled by NormalizedADM1Env, so norm_obs stays off and the saved
    # policy is evaluated exactly like every other model.
    if algo in ON_POLICY_ALGOS:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        train_env = VecNormalize(DummyVecEnv([lambda e=train_env: e]),
                                 norm_obs=False, norm_reward=True, gamma=hp['gamma'])
        # EvalCallback calls sync_envs_normalization, which asserts structurally
        # that both envs are VecNormalize -- it does not check whether anything
        # actually needs syncing.  The eval env keeps raw rewards so reported
        # returns stay comparable with the off-policy runs.
        eval_env = VecNormalize(DummyVecEnv([lambda e=eval_env: e]),
                                norm_obs=False, norm_reward=False, training=False)

    # Hyperparameter overrides for the sensitivity sweep.  The values above are
    # the ones every reported run uses; these let a run vary one of them
    # without a second copy of this script, and they are recorded in
    # run_meta.json like any other setting, so a swept run is identifiable
    # after the fact.
    _sweep = {}
    if os.environ.get('ADM1_LR'):
        hp['learning_rate'] = _sweep['learning_rate'] = float(os.environ['ADM1_LR'])
    if os.environ.get('ADM1_ARCH'):
        units = [int(u) for u in os.environ['ADM1_ARCH'].split(',')]
        hp['policy_kwargs'] = dict(hp.get('policy_kwargs') or {})
        hp['policy_kwargs']['net_arch'] = units
        _sweep['net_arch'] = units
    if os.environ.get('ADM1_BATCH') and 'batch_size' in hp:
        hp['batch_size'] = _sweep['batch_size'] = int(os.environ['ADM1_BATCH'])
    # The gradient-update budget, for the matched-compute comparison. The two
    # classes are far apart on it as configured -- an off-policy method takes
    # one update per environment step against PPO's 0.039 -- so these move the
    # count without touching anything else: ADM1_EPOCHS raises the passes an
    # on-policy method makes over each rollout, ADM1_GRAD_STEPS lowers the
    # updates an off-policy method makes per step.
    if os.environ.get('ADM1_EPOCHS') and 'n_epochs' in hp:
        hp['n_epochs'] = _sweep['n_epochs'] = int(os.environ['ADM1_EPOCHS'])
    if os.environ.get('ADM1_TRAIN_FREQ') and 'train_freq' in hp:
        hp['train_freq'] = _sweep['train_freq'] = int(os.environ['ADM1_TRAIN_FREQ'])

    model = Algo(
        policy_name,
        train_env,
        verbose=verbose,
        seed=seed,
        device=device,
        tensorboard_log=str(run_dir / 'tensorboard'),
        **hp,
    )

    callbacks = CallbackList([
        EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / 'best_model'),
            log_path=str(run_dir / 'eval'),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
        ),
        CheckpointCallback(
            save_freq=100_000,
            save_path=str(run_dir / 'checkpoints'),
            name_prefix='sac_model',
        ),
    ])
    if algo not in ON_POLICY_ALGOS:
        callbacks.callbacks.append(_make_ent_clamp_callback())

    t0 = time.time()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=(verbose >= 1),
        )
    except KeyboardInterrupt:
        print("\n  [Interrupted — saving current model]")

    elapsed = time.time() - t0
    model.save(str(run_dir / 'final_model'))

    meta = {
        'algo':            algo.upper(),
        'env':             'ADM1Env_Std',
        'scenarios':       PAPER_SCENARIOS,
        'reward_config':   reward_config_name,
        'normalize':       normalize,
        'step_size_days':  step_size,
        'obs_mode':        obs_mode,
        'seed':            seed,
        'total_timesteps': total_timesteps,
        'elapsed_min':     round(elapsed / 60, 1),
        'hyperparams':     {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                                else str(v))
                            for k, v in hp.items()},
    }
    # Present only on a sensitivity-sweep run, and naming just the settings
    # that were moved off their reported values, so the swept runs can be
    # selected without inferring them from the hyperparameters.
    if _sweep:
        meta['sweep'] = {k: (v if isinstance(v, (int, float, str)) else str(v))
                         for k, v in _sweep.items()}
    with open(run_dir / 'run_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    train_env.close()
    eval_env.close()
    print(f"\n  Done in {elapsed/60:.1f} min → {run_dir}")
    return run_dir


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train SAC on standard ADM1 with 6-scenario curriculum',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--reward-config', type=str, default='safety_first',
                        help='Reward configuration: sf-sac=safety_first, '
                             'st-sac=safety_target, baseline-rl=balanced')
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--timesteps',  type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument('--output-dir', type=str, default='models_std_paper')
    parser.add_argument('--obs-mode',   type=str, default='scada',
                        choices=['scada', 'trend', 'dev', 'full'])
    parser.add_argument('--device',     type=str, default='auto')
    parser.add_argument('--verbose',    type=int, default=1)
    parser.add_argument('--normalize', action='store_true',
                        help='Min-max normalise obs/action spaces to [-1,1].')
    parser.add_argument('--history', type=int, default=1,
                        help='stack this many past observations (1 = memoryless)')
    parser.add_argument('--algo', type=str, default='sac', choices=['sac', 'tqc', 'ppo', 'recurrentppo', 'trpo', 'ddpg', 'td3', 'a2c', 'crossq', 'ars'])
    parser.add_argument('--vfa-soft', type=float, default=None,
                        help='override the derived soft VFA limit (robustness study)')
    parser.add_argument('--vfa-hard', type=float, default=None,
                        help='override the derived hard VFA limit (robustness study)')
    parser.add_argument('--step-size', type=float, default=0.01041667,
                        help='Control interval in days (0.01041667=15min, 1.0=daily).')
    args = parser.parse_args()

    train(
        reward_config_name = args.reward_config,
        normalize          = args.normalize,
        step_size          = args.step_size,
        seed               = args.seed,
        obs_mode           = args.obs_mode,
        total_timesteps    = args.timesteps,
        output_dir         = args.output_dir,
        device             = args.device,
        verbose            = args.verbose,
        history            = args.history,
        algo               = args.algo,
        vfa_soft           = args.vfa_soft,
        vfa_hard           = args.vfa_hard,
    )


if __name__ == '__main__':
    main()

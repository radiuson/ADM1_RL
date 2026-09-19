"""ADM1 digester as a constrained MDP, for the OmniSafe algorithms.

Why this exists
---------------
Everywhere else in this project the production/safety trade-off is traced by
sweeping a scalar penalty weight (``lw0p5``, ``lw1``, ``lw2``, ``lw5``): the
safety term is folded into the reward and the weight is tuned until the
resulting violation rate lands where we want it.  That is a hand-tuned
Lagrange multiplier, and it has two defects.  The weight has no physical
units, so a reviewer cannot tell whether the sweep covered the interesting
range; and the violation rate that comes out is whatever the weight happens
to produce, rather than a rate we asked for.

The constrained formulation states the problem the way a plant states it:
maximise methane subject to the soft-VFA excursion rate staying under a
limit.  The reward carries production only, the cost carries the excursion
indicator, and the frontier is parameterised by ``cost_limit`` -- a number in
the same units as the figure axis (excursions per 60-day episode).

Reward and cost
---------------
``reward``  production + energy + stability, with every safety penalty scale
            set to zero (see ``CMDP_REWARD``), so the safety signal reaches
            the agent only through the cost channel.
``cost``    1.0 on any step whose total VFA exceeds the soft limit, else 0.0.
            This is the same indicator the evaluation reports as the
            violation rate, so ``cost_limit / max_episode_steps`` is directly
            the target violation rate.

Episodes are 60 days at a one-day control step, so a target rate of 5 %
corresponds to ``cost_limit = 3.0``.
"""
from __future__ import annotations

import os
import sys
from typing import Any, ClassVar

import numpy as np
import torch
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omnisafe.common.logger import Logger
from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import OmnisafeSpace

from env.adm1_gym_env_std import ADM1Env_Std
from env.normalized_wrapper import NormalizedADM1Env

PAPER_SCENARIOS = ['low_load', 'nominal', 'plant_load', 'high_load', 'peak_load',
                   'fog_surge', 'elevated_start', 'acidified_recovery']

# Production, energy and stability terms exactly as the unconstrained runs use
# them; every safety scale is zeroed so that _calculate_safety_penalty returns
# 0.0 and the reward channel carries no constraint information.
CMDP_REWARD = {
    'production_scale':      2000.0,
    'penalty_type':          'linear+constant',
    'ph_penalty_scale':      0.0,
    'ph_constant_penalty':   0.0,
    'vfa_penalty_scale':     0.0,
    'vfa_constant_penalty':  0.0,
    'nh3_penalty_scale':     0.0,
    'nh3_constant_penalty':  0.0,
    'energy_penalty_max':    0.2,
    'stability_penalty_max': 0.1,
}


@env_register
class ADM1CMDP(CMDP):
    """Multi-scenario ADM1 digester with the VFA excursion as an explicit cost."""

    _support_envs: ClassVar[list[str]] = ['ADM1-v0']
    metadata: ClassVar[dict[str, int]] = {}

    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True
    _num_envs = 1

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        super().__init__(env_id)
        self._device = torch.device(kwargs.get('device', 'cpu'))
        self._scenarios = list(kwargs.get('scenarios', PAPER_SCENARIOS))
        self._rng = np.random.default_rng(kwargs.get('seed', 0))

        self._envs = {
            s: NormalizedADM1Env(
                ADM1Env_Std(s, reward_config=CMDP_REWARD, obs_mode='scada', step_size=1.0)
            )
            for s in self._scenarios
        }
        ref = self._envs[self._scenarios[0]]
        self._observation_space: OmnisafeSpace = ref.observation_space
        self._action_space: OmnisafeSpace = ref.action_space
        self._max_episode_steps = int(ref.env.max_steps)

        self._env = ref
        self._vfa_soft = ref.env.VFA_SOFT_KGCOD_M3

        # Reported once per episode so the training log carries the violation
        # rate in the same units as the evaluation tables.
        self.env_spec_log = {'Env/ViolationRate': 0.0, 'Env/MeanCH4': 0.0}
        self._ep_viol = 0
        self._ep_ch4 = 0.0
        self._ep_len = 0

    # -- helpers ---------------------------------------------------------

    def _vfa(self) -> float:
        st = self._env.env.solver.state
        return float(sum(st[k] for k in ('S_va', 'S_bu', 'S_pro', 'S_ac')))

    def _t(self, x, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(x, dtype=dtype, device=self._device)

    # -- CMDP interface --------------------------------------------------

    def step(self, action: torch.Tensor):
        a = action.detach().cpu().numpy().astype(np.float32)
        obs, reward, terminated, truncated, info = self._env.step(a)

        cost = 1.0 if self._vfa() > self._vfa_soft else 0.0
        self._ep_viol += int(cost)
        self._ep_ch4 += float(self._env.env.solver.q_ch4)
        self._ep_len += 1

        if terminated or truncated:
            info['final_observation'] = self._t(obs)
            self.env_spec_log['Env/ViolationRate'] = 100.0 * self._ep_viol / max(self._ep_len, 1)
            self.env_spec_log['Env/MeanCH4'] = self._ep_ch4 / max(self._ep_len, 1)

        return (self._t(obs), self._t(reward), self._t(cost),
                self._t(terminated, torch.bool), self._t(truncated, torch.bool), info)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.set_seed(seed)
        # A fresh scenario each episode, matching the curriculum the
        # unconstrained runs train on.
        self._env = self._envs[self._scenarios[self._rng.integers(len(self._scenarios))]]
        obs, info = self._env.reset(seed=int(self._rng.integers(1 << 30)))
        self._ep_viol, self._ep_ch4, self._ep_len = 0, 0.0, 0
        return self._t(obs), info

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    def spec_log(self, logger: Logger) -> None:
        logger.store({k: v for k, v in self.env_spec_log.items()})

    def set_seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def render(self) -> Any:
        raise NotImplementedError

    def close(self) -> None:
        for e in self._envs.values():
            e.close()

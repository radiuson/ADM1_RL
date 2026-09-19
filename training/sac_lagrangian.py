"""SAC with a Lagrangian multiplier on the safety cost.

Shaping the reward with a penalty weight leaves the weight itself
unspecified: it is not a quantity an operator can state, and the sweep in
this work shows that its value, not the form of the penalty, is what fixes
the operating point.  The constrained formulation replaces it with a
quantity that can be stated - the share of days on which the VFA limit may
be exceeded - and learns the multiplier that enforces it.

    maximise  E[sum r]   subject to   E[share of steps with VFA > limit] <= d

The multiplier follows dual ascent on the constraint violation,
    lambda <- max(0, lambda + eta * (J_C - d)),
with J_C estimated from the episodes seen since the last update.  The
reward passed to SAC is r - lambda * c, where c is 1 on a violating step
and 0 otherwise, so the safety term carries no hand-chosen scale.
"""

from collections import deque
from typing import Optional

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CostWrapper(gym.Wrapper):
    """Expose the safety cost and apply the current multiplier to the reward."""

    def __init__(self, env, limit_kgcod_m3: float):
        super().__init__(env)
        self.limit = float(limit_kgcod_m3)
        self.lambda_ = 0.0
        # The dual regulates whatever this window reports, so it must reflect
        # the current policy.  Sixty-four episodes span more policy change
        # than one dual interval, and the multiplier then keeps pressing on
        # a rate the policy has already left behind - measured here as a
        # training estimate near 27 % against an evaluated 16 %.
        self.episode_costs = deque(maxlen=8)
        self._ep_cost = 0.0
        self._ep_len = 0

    def reset(self, **kwargs):
        self._ep_cost = 0.0
        self._ep_len = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        base = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        vfa = info.get('total_vfa')
        if vfa is None:
            vfa = base._calculate_total_vfa(base.current_state)
        cost = 1.0 if vfa > self.limit else 0.0
        self._ep_cost += cost
        self._ep_len += 1
        info['cost'] = cost
        if terminated or truncated:
            self.episode_costs.append(self._ep_cost / max(self._ep_len, 1))
        return obs, reward - self.lambda_ * cost, terminated, truncated, info

    def current_violation_rate(self) -> Optional[float]:
        if not self.episode_costs:
            return None
        return float(np.mean(self.episode_costs))


class LagrangianCallback(BaseCallback):
    """Dual ascent on the multiplier held by the CostWrapper."""

    def __init__(self, wrapper: CostWrapper, budget: float,
                 lr: float = 0.5, update_every: int = 2000,
                 lambda_max: float = 50.0, verbose: int = 0):
        super().__init__(verbose)
        self.wrapper = wrapper
        self.budget = float(budget)
        self.lr = float(lr)
        self.update_every = int(update_every)
        self.lambda_max = float(lambda_max)
        self.history = []

    def _on_step(self) -> bool:
        if self.n_calls % self.update_every:
            return True
        rate = self.wrapper.current_violation_rate()
        if rate is None:
            return True
        self.wrapper.lambda_ = float(np.clip(
            self.wrapper.lambda_ + self.lr * (rate - self.budget),
            0.0, self.lambda_max))
        self.history.append((self.num_timesteps, rate, self.wrapper.lambda_))
        if self.verbose:
            print(f"  t={self.num_timesteps:>7}  violation={rate:.3f}  "
                  f"budget={self.budget:.3f}  lambda={self.wrapper.lambda_:.3f}")
        return True


class PIDLagrangianCallback(BaseCallback):
    """Dual update with proportional and derivative action (Stooke et al. 2020).

    Plain dual ascent is integral control on the constraint violation: the
    multiplier only responds to accumulated error, so it lags a process whose
    own time constant is long and then overshoots.  The measured effect here is
    a budget that is systematically under-delivered, by up to fifteen points at
    the loosest setting.

    The proportional term responds to the current violation immediately, and the
    derivative term acts on the rise in cost rather than on the error, one-sided
    so that it damps an approaching violation without slowing recovery once the
    constraint is met again.  Only the integral term accumulates.

    Responsiveness in the dual is useless if the signal it acts on is stale, so
    this is paired with a short cost window on the wrapper.

        I  <- max(0, I + K_I * (J_C - d))
        D  <- max(0, J_C - J_C_prev)
        lambda <- max(0, K_P * (J_C - d) + I + K_D * D)
    """

    def __init__(self, wrapper: CostWrapper, budget: float,
                 K_P: float = 5.0, K_I: float = 0.5, K_D: float = 5.0,
                 update_every: int = 2000, lambda_max: float = 50.0,
                 verbose: int = 0):
        super().__init__(verbose)
        self.wrapper = wrapper
        self.budget = float(budget)
        self.K_P, self.K_I, self.K_D = float(K_P), float(K_I), float(K_D)
        self.update_every = int(update_every)
        self.lambda_max = float(lambda_max)
        self.integral = 0.0
        self.prev_cost = None
        self.history = []

    def _on_step(self) -> bool:
        if self.n_calls % self.update_every:
            return True
        rate = self.wrapper.current_violation_rate()
        if rate is None:
            return True
        err = rate - self.budget
        self.integral = max(0.0, self.integral + self.K_I * err)
        deriv = 0.0 if self.prev_cost is None else max(0.0, rate - self.prev_cost)
        self.prev_cost = rate
        lam = self.K_P * err + self.integral + self.K_D * deriv
        self.wrapper.lambda_ = float(np.clip(lam, 0.0, self.lambda_max))
        self.history.append((self.num_timesteps, rate, self.wrapper.lambda_))
        if self.verbose:
            print(f"  t={self.num_timesteps:>7}  J_C={rate:.3f}  d={self.budget:.3f}  "
                  f"P={self.K_P*err:+.3f} I={self.integral:.3f} "
                  f"D={self.K_D*deriv:+.3f}  lambda={self.wrapper.lambda_:.3f}")
        return True

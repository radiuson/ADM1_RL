#!/usr/bin/env python3
"""
Greedy NMPC for Standard (2-dim) ADM1 Environment
===================================================

Nonlinear Model Predictive Controller that uses the ADM1 solver directly as
the prediction model.  At each step it optimises [q_ad, feed_mult] over a
short horizon N_p using scipy.optimize.minimize, then applies only the first
action (receding-horizon principle).

Because the prediction model is the true ADM1 (oracle disturbance assumption),
this represents an optimistic upper bound for model-based control and is
labelled "NMPC" in the paper.

Paper label: "NMPC"
Observation indices (full 12-dim obs from ADM1Env_Std):
    obs[0]  = total_vfa      obs[4] = pH    obs[9] = q_ch4
"""

import copy
from typing import Optional

import numpy as np
import scipy.optimize

from baselines.baseline_controllers import BaseController


# ── Environment snapshot (std env — no thermal states) ───────────────────────

class _StdEnvSnapshot:
    """Save / restore ADM1Env_Std state without mutating the live episode."""

    __slots__ = [
        'adm1_state',
        'current_step', 'current_time_days',
        'q_ch4', 'prev_q_ch4', 'total_ch4_produced', 'episode_reward',
        'q_ad_current', 'feed_mult_current',
    ]

    def __init__(self, env):
        self.adm1_state          = env.solver.get_state()
        self.current_step        = env.current_step
        self.current_time_days   = env.current_time_days
        self.q_ch4               = env.q_ch4
        self.prev_q_ch4          = env.prev_q_ch4
        self.total_ch4_produced  = env.total_ch4_produced
        self.episode_reward      = env.episode_reward
        self.q_ad_current        = env.q_ad_current
        self.feed_mult_current   = env.feed_mult_current

    def restore(self, env):
        env.solver.set_state(self.adm1_state)
        env.current_step        = self.current_step
        env.current_time_days   = self.current_time_days
        env.q_ch4               = self.q_ch4
        env.prev_q_ch4          = self.prev_q_ch4
        env.total_ch4_produced  = self.total_ch4_produced
        env.episode_reward      = self.episode_reward
        env.q_ad_current        = self.q_ad_current
        env.feed_mult_current   = self.feed_mult_current


# ── NMPC controller ───────────────────────────────────────────────────────────

class NMPCStdController(BaseController):
    """
    Greedy NMPC for ADM1Env_Std.

    At each control step:
      1. Save current environment state.
      2. Optimise a flat sequence of N_p × 2 decision variables over the
         prediction horizon to maximise cumulative CH4 subject to VFA and pH
         constraints (via penalty).
      3. Restore original state; apply only the first action.

    Constraint handling:
      - VFA > vfa_soft_limit : linear penalty × vfa_penalty_w
      - pH outside [ph_low, ph_high] : quadratic penalty × ph_penalty_w

    Args:
        N_p              Prediction horizon (steps).  Default 8 (≈ 2 hours).
        vfa_soft_limit   VFA threshold above which penalty activates (kg COD/m³).
        vfa_penalty_w    Penalty weight per unit VFA excess per step.
        ph_penalty_w     Penalty weight per unit pH deviation squared per step.
        max_iter         scipy L-BFGS-B iteration limit per control step.
        nominal_q_ad     Initial guess for q_ad in optimisation.
        nominal_feed     Initial guess for feed_mult in optimisation.
    """

    def __init__(
        self,
        N_p: int               = 8,
        vfa_soft_limit: float  = 0.28,   # just below hard threshold 0.30
        vfa_penalty_w: float   = 100.0,  # reduced from 500 — less conservative
        ph_penalty_w: float    = 300.0,
        ph_low: float          = 6.8,
        ph_high: float         = 7.8,
        max_iter: int          = 100,    # increased from 30 — better optimization
        nominal_q_ad: float    = 178.0,
        nominal_feed: float    = 1.0,
    ):
        super().__init__(name="NMPC")
        self.N_p             = N_p
        self.vfa_soft        = vfa_soft_limit
        self.vfa_w           = vfa_penalty_w
        self.ph_w            = ph_penalty_w
        self.ph_low          = ph_low
        self.ph_high         = ph_high
        self.max_iter        = max_iter
        self.nominal_q_ad    = nominal_q_ad
        self.nominal_feed    = nominal_feed

        # Cached warm start from previous solve
        self._last_u: Optional[np.ndarray] = None
        self._env = None   # set on first call to get_action

    # ─────────────────────────────────────────────────────────────────────────

    def _rollout(self, u_flat: np.ndarray, env) -> float:
        """
        Roll out u_flat (N_p × 2 flat array) in a *copy* of env.
        Returns negative cumulative reward (scipy minimises).
        Note: parameter order matches scipy.optimize.minimize calling convention:
        fun(x, *args) → _rollout(u_flat, env).
        """
        u = u_flat.reshape(self.N_p, 2)
        snapshot = _StdEnvSnapshot(env)
        cumulative = 0.0

        for k in range(self.N_p):
            action = np.array([u[k, 0], u[k, 1]], dtype=np.float32)
            _, reward, terminated, truncated, info = env.step(action)

            # Add constraint penalty on top of environment reward
            vfa = info.get('total_vfa', 0.0)
            ph  = info.get('pH', 7.2)

            vfa_excess = max(0.0, vfa - self.vfa_soft)
            ph_dev = max(0.0, self.ph_low - ph, ph - self.ph_high)

            penalty = self.vfa_w * vfa_excess + self.ph_w * ph_dev ** 2
            cumulative += reward - penalty

            if terminated or truncated:
                # Heavy terminal penalty for early termination
                cumulative -= 500.0
                break

        snapshot.restore(env)
        return -cumulative   # negate for minimisation

    def _make_u0(self) -> np.ndarray:
        """Warm-start from previous solution (shift by 1, repeat last)."""
        if self._last_u is not None and len(self._last_u) == self.N_p * 2:
            u = self._last_u.reshape(self.N_p, 2)
            u_shifted = np.vstack([u[1:], u[-1:]])
            return u_shifted.flatten()
        # Cold start
        u0 = np.tile([self.nominal_q_ad, self.nominal_feed], self.N_p)
        return u0.astype(np.float64)

    # ─────────────────────────────────────────────────────────────────────────

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Optimise over prediction horizon and return the first action.

        Requires that ``set_env(env)`` has been called before the episode starts,
        OR that ``observation`` is the current env observation (the env is stored
        internally after the first call).

        Returns:
            2-dim action array [q_ad, feed_mult].
        """
        if self._env is None:
            raise RuntimeError(
                "NMPCStdController.set_env(env) must be called before get_action()."
            )

        self.step_count += 1
        env = self._env

        # The two decision variables differ in scale by ~400x (q_ad spans
        # 250 m3/d, feed_mult spans 0.6).  L-BFGS-B applies one finite-
        # difference step to every dimension, so a step small enough for
        # feed_mult perturbs q_ad by far less than the ODE solver tolerance
        # and its gradient evaluates to exactly zero - the optimiser then
        # returns the initial guess unchanged.  Optimising in a normalised
        # [0, 1] space gives both dimensions a comparable step.
        lb = np.tile([50.0,  0.7], self.N_p)
        ub = np.tile([300.0, 1.3], self.N_p)
        span = ub - lb
        bounds = [(0.0, 1.0)] * (2 * self.N_p)

        u0 = (self._make_u0() - lb) / span

        result = scipy.optimize.minimize(
            lambda z, e: self._rollout(lb + z * span, e),
            u0,
            args=(env,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': self.max_iter, 'ftol': 1e-6, 'eps': 1e-3},
        )

        u_opt = (lb + result.x * span).reshape(self.N_p, 2)
        self._last_u = (lb + result.x * span).copy()

        q_ad      = float(np.clip(u_opt[0, 0], 50.0, 300.0))
        feed_mult = float(np.clip(u_opt[0, 1],  0.7,   1.3))

        return np.array([q_ad, feed_mult], dtype=np.float32)

    def set_env(self, env) -> None:
        """Attach the live ADM1Env_Std instance (must be called before each episode)."""
        self._env = env

    def reset(self):
        self.step_count = 0
        self._last_u = None
        self._env    = None

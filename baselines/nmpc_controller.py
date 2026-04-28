#!/usr/bin/env python3
"""
NMPC (Oracle) Controller for ADM1 Biogas
==========================================

Oracle Nonlinear Model Predictive Control: uses the true ADM1 model as the
internal prediction model and assumes perfect knowledge of future disturbances
(influent substrate concentrations, temperature ramps).

This represents a practical upper bound for model-based control and is
included as a rigorous comparison baseline for the RL experiments.  Because
it has access to future disturbance information unavailable in real deployment,
its performance should be interpreted as an optimistic benchmark rather than
an achievable target.

See ``mpc_controller.MPCController`` for the realistic variant that applies
the standard persistent-disturbance assumption.

Paper label: "NMPC (oracle)"
"""

import copy
import time
import warnings
from typing import Optional

import numpy as np
import scipy.optimize

from baselines.baseline_controllers import BaseController


class _EnvSnapshot:
    """Lightweight container for saving and restoring ADM1Env_v2 state.

    Used by NMPCController to roll back the environment after each
    optimisation rollout so the real episode state is never mutated.
    """

    __slots__ = [
        # ADM1 solver state
        'adm1_state', 'T_L', 'T_a', 'q_ad', 'Q_HEX', 'T_env',
        # Environment bookkeeping
        'current_step', 'current_time_days',
        'q_ch4', 'prev_q_ch4', 'total_ch4_produced', 'episode_reward',
        'q_ad_current', 'feed_mult_current', 'Q_HEX_current',
        # Solver influent (needed so the restored step uses the correct feed)
        'influent',
    ]

    def __init__(self, env):
        s = env.solver
        self.adm1_state         = s.get_state()
        self.T_L                = s.T_L
        self.T_a                = s.T_a
        self.q_ad               = s.q_ad
        self.Q_HEX              = s.Q_HEX
        self.T_env              = s.T_env
        self.influent           = copy.copy(getattr(s, 'influent', {}))
        self.current_step       = env.current_step
        self.current_time_days  = env.current_time_days
        self.q_ch4              = env.q_ch4
        self.prev_q_ch4         = env.prev_q_ch4
        self.total_ch4_produced = env.total_ch4_produced
        self.episode_reward     = env.episode_reward
        self.q_ad_current       = env.q_ad_current
        self.feed_mult_current  = env.feed_mult_current
        self.Q_HEX_current      = env.Q_HEX_current

    def restore(self, env):
        s = env.solver
        s.set_state(self.adm1_state)
        s.T_L     = self.T_L
        s.T_a     = self.T_a
        s.q_ad    = self.q_ad
        s.Q_HEX   = self.Q_HEX
        s.T_env   = self.T_env
        if self.influent:
            s.set_influent(self.influent)
        env.current_step        = self.current_step
        env.current_time_days   = self.current_time_days
        env.q_ch4               = self.q_ch4
        env.prev_q_ch4          = self.prev_q_ch4
        env.total_ch4_produced  = self.total_ch4_produced
        env.episode_reward      = self.episode_reward
        env.q_ad_current        = self.q_ad_current
        env.feed_mult_current   = self.feed_mult_current
        env.Q_HEX_current       = self.Q_HEX_current


class NMPCController(BaseController):
    """
    Oracle Nonlinear Model Predictive Control using the true ADM1 model.

    At every control step:

    1. Save the current environment state (snapshot).
    2. Optimise a sequence of H actions to maximise cumulative reward,
       rolling out the true ADM1 ODE as the internal prediction model.
    3. Restore the snapshot.
    4. Apply only the first action of the optimal sequence (receding horizon).

    The optimiser is L-BFGS-B with optional random restarts.  A small
    regularisation term penalises deviations from the nominal operating
    point to prevent short-horizon exploitation of extreme actuator positions
    that harm long-run dynamics.

    Args:
        env        : ADM1Env_v2 instance (holds state and solver reference).
        horizon    : Prediction horizon in environment steps (default 4 = 1 hour).
        max_iter   : Maximum optimiser iterations per control step (default 10).
        n_restarts : Number of random restarts for the optimiser (default 1).
        verbose    : Print timing every ``verbose`` steps; 0 = silent.
    """

    # Conservative action bounds for the optimiser.  These are tighter than
    # the environment limits to avoid local-minimum traps caused by extreme
    # actuator positions when the MPC horizon is short relative to ADM1 dynamics.
    ACTION_LOW  = np.array([100.0,  0.8, -3000.0], dtype=np.float64)
    ACTION_HIGH = np.array([250.0,  1.2,  3000.0], dtype=np.float64)
    ACTION_DIM  = 3

    # Nominal operating point for regularisation
    ACTION_NOM = np.array([178.47, 1.0, 0.0], dtype=np.float64)
    # Regularisation weight: small enough not to dominate, large enough to
    # break ties between equally-rewarded trajectories.
    REG_WEIGHT = 1e-3

    def __init__(
        self,
        env,
        horizon: int = 4,
        max_iter: int = 10,
        n_restarts: int = 1,
        verbose: int = 0,
    ):
        super().__init__(name="NMPC")
        self.env        = env
        self.H          = horizon
        self.max_iter   = max_iter
        self.n_restarts = n_restarts
        self.verbose    = verbose

        # Flattened (H × ACTION_DIM) bounds for scipy
        lb = np.tile(self.ACTION_LOW,  self.H)
        ub = np.tile(self.ACTION_HIGH, self.H)
        self._bounds = list(zip(lb, ub))

        # Warm-start solution cache (H × ACTION_DIM)
        self._prev_sol: Optional[np.ndarray] = None
        self._total_opt_time = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────────

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute optimal first action via receding-horizon optimisation.

        Args:
            observation: Current environment observation vector.

        Returns:
            3-dim action array [q_ad, feed_mult, Q_HEX], clipped to
            ACTION_LOW / ACTION_HIGH.
        """
        self.step_count += 1
        snap = _EnvSnapshot(self.env)

        x0 = self._warm_start(snap)
        best_x, best_val = x0, np.inf

        t0 = time.perf_counter()
        for _ in range(self.n_restarts):
            result = scipy.optimize.minimize(
                self._objective,
                x0=x0,
                args=(snap,),
                method='L-BFGS-B',
                bounds=self._bounds,
                options={'maxiter': self.max_iter, 'ftol': 1e-5, 'gtol': 1e-4},
            )
            if result.fun < best_val:
                best_val = result.fun
                best_x   = result.x
            if self.n_restarts > 1:
                x0 = self._random_x0()

        opt_time = time.perf_counter() - t0
        self._total_opt_time += opt_time

        if self.verbose > 0 and self.step_count % self.verbose == 0:
            print(f"[NMPC] step={self.step_count:4d}  "
                  f"opt_time={opt_time:.2f}s  "
                  f"total={self._total_opt_time / 60:.1f}min")

        # Restore real env state (rollouts mutated it)
        snap.restore(self.env)

        self._prev_sol = best_x.reshape(self.H, self.ACTION_DIM)
        first_action   = self._prev_sol[0]
        return np.clip(first_action, self.ACTION_LOW, self.ACTION_HIGH).astype(np.float32)

    def reset(self):
        self.step_count      = 0
        self._prev_sol       = None
        self._total_opt_time = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _objective(self, u_flat: np.ndarray, snap: '_EnvSnapshot') -> float:
        """
        Rollout H steps with candidate action sequence; return negative
        cumulative reward (minimisation objective).

        The environment state is restored at the start of each call so
        repeated scipy function evaluations are independent.
        """
        snap.restore(self.env)
        u_seq = u_flat.reshape(self.H, self.ACTION_DIM)
        total_neg_reward = 0.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for h in range(self.H):
                action = np.clip(u_seq[h], self.ACTION_LOW, self.ACTION_HIGH)
                try:
                    _, reward, terminated, truncated, _ = self.env.step(action)
                except Exception:
                    # Solver failure → large penalty for remaining horizon
                    total_neg_reward += 10.0 * (self.H - h)
                    break
                total_neg_reward -= reward
                if terminated or truncated:
                    total_neg_reward += 2.0 * (self.H - h - 1)
                    break

        # Regularisation: penalise deviations from nominal operating point
        action_range   = self.ACTION_HIGH - self.ACTION_LOW
        normalised_dev = (u_seq - self.ACTION_NOM) / action_range
        total_neg_reward += self.REG_WEIGHT * float(np.sum(normalised_dev ** 2))

        return total_neg_reward

    def _warm_start(self, snap: '_EnvSnapshot') -> np.ndarray:
        """Construct a warm-start action sequence for the optimiser."""
        if self._prev_sol is not None:
            # Shift previous solution: drop step 0, repeat last step
            shifted = np.roll(self._prev_sol, -1, axis=0)
            shifted[-1] = shifted[-2]
            return shifted.flatten()
        # Default: current operating point repeated over the horizon
        x0 = np.tile(
            [snap.q_ad_current, snap.feed_mult_current, snap.Q_HEX_current],
            self.H,
        )
        return x0.astype(np.float64)

    def _random_x0(self) -> np.ndarray:
        """Sample a random feasible starting point for optimiser restarts."""
        lo = np.tile(self.ACTION_LOW,  self.H)
        hi = np.tile(self.ACTION_HIGH, self.H)
        return np.random.uniform(lo, hi)

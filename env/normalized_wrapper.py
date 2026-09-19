"""
Observation/action normalisation wrapper for ADM1Env_Std.

Both spaces are min-max mapped to [-1, 1] using the environment's declared
bounds.  The map is fixed and analytic (no running statistics), so the same
transform can be applied at deployment time to real plant measurements.

Motivation
----------
The raw SCADA observation spans four orders of magnitude
(vfa_alk_ratio 0-2 vs q_ch4 0-4000).  Without normalisation the first layer
is dominated by the largest-range channels and the FOS/TAC signal contributes
~0.05% of the input magnitude.  The raw action space (q_ad 50-300,
feed_mult 0.7-1.3) likewise breaks the standard SB3 assumption of a
[-1, 1] continuous action space.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class NormalizedADM1Env(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._o_lo = np.asarray(env.observation_space.low,  dtype=np.float64)
        self._o_hi = np.asarray(env.observation_space.high, dtype=np.float64)
        self._a_lo = np.asarray(env.action_space.low,       dtype=np.float64)
        self._a_hi = np.asarray(env.action_space.high,      dtype=np.float64)

        self._o_span = np.where(self._o_hi - self._o_lo > 0, self._o_hi - self._o_lo, 1.0)
        self._a_span = self._a_hi - self._a_lo

        n_o = self._o_lo.shape[0]
        n_a = self._a_lo.shape[0]
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(n_o,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(n_a,), dtype=np.float32)

    def norm_obs(self, obs):
        """Raw observation -> [-1, 1].  Public: reuse for SCADA deployment."""
        z = (np.asarray(obs, dtype=np.float64) - self._o_lo) / self._o_span
        return np.clip(2.0 * z - 1.0, -1.0, 1.0).astype(np.float32)

    def denorm_action(self, action):
        """Policy action in [-1, 1] -> raw actuator units."""
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        return (self._a_lo + (a + 1.0) * 0.5 * self._a_span).astype(np.float32)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        return self.norm_obs(obs), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(self.denorm_action(action))
        info['raw_action'] = self.denorm_action(action)
        return self.norm_obs(obs), r, term, trunc, info

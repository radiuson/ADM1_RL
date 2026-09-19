"""Give the policy a memory of the recent past.

The controller sees six plant-measurable signals of a thirty-eight state
reactor, so the control problem is partially observed.  A memoryless policy can
be arbitrarily worse than a history-based one on such a problem, and the
symptoms reported for memoryless off-policy methods under partial observability
- unstable returns and high variance across seeds - match what the fixed-window
policies here show.

Conventional controllers already carry memory: the integral term of a PI loop
accumulates past error, and the rule-based supervisor carries its previous feed
rate.  Stacking the last k observations gives the learned policy the same kind
of state, in the simplest form that needs no change of algorithm.
"""
from collections import deque

import gymnasium as gym
import numpy as np


class HistoryWrapper(gym.ObservationWrapper):
    def __init__(self, env, k=5):
        super().__init__(env)
        self.k = int(k)
        self._buf = deque(maxlen=self.k)
        lo = np.tile(env.observation_space.low, self.k)
        hi = np.tile(env.observation_space.high, self.k)
        self.observation_space = gym.spaces.Box(lo, hi, dtype=np.float32)

    def observation(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if not self._buf:
            for _ in range(self.k):
                self._buf.append(obs)
        else:
            self._buf.append(obs)
        return np.concatenate(list(self._buf)).astype(np.float32)

    def reset(self, **kw):
        self._buf.clear()
        return super().reset(**kw)

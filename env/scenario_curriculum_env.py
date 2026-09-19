#!/usr/bin/env python3
"""
Scenario-Difficulty Curriculum Environment
==========================================

Wraps ADM1Env_v2 and progressively introduces harder scenarios during
training, based on the agent's cumulative step count.

Stage format — two variants are supported:
    2-tuple: (step_threshold, [scenarios])
             reward config is fixed to the one passed at __init__
    3-tuple: (step_threshold, [scenarios], reward_config_name)
             reward config switches when the stage is entered;
             reward_config_name must be a key in REWARD_CONFIGS

The last entry's threshold is ignored (treated as infinity).

Repeat a scenario in the pool list to increase its sampling probability,
e.g. ['high_load', 'high_load', 'nominal'] gives high_load 2/3 probability.

Predefined schedules
--------------------
DEFAULT_STAGES     nominal → +high+low → all 6           (300 k steps)
HIGH_LOAD_FIRST    high_load → +nominal+low → all 6      (300 k steps)
FAST_STAGES        nominal (50 k) → +h+l (150 k) → all 6
SLOW_STAGES        nominal (150 k) → +h+l (250 k) → all 6
UNIFORM_STAGES     all 6 from step 0
EF_STAGES          Exploration-First: all 6 + sf_linear_only (200 k)
                   → all 6 + safety_first (400 k)
                   → hard-weighted + safety_first (∞)
"""

import random
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

try:
    from .adm1_gym_env import ADM1Env_v2
except ImportError:
    from adm1_gym_env import ADM1Env_v2

try:
    from training.reward_configs import REWARD_CONFIGS as _REWARD_CONFIGS
except ImportError:
    _REWARD_CONFIGS = {}


ALL_SCENARIOS = [
    'nominal', 'high_load', 'low_load',
    'shock_load', 'temperature_drop', 'cold_winter',
]

# ── Predefined stage schedules ────────────────────────────────────────────────

DEFAULT_STAGES: List = [
    (100_000, ['nominal']),
    (200_000, ['nominal', 'high_load', 'low_load']),
    (300_000, ALL_SCENARIOS),
]

HIGH_LOAD_FIRST_STAGES: List = [
    (100_000, ['high_load']),
    (200_000, ['high_load', 'nominal', 'low_load']),
    (300_000, ALL_SCENARIOS),
]

FAST_STAGES: List = [
    (50_000,  ['nominal']),
    (150_000, ['nominal', 'high_load', 'low_load']),
    (300_000, ALL_SCENARIOS),
]

UNIFORM_STAGES: List = [
    (300_000, ALL_SCENARIOS),
]

SLOW_STAGES: List = [
    (150_000, ['nominal']),
    (250_000, ['nominal', 'high_load', 'low_load']),
    (300_000, ALL_SCENARIOS),
]

# Exploration-First SDC:
#   Stage 1 — all 6 scenarios, sf_linear_only reward (no constant penalty)
#             Policy freely explores high-production operating points.
#   Stage 2 — all 6 scenarios, safety_first reward (full linear+constant penalty)
#             Policy learns hard safety constraints from a high-production baseline.
#   Stage 3 — hard scenarios oversampled (high_load 28%, cold_winter 28%,
#             temperature_drop 22%, others ~7% each), safety_first
#             Fine-tunes safety generalisation on the most challenging scenarios.
_EF_HARD_POOL = (
    ['high_load'] * 5 +
    ['cold_winter'] * 5 +
    ['temperature_drop'] * 4 +
    ['nominal'] * 2 +
    ['shock_load'] * 2 +
    ['low_load'] * 2
)  # 20 items → approx 25/25/20/10/10/10 %

EF_STAGES: List = [
    (200_000, ALL_SCENARIOS,    'sf_linear_only'),
    (400_000, ALL_SCENARIOS,    'safety_first'),
    (999_999, _EF_HARD_POOL,   'safety_first'),
]


# ── Environment ───────────────────────────────────────────────────────────────

class ScenarioCurriculumEnv(gym.Env):
    """
    Curriculum env that randomly selects from the current stage's scenario
    pool on each episode reset, optionally switching the reward config between
    stages when 3-tuple stage entries are used.

    Args:
        reward_config:  Base reward config dict (used when a stage has no
                        explicit reward name, or as fallback).
        stages:         List of stage entries.  Each entry is either
                        (threshold, [scenarios]) or
                        (threshold, [scenarios], reward_config_name).
        obs_mode:       'full' (13-dim) or 'simple' (5-dim).
        rng_seed:       Seed for the scenario-selection RNG (independent from
                        the env's physics seed).
    """

    DEFAULT_STAGES = DEFAULT_STAGES
    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        reward_config: Dict,
        stages: Optional[List] = None,
        obs_mode: str = 'full',
        rng_seed: Optional[int] = None,
    ):
        super().__init__()
        self._base_reward_config = reward_config
        self._obs_mode = obs_mode
        self._rng = random.Random(rng_seed)
        self._total_steps: int = 0
        self._current_scenario: Optional[str] = None
        self._env: Optional[ADM1Env_v2] = None

        raw_stages = stages if stages is not None else DEFAULT_STAGES
        self._stages = self._parse_stages(raw_stages)

        # Set initial reward config from Stage 1 (or base if no name given)
        first_reward_name = self._stages[0][2]
        if first_reward_name and first_reward_name in _REWARD_CONFIGS:
            self._current_reward_config = _REWARD_CONFIGS[first_reward_name]
            self._current_reward_name: Optional[str] = first_reward_name
        else:
            self._current_reward_config = reward_config
            self._current_reward_name = None

        # Build initial inner env to expose spaces
        first_scenario = self._stages[0][1][0]
        self._build_inner_env(first_scenario)

        self.observation_space = self._env.observation_space
        self.action_space      = self._env.action_space

    # ── Stage parsing ─────────────────────────────────────────────────────────

    @staticmethod
    def _parse_stages(stages: List) -> List[Tuple]:
        """Normalise 2-tuple and 3-tuple stage entries to 3-tuples."""
        out = []
        for entry in stages:
            if len(entry) == 3:
                out.append((entry[0], list(entry[1]), entry[2]))
            else:
                out.append((entry[0], list(entry[1]), None))
        return out

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_current_stage(self) -> Tuple[List[str], Optional[str]]:
        """Return (scenario_pool, reward_config_name) for the current step."""
        for threshold, pool, reward_name in self._stages:
            if self._total_steps < threshold:
                return pool, reward_name
        last = self._stages[-1]
        return last[1], last[2]

    def _get_available_scenarios(self) -> List[str]:
        pool, _ = self._get_current_stage()
        return pool

    def _build_inner_env(self, scenario: str) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
        self._env = ADM1Env_v2(
            scenario_name=scenario,
            reward_config=self._current_reward_config,
            obs_mode=self._obs_mode,
        )
        self._current_scenario = scenario

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        pool, reward_name = self._get_current_stage()

        # Switch reward config if the stage specifies a different one
        reward_changed = False
        if reward_name is not None and reward_name != self._current_reward_name:
            new_cfg = _REWARD_CONFIGS.get(reward_name)
            if new_cfg is not None:
                self._current_reward_config = new_cfg
                self._current_reward_name   = reward_name
                reward_changed = True

        chosen = self._rng.choice(pool)
        if reward_changed or chosen != self._current_scenario:
            self._build_inner_env(chosen)

        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        self._total_steps += 1
        return self._env.step(action)

    def render(self):
        return self._env.render()

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def set_tau_a(self, tau_a: float) -> None:
        if self._env is not None and hasattr(self._env, 'set_tau_a'):
            self._env.set_tau_a(tau_a)

    @property
    def current_scenario(self) -> Optional[str]:
        return self._current_scenario

    @property
    def current_reward_name(self) -> Optional[str]:
        return self._current_reward_name

    @property
    def total_steps(self) -> int:
        return self._total_steps

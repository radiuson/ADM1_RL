#!/usr/bin/env python3
"""
ADM1 Gymnasium Environment — Standard (No Thermal Extension)
=============================================================

Gymnasium-compatible environment for RL-based control of anaerobic digestion
using the *standard* ADM1 model (no dynamic temperature states, no Q_HEX).
Temperature is fixed at 35 °C throughout.

Used for the cross-model comparison experiments: all controllers are evaluated
on the four load-variation scenarios (nominal, high_load, low_load, shock_load),
with thermal scenarios excluded.

Key differences vs ADM1Env_v2 (adm1_gym_env.py):
  - Uses ADM1SolverStd (38-state ODE, no T_L/T_a)
  - Action space is 2-dim: [q_ad, feed_mult]   (no Q_HEX)
  - Full obs is 11-dim: drops T_L_norm at index 12
  - Simple obs is 4-dim: [pH, q_ch4, q_ad_current, feed_mult_current]
  - Only four scenarios supported: nominal, high_load, low_load, shock_load
  - No thermal parameter initialisation in reset()
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

try:
    from .adm1_solver_std import ADM1SolverStd
    from .scenario_manager import ScenarioManager
except ImportError:
    from adm1_solver_std import ADM1SolverStd
    from scenario_manager import ScenarioManager


# Scenarios supported by this environment (temperature scenarios excluded)
STD_SCENARIOS = ['nominal', 'high_load', 'low_load', 'shock_load']


class ADM1Env_Std(gym.Env):
    """
    Gymnasium environment wrapping the standard (non-thermal) ADM1 solver.

    Observation space — full (12-dim, obs_mode='full', default):
        idx  variable            range                description
        ---  --------            -----                -----------
         0   total_vfa           [0, 0.8]  kg COD/m³   S_ac+S_pro+S_bu+S_va
         1   alkalinity          [0, 0.3]  kmol/m³      0.8·S_IC + S_NH3
         2   vfa_alk_ratio       [0, 2.0]  —             total_vfa / alkalinity
         3   S_h2                [0, 1e-4] kg COD/m³   dissolved hydrogen
         4   pH                  [5.5, 8.5] —
         5   S_nh3               [0, 0.01] kmol N/m³   free ammonia
         6   S_IN                [0, 0.2]  kmol N/m³   inorganic nitrogen
         7   X_ac                [0, 3.0]  kg COD/m³   acetoclastic biomass
         8   X_h2                [0, 3.0]  kg COD/m³   hydrogenotrophic biomass
         9   q_ch4               [0, 600]  m³/day       methane flow rate
        10   q_ad_current        [50, 300] m³/day       current feed flow
        11   feed_mult_current   [0.7, 1.3] —           current feed multiplier
        (no T_L_norm — temperature is fixed at 35 °C in the standard model)

    Simple observation (4-dim, obs_mode='simple'):
        [pH, q_ch4, q_ad_current, feed_mult_current]
        = full_obs indices [4, 9, 10, 11]

    Action space (2-dim continuous):
        [q_ad (m³/day), feed_mult]
        bounds: [50, 300] × [0.7, 1.3]

    Safety violation thresholds (same as ADM1Env_v2):
        pH < 6.8 or pH > 7.8,  VFA > 0.2 kg COD/m³,  NH3 > 0.002 kmol/m³

    Episode termination (catastrophic failure only):
        pH < 5.8 or > 8.8,  NH3 > 0.01,  VFA > 0.8

    Supported scenarios: nominal, high_load, low_load, shock_load
    """

    metadata = {'render_modes': ['human']}

    # Indices in the full 11-dim observation used by 'simple' mode
    SIMPLE_OBS_INDICES = [4, 9, 10, 11]
    # [pH, q_ch4, q_ad_current, feed_mult_current]

    def __init__(
        self,
        scenario_name: str = 'nominal',
        step_size: float = 0.01041667,   # 15 minutes in days
        V_liq: float = 3400.0,
        V_gas: float = 300.0,
        reward_config: Optional[Dict] = None,
        enable_disturbances: bool = True,
        random_seed: Optional[int] = None,
        obs_mode: str = 'full',          # 'full' (11-dim) | 'simple' (4-dim)
    ):
        """
        Initialise the standard ADM1 environment.

        Args:
            scenario_name:       One of STD_SCENARIOS.
            step_size:           Simulation timestep in days (default: 15 min).
            V_liq:               Liquid volume (m³).
            V_gas:               Gas headspace volume (m³).
            reward_config:       Custom reward configuration dict (optional).
            enable_disturbances: Inject scenario disturbances (e.g. shock_load spike).
            random_seed:         RNG seed for reproducibility.
            obs_mode:            'full' (11-dim) or 'simple' (4-dim).
        """
        super().__init__()

        if scenario_name not in STD_SCENARIOS:
            raise ValueError(
                f"Scenario '{scenario_name}' not supported by ADM1Env_Std. "
                f"Supported: {STD_SCENARIOS}"
            )

        self.scenario_name       = scenario_name
        self.step_size           = step_size
        self.V_liq               = V_liq
        self.V_gas               = V_gas
        self.enable_disturbances = enable_disturbances
        self._seed               = random_seed
        self.obs_mode            = obs_mode

        # Scenario configuration
        self.scenario_manager  = ScenarioManager()
        self.scenario_config   = self.scenario_manager.load_scenario(scenario_name)
        self.scenario_duration = self.scenario_config['duration_days']
        self.max_steps         = int(self.scenario_duration / step_size)

        # Reward configuration
        self.reward_config = reward_config or {
            'production_scale':     2000.0,
            'ph_penalty_scale':     2.0,
            'vfa_penalty_scale':    3.0,
            'nh3_penalty_scale':    50.0,
            'energy_penalty_max':   0.2,
            'stability_penalty_max': 0.1,
        }

        # ── Action space: [q_ad, feed_mult] — 2-dim, no Q_HEX ────────────────
        self.action_space = spaces.Box(
            low=np.array([50.0, 0.7], dtype=np.float32),
            high=np.array([300.0, 1.3], dtype=np.float32),
            dtype=np.float32,
        )

        # ── Observation space ─────────────────────────────────────────────────
        _full_low = np.array([
            0.0,   # 0:  total_vfa
            0.0,   # 1:  alkalinity
            0.0,   # 2:  vfa_alk_ratio
            0.0,   # 3:  S_h2
            5.5,   # 4:  pH
            0.0,   # 5:  S_nh3
            0.0,   # 6:  S_IN
            0.0,   # 7:  X_ac
            0.0,   # 8:  X_h2
            0.0,   # 9:  q_ch4
            50.0,  # 10: q_ad_current
            0.7,   # 11: feed_mult_current
        ], dtype=np.float32)
        _full_high = np.array([
            0.8,    # 0:  total_vfa
            0.3,    # 1:  alkalinity
            2.0,    # 2:  vfa_alk_ratio
            1e-4,   # 3:  S_h2
            8.5,    # 4:  pH
            0.01,   # 5:  S_nh3
            0.2,    # 6:  S_IN
            3.0,    # 7:  X_ac
            3.0,    # 8:  X_h2
            600.0,  # 9:  q_ch4
            300.0,  # 10: q_ad_current
            1.3,    # 11: feed_mult_current
        ], dtype=np.float32)

        if obs_mode == 'simple':
            obs_low  = _full_low[self.SIMPLE_OBS_INDICES]
            obs_high = _full_high[self.SIMPLE_OBS_INDICES]
        else:
            obs_low, obs_high = _full_low, _full_high

        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # ── Solver ────────────────────────────────────────────────────────────
        self.solver = ADM1SolverStd(V_liq=V_liq, V_gas=V_gas)

        # ── Internal state ────────────────────────────────────────────────────
        self.current_step        = 0
        self.current_time_days   = 0.0
        self.current_state       = None
        self.q_ch4               = 0.0
        self.prev_q_ch4          = 0.0
        self.total_ch4_produced  = 0.0
        self.episode_reward      = 0.0
        self.q_ad_current        = 178.4674
        self.feed_mult_current   = 1.0

        self.ph_history  = []
        self.vfa_history = []
        self.ch4_history = []
        self.violation_count = {
            'ph_low': 0, 'ph_high': 0, 'vfa_high': 0, 'nh3_high': 0
        }

        self._load_data()

    # ──────────────────────────────────────────────────────────────────────────
    # Data loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_data(self):
        data_path = Path(__file__).parent / 'data' / 'digester_influent.csv'
        if not data_path.exists():
            raise FileNotFoundError(
                f"Influent data not found at {data_path}.\n"
                f"Expected: env/data/digester_influent.csv"
            )
        self.influent_df = pd.read_csv(data_path)

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_influent_dict(self, step: int = 0) -> Dict[str, float]:
        sim_time   = step * self.step_size
        csv_index  = int(sim_time / 0.010417)
        csv_index  = min(csv_index, len(self.influent_df) - 1)
        row        = self.influent_df.iloc[csv_index]

        required = [
            'S_su', 'S_aa', 'S_fa', 'S_va', 'S_bu', 'S_pro', 'S_ac',
            'S_h2', 'S_ch4', 'S_IC', 'S_IN', 'S_I',
            'X_xc', 'X_ch', 'X_pr', 'X_li', 'X_su', 'X_aa', 'X_fa',
            'X_c4', 'X_pro', 'X_ac', 'X_h2', 'X_I',
            'S_cation', 'S_anion',
        ]
        return {v: float(row[v]) if v in row else 0.0 for v in required}

    def _calculate_total_vfa(self, state: Dict[str, float]) -> float:
        return (state.get('S_ac', 0.0) + state.get('S_pro', 0.0)
                + state.get('S_bu', 0.0) + state.get('S_va', 0.0))

    def _calculate_alkalinity(self, state: Dict[str, float]) -> float:
        return state.get('S_IC', 0.0) * 0.8 + state.get('S_nh3', 0.0)

    # ──────────────────────────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        if seed is not None:
            self._seed = seed
            np.random.seed(seed)

        # Reset counters
        self.current_step       = 0
        self.current_time_days  = 0.0
        self.total_ch4_produced = 0.0
        self.prev_q_ch4         = 0.0
        self.q_ch4              = 0.0
        self.episode_reward     = 0.0
        self.q_ad_current       = 178.4674
        self.feed_mult_current  = 1.0

        self.ph_history  = []
        self.vfa_history = []
        self.ch4_history = []
        self.violation_count = {
            'ph_low': 0, 'ph_high': 0, 'vfa_high': 0, 'nh3_high': 0
        }

        # Support runtime scenario switching
        if options and 'scenario' in options:
            new_scenario = options['scenario']
            if new_scenario not in STD_SCENARIOS:
                raise ValueError(
                    f"Scenario '{new_scenario}' not supported. Use one of {STD_SCENARIOS}"
                )
            self.scenario_name    = new_scenario
            self.scenario_config  = self.scenario_manager.load_scenario(new_scenario)
            self.scenario_duration = self.scenario_config['duration_days']
            self.max_steps        = int(self.scenario_duration / self.step_size)

        # Load initial state
        initial_state = self.scenario_manager.get_initial_state(self.scenario_name)
        self.solver.set_state(initial_state)

        # Set initial influent
        base_influent     = self._build_influent_dict(0)
        modified_influent = self.scenario_manager.apply_influent_multiplier(
            base_influent, self.scenario_name
        )
        self.solver.set_influent(modified_influent)
        self.solver.set_flow_rate(self.q_ad_current)

        self.current_state = self.solver.state.copy()
        observation = self._get_observation()

        info = {
            'scenario':      self.scenario_name,
            'max_steps':     self.max_steps,
            'duration_days': self.scenario_duration,
        }
        return observation, info

    # ──────────────────────────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.

        Args:
            action: 2-dim array [q_ad (m³/day), feed_mult]

        Returns:
            observation, reward, terminated, truncated, info
        """
        q_ad            = np.clip(float(action[0]), 50.0, 300.0)
        feed_multiplier = np.clip(float(action[1]),  0.7,   1.3)

        self.q_ad_current      = q_ad
        self.feed_mult_current = feed_multiplier
        self.solver.set_flow_rate(q_ad)

        # Build influent
        base_influent     = self._build_influent_dict(self.current_step)
        modified_influent = self.scenario_manager.apply_influent_multiplier(
            base_influent, self.scenario_name
        )

        # Apply feed multiplier to organic substrates
        for key in ['X_ch', 'X_pr', 'X_li', 'X_xc']:
            if key in modified_influent:
                modified_influent[key] *= feed_multiplier

        # Inject disturbances (load-based only; temperature disturbances ignored)
        if self.enable_disturbances:
            disturbance = self.scenario_manager.check_disturbances(self.current_time_days)
            if disturbance and disturbance['type'] != 'temperature_ramp':
                modified_influent = self.scenario_manager.apply_disturbance(
                    modified_influent, disturbance
                )

        self.solver.set_influent(modified_influent)

        # Integrate one step (no Q_HEX)
        try:
            new_state, q_ch4 = self.solver.step(dt=self.step_size)
            self.current_state = new_state
            self.q_ch4         = max(0.0, q_ch4)

            self.current_step       += 1
            self.current_time_days  += self.step_size
            self.total_ch4_produced += self.q_ch4 * self.step_size

            self.ph_history.append(new_state.get('pH', 7.0))
            self.vfa_history.append(self._calculate_total_vfa(new_state))
            self.ch4_history.append(self.q_ch4)

        except Exception as e:
            print(f"Solver error at step {self.current_step}: {e}")
            self.q_ch4 = 0.0

        observation = self._get_observation()
        reward      = self._calculate_reward()
        self.episode_reward += reward

        terminated = self._is_catastrophic_failure()
        truncated  = self.current_step >= self.max_steps

        info = self._build_info_dict(terminated or truncated)
        self.prev_q_ch4 = self.q_ch4

        return observation, reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────────
    # Observation
    # ──────────────────────────────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """Build 11-dim (full) or 4-dim (simple) observation vector."""
        s = self.current_state

        total_vfa     = self._calculate_total_vfa(s)
        alkalinity    = self._calculate_alkalinity(s)
        vfa_alk_ratio = total_vfa / max(alkalinity, 1e-6)

        full_obs = np.array([
            total_vfa,                   # 0
            alkalinity,                  # 1
            vfa_alk_ratio,               # 2
            s.get('S_h2',    0.0),       # 3
            s.get('pH',      7.0),       # 4
            s.get('S_nh3',   0.0),       # 5
            s.get('S_IN',    0.0),       # 6
            s.get('X_ac',    0.0),       # 7
            s.get('X_h2',    0.0),       # 8
            self.q_ch4,                  # 9
            self.q_ad_current,           # 10
            self.feed_mult_current,      # 11
        ], dtype=np.float32)

        if self.obs_mode == 'simple':
            return full_obs[self.SIMPLE_OBS_INDICES]
        return full_obs

    # ──────────────────────────────────────────────────────────────────────────
    # Reward
    # ──────────────────────────────────────────────────────────────────────────

    def _calculate_reward(self) -> float:
        rc = self.reward_config

        production_reward = self.q_ch4 / rc['production_scale']
        safety_penalty    = self._calculate_safety_penalty()
        energy_penalty    = -(self.q_ad_current / 300.0) ** 2 * rc['energy_penalty_max']
        volatility        = abs(self.q_ch4 - self.prev_q_ch4)
        stability_penalty = -(volatility / 100.0) * rc['stability_penalty_max']

        return production_reward + safety_penalty + energy_penalty + stability_penalty

    def _calculate_safety_penalty(self) -> float:
        s = self.current_state
        rc = self.reward_config
        total_penalty = 0.0
        penalty_type  = rc.get('penalty_type', 'quadratic')

        pH = s.get('pH', 7.0)
        if pH < 6.8:
            dev = 6.8 - pH
            if penalty_type == 'linear+constant':
                total_penalty -= rc.get('ph_constant_penalty', 0.5)
                total_penalty -= dev * rc['ph_penalty_scale']
            else:
                total_penalty -= (dev ** 2) * rc['ph_penalty_scale']
            self.violation_count['ph_low'] += 1
        elif pH > 7.8:
            dev = pH - 7.8
            if penalty_type == 'linear+constant':
                total_penalty -= rc.get('ph_constant_penalty', 0.5)
                total_penalty -= dev * rc['ph_penalty_scale']
            else:
                total_penalty -= (dev ** 2) * rc['ph_penalty_scale']
            self.violation_count['ph_high'] += 1

        total_vfa = self._calculate_total_vfa(s)
        if total_vfa > 0.2:
            excess = total_vfa - 0.2
            if penalty_type == 'linear+constant':
                total_penalty -= rc.get('vfa_constant_penalty', 1.0)
                total_penalty -= excess * rc['vfa_penalty_scale']
            else:
                total_penalty -= (excess ** 2) * rc['vfa_penalty_scale']
            self.violation_count['vfa_high'] += 1

        S_nh3 = s.get('S_nh3', 0.0)
        if S_nh3 > 0.002:
            excess = S_nh3 - 0.002
            if penalty_type == 'linear+constant':
                total_penalty -= rc.get('nh3_constant_penalty', 1.5)
                total_penalty -= excess * rc['nh3_penalty_scale']
            else:
                total_penalty -= (excess ** 2) * rc['nh3_penalty_scale']
            self.violation_count['nh3_high'] += 1

        return total_penalty

    def _is_catastrophic_failure(self) -> bool:
        s   = self.current_state
        pH  = s.get('pH', 7.0)
        if pH < 5.8 or pH > 8.8:
            return True
        if s.get('S_nh3', 0.0) > 0.01:
            return True
        if self._calculate_total_vfa(s) > 0.8:
            return True
        return False

    # ──────────────────────────────────────────────────────────────────────────
    # Info
    # ──────────────────────────────────────────────────────────────────────────

    def _build_info_dict(self, is_done: bool) -> Dict[str, Any]:
        s = self.current_state
        info = {
            'step':            self.current_step,
            'time_days':       self.current_time_days,
            'q_ch4':           self.q_ch4,
            'q_co2':           self.solver.q_co2,
            'pH':              s.get('pH', 7.0),
            'total_vfa':       self._calculate_total_vfa(s),
            'alkalinity':      self._calculate_alkalinity(s),
            'S_nh3':           s.get('S_nh3', 0.0),
            'q_ad':            self.q_ad_current,
            'feed_multiplier': self.feed_mult_current,
            'total_ch4':       self.total_ch4_produced,
        }
        if is_done:
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.current_step,
            }
            info['ch4_produced'] = self.total_ch4_produced
            info['ch4_avg_flow'] = (
                self.total_ch4_produced
                / max(1, self.current_step * self.step_size)
            )
            info['avg_ph']           = np.mean(self.ph_history)  if self.ph_history  else 7.0
            info['avg_vfa']          = np.mean(self.vfa_history) if self.vfa_history else 0.0
            info['ch4_std']          = np.std(self.ch4_history)  if self.ch4_history else 0.0
            info['violation_count']  = self.violation_count.copy()
        return info

    # ──────────────────────────────────────────────────────────────────────────
    # Render / close
    # ──────────────────────────────────────────────────────────────────────────

    def render(self, mode: str = 'human'):
        if mode == 'human':
            s         = self.current_state
            total_vfa = self._calculate_total_vfa(s)
            alk       = self._calculate_alkalinity(s)
            print(f"\n=== Step {self.current_step} (Day {self.current_time_days:.2f}) ===")
            print(f"Scenario : {self.scenario_name}")
            print(f"pH       : {s.get('pH', 0):.2f}")
            print(f"CH4 flow : {self.q_ch4:.2f} m³/d")
            print(f"Total VFA: {total_vfa:.4f} kg COD/m³")
            print(f"Alk      : {alk:.4f} kmol/m³")
            print(f"q_ad     : {self.q_ad_current:.1f} m³/d")
            print(f"feed_mult: {self.feed_mult_current:.2f}")

    def close(self):
        pass


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 70)
    print("ADM1Env_Std — smoke test")
    print("=" * 70)

    for sc in STD_SCENARIOS:
        print(f"\n[{sc}]")
        try:
            env = ADM1Env_Std(scenario_name=sc)
            obs, info = env.reset(seed=42)
            print(f"  obs shape  : {obs.shape}")
            print(f"  action shape: {env.action_space.shape}")

            for _ in range(3):
                act = env.action_space.sample()
                obs, rew, term, trunc, info = env.step(act)
            print(f"  step ok — pH={info['pH']:.2f}, q_ch4={info['q_ch4']:.1f}")
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Done.")

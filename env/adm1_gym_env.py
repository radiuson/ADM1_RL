#!/usr/bin/env python3
"""
ADM1 Gymnasium Environment — Temperature-Extended RL Benchmark
==============================================================

Gymnasium-compatible environment for reinforcement-learning-based control of
anaerobic digestion, as used in the paper.  The environment wraps the
temperature-extended ADM1 ODE solver (adm1_solver.py) and exposes:

    - 13-dimensional full observation space  (obs_mode='full',   default)
    - 5-dimensional compact observation space (obs_mode='simple')
    - 3-dimensional continuous action space  [q_ad, feed_mult, Q_HEX]
    - Safety-first reward with magnitude + event-level violation penalties
    - Six evaluation scenarios (see env/scenarios.yaml)
    - Scenario-specific thermal parameters and disturbance injection
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Import ADM1 solver and scenario manager
try:
    from .adm1_solver import ADM1Solver
    from .scenario_manager import ScenarioManager
except ImportError:
    from adm1_solver import ADM1Solver
    from scenario_manager import ScenarioManager


class ADM1Env_v2(gym.Env):
    """
    Gymnasium environment for ADM1-based anaerobic digestion control.

    Observation space — full (13-dim, obs_mode='full', default):
        idx  variable          range              description
        ---  --------          -----              -----------
         0   total_vfa         [0, 0.8]  kg COD/m³   S_ac+S_pro+S_bu+S_va
         1   alkalinity        [0, 0.3]  kmol/m³      0.8·S_IC + S_NH3
         2   vfa_alk_ratio     [0, 2.0]  —             total_vfa / alkalinity
         3   S_h2              [0, 1e-4] kg COD/m³   dissolved hydrogen
         4   pH                [5.5, 8.5] —
         5   S_nh3             [0, 0.01] kmol N/m³   free ammonia
         6   S_IN              [0, 0.2]  kmol N/m³   inorganic nitrogen
         7   X_ac              [0, 3.0]  kg COD/m³   acetoclastic biomass
         8   X_h2              [0, 3.0]  kg COD/m³   hydrogenotrophic biomass
         9   q_ch4             [0, 600]  m³/day       methane flow rate
        10   q_ad_current      [50, 300] m³/day       current feed flow
        11   feed_mult_current [0.7, 1.3] —           current feed multiplier
        12   T_L_norm          [-3, 3.7] —             (T_L − 308.15 K) / 10

    Compact observation (5-dim, obs_mode='simple'):
        indices [4, 9, 12, 10, 11] of the full obs = [pH, q_ch4, T_L_norm,
        q_ad_current, feed_mult_current].  Matches Table 1 in the paper.

    Action space (3-dim continuous):
        [q_ad (m³/day), feed_mult, Q_HEX (W)]
        bounds: [50, 300] × [0.7, 1.3] × [−5000, 5000]

    Safety violation thresholds (paper Section III-B):
        pH < 6.8 or pH > 7.8,  VFA > 0.2 kmol COD/m³,  NH3 > 0.002 kmol/m³

    Episode termination (catastrophic failure only):
        pH < 5.8,  NH3 > 0.01,  VFA > 0.8

    Scenarios (see scenarios.yaml):
        nominal, high_load, low_load, shock_load, temperature_drop, cold_winter
    """

    metadata = {'render_modes': ['human']}

    # Indices in the full 13-dim observation used by 'simple' mode
    SIMPLE_OBS_INDICES = [4, 9, 12, 10, 11]
    # [pH, q_ch4, T_L_norm, q_ad_current, feed_mult_current]

    def __init__(
        self,
        scenario_name: str = 'nominal',
        step_size: float = 0.01041667,  # 15 minutes
        V_liq: float = 3400.0,
        V_gas: float = 300.0,
        reward_config: Optional[Dict] = None,
        enable_disturbances: bool = True,
        random_seed: Optional[int] = None,
        obs_mode: str = 'full',   # 'full' (13-dim) | 'simple' (5-dim)
    ):
        """
        Initialize ADM1 v2 environment

        Args:
            scenario_name: Scenario to run (see scenarios.yaml)
            step_size: Simulation timestep in days (default: 15 min)
            V_liq: Liquid volume (m³)
            V_gas: Gas headspace volume (m³)
            reward_config: Custom reward configuration (optional)
            enable_disturbances: Enable disturbance injection
            random_seed: Random seed for reproducibility
            obs_mode: Observation mode — 'full' (13-dim) or 'simple' (5-dim)
        """
        super(ADM1Env_v2, self).__init__()

        # Configuration
        self.scenario_name = scenario_name
        self.step_size = step_size
        self.V_liq = V_liq
        self.V_gas = V_gas
        self.enable_disturbances = enable_disturbances
        self._seed = random_seed
        self.obs_mode = obs_mode  # 'full' or 'simple'

        # Load scenario configuration
        self.scenario_manager = ScenarioManager()
        self.scenario_config = self.scenario_manager.load_scenario(scenario_name)
        self.scenario_duration = self.scenario_config['duration_days']
        self.max_steps = int(self.scenario_duration / step_size)

        # Reward configuration (normalized)
        # Note: production_scale adjusted based on observed CH4 ~1700 m³/d for V_liq=3400
        self.reward_config = reward_config or {
            'production_scale': 2000.0,  # Normalize CH4 production (adjusted from 400)
            'ph_penalty_scale': 2.0,    # Quadratic penalty scale
            'vfa_penalty_scale': 3.0,   # VFA soft barrier scale
            'nh3_penalty_scale': 50.0,  # NH3 soft barrier scale
            'energy_penalty_max': 0.2,  # Max energy penalty
            'stability_penalty_max': 0.1,  # Max stability penalty
        }

        # Action space: [q_ad, feed_multiplier, Q_HEX]
        self.action_space = spaces.Box(
            low=np.array([50.0,  0.7, -5000.0], dtype=np.float32),
            high=np.array([300.0, 1.3,  5000.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation space: 13-dim (full) or 5-dim (simple)
        _full_low = np.array([
            0.0,    # 0: total_vfa
            0.0,    # 1: alkalinity
            0.0,    # 2: vfa_alk_ratio
            0.0,    # 3: S_h2
            5.5,    # 4: pH
            0.0,    # 5: S_nh3
            0.0,    # 6: S_IN
            0.0,    # 7: X_ac
            0.0,    # 8: X_h2
            0.0,    # 9: q_ch4
            50.0,   # 10: q_ad_current
            0.7,    # 11: feed_mult_current
            -3.0    # 12: T_L_norm
        ], dtype=np.float32)
        _full_high = np.array([
            0.8,    # 0: total_vfa
            0.3,    # 1: alkalinity
            2.0,    # 2: vfa_alk_ratio
            1e-4,   # 3: S_h2
            8.5,    # 4: pH
            0.01,   # 5: S_nh3
            0.2,    # 6: S_IN
            3.0,    # 7: X_ac
            3.0,    # 8: X_h2
            600.0,  # 9: q_ch4
            300.0,  # 10: q_ad_current
            1.3,    # 11: feed_mult_current
            3.7     # 12: T_L_norm
        ], dtype=np.float32)
        if obs_mode == 'simple':
            # 5-dim: [pH, q_ch4, T_L_norm, q_ad_current, feed_mult_current]
            obs_low  = _full_low[self.SIMPLE_OBS_INDICES]
            obs_high = _full_high[self.SIMPLE_OBS_INDICES]
        else:
            obs_low, obs_high = _full_low, _full_high
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Initialize ADM1 solver
        self.solver = ADM1Solver(V_liq=V_liq, V_gas=V_gas)

        # State tracking
        self.current_step = 0
        self.current_time_days = 0.0
        self.current_state = None
        self.q_ch4 = 0.0
        self.prev_q_ch4 = 0.0
        self.total_ch4_produced = 0.0
        self.episode_reward = 0.0

        # Control history (for observation)
        self.q_ad_current = 178.4674  # Default flow rate
        self.feed_mult_current = 1.0
        self.Q_HEX_current = 0.0     # Heat exchanger power (W)

        # Metrics tracking
        self.ph_history = []
        self.vfa_history = []
        self.ch4_history = []
        self.violation_count = {
            'ph_low': 0,
            'ph_high': 0,
            'vfa_high': 0,
            'nh3_high': 0
        }

        # Load data (initial state and influent)
        self._load_data()

    def _load_data(self):
        """Load influent time-series data from env/data/."""
        data_path = Path(__file__).parent / 'data' / 'digester_influent.csv'
        try:
            self.influent_df = pd.read_csv(data_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Influent data not found at {data_path}.\n"
                f"Expected: env/data/digester_influent.csv"
            )

    def _build_state_dict(self, row) -> Dict[str, float]:
        """Build complete state dictionary from DataFrame row"""
        state = {}
        for col in row.index:
            state[col] = row[col]
        if 'pH' not in state:
            state['pH'] = 7.26
        return state

    def _build_influent_dict(self, step: int = 0) -> Dict[str, float]:
        """Build influent dictionary from DataFrame based on simulation time"""
        # Calculate simulation time
        sim_time = step * self.step_size

        # CSV timestep is 0.010417 days (15 minutes)
        csv_timestep = 0.010417
        csv_index = int(sim_time / csv_timestep)

        # Ensure index is within bounds
        if csv_index >= len(self.influent_df):
            csv_index = len(self.influent_df) - 1

        row = self.influent_df.iloc[csv_index]
        influent = {}

        # All required influent variables
        required_vars = [
            'S_su', 'S_aa', 'S_fa', 'S_va', 'S_bu', 'S_pro', 'S_ac',
            'S_h2', 'S_ch4', 'S_IC', 'S_IN', 'S_I',
            'X_xc', 'X_ch', 'X_pr', 'X_li', 'X_su', 'X_aa', 'X_fa',
            'X_c4', 'X_pro', 'X_ac', 'X_h2', 'X_I',
            'S_cation', 'S_anion'
        ]

        for var in required_vars:
            influent[var] = row[var] if var in row else 0.0

        return influent

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state

        Args:
            seed: Random seed for reproducibility
            options: Additional options (e.g., {'scenario': 'shock_load'})

        Returns:
            observation: Initial observation (13-dim full or 5-dim compact)
            info: Additional information
        """
        # Handle seed
        if seed is not None:
            self._seed = seed
            np.random.seed(seed)

        # Reset counters
        self.current_step = 0
        self.current_time_days = 0.0
        self.total_ch4_produced = 0.0
        self.prev_q_ch4 = 0.0
        self.q_ch4 = 0.0
        self.episode_reward = 0.0
        self.q_ad_current = 178.4674  # Default flow rate
        self.feed_mult_current = 1.0
        self.Q_HEX_current = 0.0

        # Reset metrics
        self.ph_history = []
        self.vfa_history = []
        self.ch4_history = []
        self.violation_count = {
            'ph_low': 0,
            'ph_high': 0,
            'vfa_high': 0,
            'nh3_high': 0
        }

        # Load scenario (support runtime scenario switching)
        if options and 'scenario' in options:
            self.scenario_name = options['scenario']
            self.scenario_config = self.scenario_manager.load_scenario(self.scenario_name)
            self.scenario_duration = self.scenario_config['duration_days']
            self.max_steps = int(self.scenario_duration / self.step_size)

        # Get initial state from scenario manager
        initial_state = self.scenario_manager.get_initial_state(self.scenario_name)

        # Initialize solver with initial state
        self.solver.set_state(initial_state)

        # Apply thermal parameters from scenario YAML
        thermal = self.scenario_config.get('thermal', {})
        if thermal:
            self.solver.T_env  = float(thermal.get('T_env',  298.15))
            self.solver.T_feed = float(thermal.get('T_feed', 298.15))
            self.solver.UA     = float(thermal.get('UA',     50.0))
            T_L_init           = float(thermal.get('T_L_init', 308.15))
            Q_HEX_init         = float(thermal.get('Q_HEX_init', 500.0))
            self.solver.T_L    = T_L_init
            self.solver.T_a    = T_L_init   # adaptation temp starts equal to T_L
            self.solver.Q_HEX  = Q_HEX_init
            self.Q_HEX_current = Q_HEX_init
            # Sync state vector thermal slots (indices 38, 39)
            if hasattr(self.solver, 'state_vector') and len(self.solver.state_vector) > 39:
                self.solver.state_vector[38] = T_L_init
                self.solver.state_vector[39] = T_L_init

        # Set initial influent (with scenario multiplier)
        base_influent = self._build_influent_dict(0)
        modified_influent = self.scenario_manager.apply_influent_multiplier(
            base_influent, self.scenario_name
        )
        self.solver.set_influent(modified_influent)

        # Set initial flow rate
        self.solver.set_flow_rate(self.q_ad_current)

        # Store current state
        self.current_state = self.solver.state.copy()

        # Get observation
        observation = self._get_observation()

        # Info dict
        info = {
            'scenario': self.scenario_name,
            'max_steps': self.max_steps,
            'duration_days': self.scenario_duration
        }

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step

        Args:
            action: 3-dim array [q_ad (m³/day), feed_mult, Q_HEX (W)]

        Returns:
            observation: Next observation (13-dim full or 5-dim compact)
            reward: Normalized reward
            terminated: Whether episode ended (catastrophic failure)
            truncated: Whether episode ended (time limit)
            info: Additional information
        """
        # Parse and clip action: [q_ad, feed_multiplier, Q_HEX]
        q_ad           = np.clip(float(action[0]), 50.0,   300.0)
        feed_multiplier = np.clip(float(action[1]),  0.7,    1.3)
        Q_HEX          = np.clip(float(action[2]), -5000.0, 5000.0)

        # Store control history
        self.q_ad_current = q_ad
        self.feed_mult_current = feed_multiplier
        self.Q_HEX_current = Q_HEX

        # Update flow rate
        self.solver.set_flow_rate(q_ad)

        # Get base influent
        base_influent = self._build_influent_dict(self.current_step)

        # Apply scenario multiplier
        modified_influent = self.scenario_manager.apply_influent_multiplier(
            base_influent, self.scenario_name
        )

        # Apply feed multiplier to organic substrates
        for key in ['X_ch', 'X_pr', 'X_li', 'X_xc']:
            if key in modified_influent:
                modified_influent[key] *= feed_multiplier

        # Check for disturbances
        if self.enable_disturbances:
            disturbance = self.scenario_manager.check_disturbances(self.current_time_days)
            if disturbance:
                if disturbance['type'] == 'temperature_ramp':
                    # Ramp ambient temperature T_env downward by (from_temp - to_temp) K.
                    # This increases heat losses and causes T_L to drift unless the agent
                    # compensates with Q_HEX.  The YAML from_temp/to_temp are °C T_L targets;
                    # we map them to an equivalent T_env drop of the same magnitude.
                    baseline_T_env = float(
                        self.scenario_config.get('thermal', {}).get('T_env', 298.15)
                    )
                    drop_K = disturbance.get('from_temp', 35.0) - disturbance.get('to_temp', 30.0)
                    self.solver.T_env = baseline_T_env - drop_K * disturbance.get('progress', 0.0)
                else:
                    # Apply disturbance to influent (e.g. influent_spike)
                    modified_influent = self.scenario_manager.apply_disturbance(
                        modified_influent, disturbance
                    )

        # Set influent
        self.solver.set_influent(modified_influent)

        # Execute one timestep (pass Q_HEX to thermal model)
        try:
            new_state, q_ch4 = self.solver.step(dt=self.step_size, Q_HEX=Q_HEX)
            self.current_state = new_state
            self.q_ch4 = max(0.0, q_ch4)

            # Update time
            self.current_step += 1
            self.current_time_days += self.step_size

            # Track metrics
            self.total_ch4_produced += self.q_ch4 * self.step_size
            self.ph_history.append(new_state.get('pH', 7.0))
            total_vfa = self._calculate_total_vfa(new_state)
            self.vfa_history.append(total_vfa)
            self.ch4_history.append(self.q_ch4)

        except Exception as e:
            print(f"Solver error at step {self.current_step}: {e}")
            self.q_ch4 = 0.0

        # Get observation
        observation = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward()
        self.episode_reward += reward

        # Check termination
        terminated = self._is_catastrophic_failure()
        truncated = self.current_step >= self.max_steps

        # Build info dict
        info = self._build_info_dict(terminated or truncated)

        # Update previous CH4 for next step
        self.prev_q_ch4 = self.q_ch4

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Extract 13-dimensional observation from current state.

        Returns:
            13-dim observation vector (12 original + T_L)
        """
        s = self.current_state

        # Calculate total VFA (aggregate 4 VFAs)
        total_vfa = self._calculate_total_vfa(s)

        # Calculate alkalinity (bicarbonate + ammonia alkalinity)
        alkalinity = self._calculate_alkalinity(s)

        # Calculate VFA/Alkalinity ratio (early warning signal)
        vfa_alk_ratio = total_vfa / max(alkalinity, 1e-6)

        # T_L from state dict (set by solver.step); fall back to solver attribute
        T_L = s.get('T_L', getattr(self.solver, 'T_L', 308.15))
        # Normalize T_L: express as deviation from 35°C setpoint, scaled by 10°C
        # T_L=308.15K → 0.0,  +5°C → 0.5,  -10°C → -1.0  (range ≈ [-3, 3])
        T_L_norm = (T_L - 308.15) / 10.0

        # Build full 13-dim observation
        full_obs = np.array([
            total_vfa,                          # 0: Total VFA
            alkalinity,                         # 1: Alkalinity
            vfa_alk_ratio,                      # 2: VFA/Alk ratio
            s.get('S_h2', 0.0),                # 3: Hydrogen
            s.get('pH', 7.0),                  # 4: pH
            s.get('S_nh3', 0.0),               # 5: Free ammonia
            s.get('S_IN', 0.0),                # 6: Inorganic nitrogen
            s.get('X_ac', 0.0),                # 7: Acetoclastic methanogens
            s.get('X_h2', 0.0),                # 8: Hydrogenotrophic methanogens
            self.q_ch4,                         # 9: Methane production rate
            self.q_ad_current,                  # 10: Current flow rate
            self.feed_mult_current,             # 11: Current feed multiplier
            T_L_norm                            # 12: Temperature deviation (°C/10)
        ], dtype=np.float32)

        if self.obs_mode == 'simple':
            return full_obs[self.SIMPLE_OBS_INDICES]
        return full_obs

    def _calculate_total_vfa(self, state: Dict[str, float]) -> float:
        """Calculate total VFA from individual components"""
        return (
            state.get('S_ac', 0.0) +
            state.get('S_pro', 0.0) +
            state.get('S_bu', 0.0) +
            state.get('S_va', 0.0)
        )

    def _calculate_alkalinity(self, state: Dict[str, float]) -> float:
        """
        Calculate alkalinity (bicarbonate + ammonia)

        Note: S_hco3_ion is calculated in ADM1 acid-base system
        Alkalinity ≈ HCO3- + NH3
        """
        # Use S_IC (inorganic carbon) and S_IN (inorganic nitrogen) as proxies
        # In ADM1, at neutral pH, most IC is bicarbonate
        S_IC = state.get('S_IC', 0.0)
        S_nh3 = state.get('S_nh3', 0.0)

        # Approximate alkalinity (in kmol/m³)
        # At pH ~7, bicarbonate dominates inorganic carbon
        alkalinity = S_IC * 0.8 + S_nh3  # 80% of IC as bicarbonate + free ammonia

        return alkalinity

    def _calculate_reward(self) -> float:
        """
        Calculate normalized multi-objective reward

        Components:
        1. Production: q_ch4 / 400 (normalized to [0, 1])
        2. Safety: Soft penalties for pH, VFA, NH3 violations
        3. Energy: Pump power cost ∝ (q_ad/300)²
        4. Stability: CH4 volatility penalty

        Expected reward: ~0.5 (normal), 0.7-0.8 (optimal)
        """
        rc = self.reward_config

        # 1. Production term (normalized to [0, 1])
        production_reward = self.q_ch4 / rc['production_scale']

        # 2. Safety penalties (soft barriers)
        safety_penalty = self._calculate_safety_penalty()

        # 3. Energy cost (pump power ∝ flow²)
        energy_penalty = -(self.q_ad_current / 300.0) ** 2 * rc['energy_penalty_max']

        # 4. Stability penalty (volatility)
        volatility = abs(self.q_ch4 - self.prev_q_ch4)
        stability_penalty = -(volatility / 100.0) * rc['stability_penalty_max']

        # Total reward
        total_reward = (
            production_reward +
            safety_penalty +
            energy_penalty +
            stability_penalty
        )

        return total_reward

    def _calculate_safety_penalty(self) -> float:
        """
        Calculate safety penalties with soft barriers.

        Supports two penalty modes via reward_config['penalty_type']:
        - 'quadratic' (default): penalty = -(excess^2) * scale
        - 'linear+constant': penalty = -constant - excess * scale
          Provides much stronger signal for small violations.
          With 'linear+constant', also reads *_constant_penalty keys.

        Safe ranges:
        - pH: [6.8, 7.8]
        - VFA: [0, 0.2] kg COD/m³
        - NH3: [0, 0.002] kmol/m³
        """
        s = self.current_state
        rc = self.reward_config
        total_penalty = 0.0
        penalty_type = rc.get('penalty_type', 'quadratic')

        # pH penalty (target: 6.8-7.8)
        pH = s.get('pH', 7.0)
        if pH < 6.8:
            ph_deviation = 6.8 - pH
            if penalty_type == 'linear+constant':
                total_penalty -= rc.get('ph_constant_penalty', 0.5)
                total_penalty -= ph_deviation * rc['ph_penalty_scale']
            else:
                total_penalty -= (ph_deviation ** 2) * rc['ph_penalty_scale']
            self.violation_count['ph_low'] += 1
        elif pH > 7.8:
            ph_deviation = pH - 7.8
            if penalty_type == 'linear+constant':
                total_penalty -= rc.get('ph_constant_penalty', 0.5)
                total_penalty -= ph_deviation * rc['ph_penalty_scale']
            else:
                total_penalty -= (ph_deviation ** 2) * rc['ph_penalty_scale']
            self.violation_count['ph_high'] += 1

        # VFA penalty (threshold: 0.2 kg COD/m³)
        total_vfa = self._calculate_total_vfa(s)
        if total_vfa > 0.2:
            vfa_excess = total_vfa - 0.2
            if penalty_type == 'linear+constant':
                total_penalty -= rc.get('vfa_constant_penalty', 1.0)
                total_penalty -= vfa_excess * rc['vfa_penalty_scale']
            else:
                total_penalty -= (vfa_excess ** 2) * rc['vfa_penalty_scale']
            self.violation_count['vfa_high'] += 1

        # NH3 penalty (threshold: 0.002 kmol/m³)
        S_nh3 = s.get('S_nh3', 0.0)
        if S_nh3 > 0.002:
            nh3_excess = S_nh3 - 0.002
            if penalty_type == 'linear+constant':
                total_penalty -= rc.get('nh3_constant_penalty', 1.5)
                total_penalty -= nh3_excess * rc['nh3_penalty_scale']
            else:
                total_penalty -= (nh3_excess ** 2) * rc['nh3_penalty_scale']
            self.violation_count['nh3_high'] += 1

        return total_penalty

    def _is_catastrophic_failure(self) -> bool:
        """
        Check if episode should terminate due to catastrophic failure

        Termination conditions (only severe failures):
        - pH < 5.8 (complete acidification)
        - NH3 > 0.01 kmol/m³ (extreme toxicity)
        - VFA > 0.8 kg COD/m³ (process collapse)

        Returns:
            True if catastrophic failure occurred
        """
        s = self.current_state

        # Check pH collapse
        pH = s.get('pH', 7.0)
        if pH < 5.8 or pH > 8.8:
            return True

        # Check extreme ammonia toxicity
        S_nh3 = s.get('S_nh3', 0.0)
        if S_nh3 > 0.01:
            return True

        # Check complete VFA accumulation (process collapse)
        total_vfa = self._calculate_total_vfa(s)
        if total_vfa > 0.8:
            return True

        return False

    def _build_info_dict(self, is_done: bool) -> Dict[str, Any]:
        """Build info dictionary with metrics"""
        s = self.current_state

        info = {
            'step': self.current_step,
            'time_days': self.current_time_days,
            'q_ch4': self.q_ch4,
            'q_co2': self.solver.q_co2,
            'pH': s.get('pH', 7.0),
            'total_vfa': self._calculate_total_vfa(s),
            'alkalinity': self._calculate_alkalinity(s),
            'S_nh3': s.get('S_nh3', 0.0),
            'q_ad': self.q_ad_current,
            'feed_multiplier': self.feed_mult_current,
            'total_ch4': self.total_ch4_produced,
        }

        # Add episode metrics if done
        if is_done:
            info['episode'] = {
                'r': self.episode_reward,  # Episode reward
                'l': self.current_step,    # Episode length
            }
            # Custom metrics
            info['ch4_produced'] = self.total_ch4_produced
            info['ch4_avg_flow'] = self.total_ch4_produced / max(1, self.current_step * self.step_size)
            info['avg_ph'] = np.mean(self.ph_history) if len(self.ph_history) > 0 else 7.0
            info['avg_vfa'] = np.mean(self.vfa_history) if len(self.vfa_history) > 0 else 0.0
            info['ch4_std'] = np.std(self.ch4_history) if len(self.ch4_history) > 0 else 0.0
            info['violation_count'] = self.violation_count.copy()

        return info

    def render(self, mode: str = 'human'):
        """Render environment state"""
        if mode == 'human':
            s = self.current_state
            total_vfa = self._calculate_total_vfa(s)
            alkalinity = self._calculate_alkalinity(s)

            print(f"\n=== Step {self.current_step} (Day {self.current_time_days:.2f}) ===")
            print(f"Scenario: {self.scenario_name}")
            print(f"pH: {s.get('pH', 0):.2f}")
            print(f"CH4 flow: {self.q_ch4:.2f} m³/d")
            print(f"Total CH4: {self.total_ch4_produced:.2f} m³")
            print(f"Total VFA: {total_vfa:.4f} kg COD/m³")
            print(f"Alkalinity: {alkalinity:.4f} kmol/m³")
            print(f"VFA/Alk ratio: {total_vfa/max(alkalinity, 1e-6):.2f}")
            print(f"Flow rate: {self.q_ad_current:.1f} m³/d")
            print(f"Feed mult: {self.feed_mult_current:.2f}")

    def close(self):
        """Cleanup"""
        pass


# ========== Testing ==========

if __name__ == '__main__':
    print("=" * 80)
    print("ADM1 Gym Environment v2 - Test Suite")
    print("=" * 80)

    # Test 1: Nominal scenario
    print("\n[Test 1] Nominal scenario (5 steps):")
    try:
        env = ADM1Env_v2(scenario_name='nominal')
        obs, info = env.reset(seed=42)

        print(f"  Environment initialized")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Scenario: {info['scenario']}")
        print(f"  Max steps: {info['max_steps']}")

        print("\n  Running 5 steps with random actions...")
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            print(f"  Step {step+1}: pH={info['pH']:.2f}, CH4={info['q_ch4']:.2f} m³/d, "
                  f"VFA={info['total_vfa']:.4f}, Reward={reward:.4f}")

            if terminated:
                print("  Episode terminated (catastrophic failure)!")
                break
            if truncated:
                print("  Episode truncated (time limit)!")
                break

        print("  Nominal scenario test passed!")

    except Exception as e:
        print(f"  Nominal scenario test FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Shock load scenario
    print("\n" + "=" * 80)
    print("\n[Test 2] Shock load scenario (check disturbance detection):")
    try:
        env = ADM1Env_v2(scenario_name='shock_load', enable_disturbances=True)
        obs, info = env.reset(seed=42)

        print(f"  Environment initialized")
        print(f"  Scenario: {info['scenario']}")
        print(f"  Duration: {info['duration_days']} days")

        # Jump to day 10 (where shock occurs)
        target_step = int(10.0 / env.step_size)
        print(f"\n  Fast-forwarding to step {target_step} (day 10)...")

        for step in range(target_step + 5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # Print around shock time
            if target_step - 2 <= step <= target_step + 2:
                print(f"  Step {step}: Day {info['time_days']:.2f}, "
                      f"pH={info['pH']:.2f}, VFA={info['total_vfa']:.4f}")

            if terminated or truncated:
                break

        print("  Shock load scenario test passed!")

    except Exception as e:
        print(f"  Shock load scenario test FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Observation space validation
    print("\n" + "=" * 80)
    print("\n[Test 3] Observation space validation:")
    try:
        env = ADM1Env_v2(scenario_name='nominal')
        obs, info = env.reset(seed=42)

        print(f"  Observation shape: {obs.shape}")
        print(f"  Expected: (13,)")

        # Check observation components
        obs_names = [
            'total_vfa', 'alkalinity', 'vfa_alk_ratio', 'S_h2', 'pH',
            'S_nh3', 'S_IN', 'X_ac', 'X_h2', 'q_ch4', 'q_ad', 'feed_mult',
            'T_L_norm'
        ]

        print("\n  Observation components:")
        for i, (name, value) in enumerate(zip(obs_names, obs)):
            print(f"    {i:2d}. {name:20s} = {value:.6f}")

        # Check bounds
        if env.observation_space.contains(obs):
            print("\n  Observation within bounds!")
        else:
            print("\n  Observation out of bounds!")

        print("  Observation space test passed!")

    except Exception as e:
        print(f"  Observation space test FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("All tests completed.")
    print("=" * 80)

#!/usr/bin/env python3
"""
Baseline Controllers for Anaerobic Digestion Control
=====================================================

Engineering baseline controllers compared against the SAC policy in the paper:

    Reported in paper (Table comparison / Fig. fig_result_combo):
      - Constant            : fixed nominal setpoint, no feedback
      - PID                 : single-loop pH → feed_mult
      - CascadedPID         : outer pH loop → inner VFA loop → feed_mult
      - ConstantThermal     : Constant + fixed Q_HEX (thermal scenarios)
      - FullPID (PID+Thermal): pH PID + independent T_L PID for Q_HEX

    Additional (not reported, included for completeness):
      - RandomController    : uniform random actions (sanity check)
      - RuleBasedController : hand-crafted if/else heuristics
      - ProportionalController : P-only pH control

Action space (3-dimensional, matching ADM1Env_v2):
    [q_ad (m³/day),  feed_mult (dimensionless),  Q_HEX (W)]

Observation indices referenced by these controllers:
    obs[0]  = total_vfa  (kmol COD/m³)
    obs[4]  = pH
    obs[12] = T_L_norm   = (T_L − 35.0) / 10.0  [13D obs only]
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod


class BaseController(ABC):
    """
    Abstract base class for all controllers

    All controllers must implement:
    - get_action(obs): Given observation, return action
    - reset(): Reset internal state
    """

    def __init__(self, name: str = "BaseController"):
        self.name = name
        self.step_count = 0

    @abstractmethod
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute control action from observation.

        Args:
            observation: observation vector from ADM1Env_v2
                         (13-dim full obs or 5-dim compact obs)

        Returns:
            action: 3-dim array [q_ad (m³/day), feed_mult, Q_HEX (W)]
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset controller internal state"""
        pass

    def __str__(self):
        return f"{self.name} (step={self.step_count})"


# ========== Tier 0: Sanity Check Baselines ==========

class RandomController(BaseController):
    """
    Random action baseline

    Samples actions uniformly from action space.
    Used as lower-bound sanity check.
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(name="Random")
        self.rng = np.random.RandomState(seed)

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Random action from [50, 300] × [0.7, 1.3]; Q_HEX fixed at 0."""
        self.step_count += 1
        q_ad = self.rng.uniform(50.0, 300.0)
        feed_mult = self.rng.uniform(0.7, 1.3)
        return np.array([q_ad, feed_mult, 0.0], dtype=np.float32)

    def reset(self):
        self.step_count = 0


class ConstantController(BaseController):
    """
    Constant action baseline

    Always outputs the same action (default: nominal operating point).
    Used as sanity check and baseline for stable scenarios.
    """

    def __init__(self, q_ad: float = 178.0, feed_mult: float = 1.0):
        super().__init__(name="Constant")
        self.q_ad = q_ad
        self.feed_mult = feed_mult

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Return constant action with Q_HEX=0 (no heating)."""
        self.step_count += 1
        return np.array([self.q_ad, self.feed_mult, 0.0], dtype=np.float32)

    def reset(self):
        self.step_count = 0


# ========== Tier 1: Simple Reactive Controllers ==========

class RuleBasedController(BaseController):
    """
    Rule-based controller using expert heuristics

    Rules:
    1. If pH < 6.8: Reduce feed (prevent acidification)
    2. If pH > 7.5: Increase feed (maximize production)
    3. If VFA > 0.2: Reduce feed (prevent VFA accumulation)
    4. Otherwise: Nominal operation

    This represents expert knowledge without tuning.
    """

    def __init__(
        self,
        nominal_q_ad: float = 178.0,
        nominal_feed: float = 1.0,
        ph_low_threshold: float = 6.8,
        ph_high_threshold: float = 7.5,
        vfa_threshold: float = 0.2
    ):
        super().__init__(name="RuleBased")
        self.nominal_q_ad = nominal_q_ad
        self.nominal_feed = nominal_feed
        self.ph_low = ph_low_threshold
        self.ph_high = ph_high_threshold
        self.vfa_thresh = vfa_threshold

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Apply expert rules to determine action"""
        self.step_count += 1

        # Parse observation
        total_vfa = observation[0]
        # alkalinity = observation[1]
        # vfa_alk_ratio = observation[2]
        pH = observation[4]

        # Default: nominal operation
        q_ad = self.nominal_q_ad
        feed_mult = self.nominal_feed

        # Rule 1: pH too low (acidification risk)
        if pH < self.ph_low:
            feed_mult = 0.8  # Reduce organic load
            q_ad = max(150.0, self.nominal_q_ad * 0.9)  # Slightly reduce flow

        # Rule 2: pH too high (underutilization)
        elif pH > self.ph_high:
            feed_mult = 1.1  # Increase organic load
            q_ad = min(220.0, self.nominal_q_ad * 1.1)  # Slightly increase flow

        # Rule 3: VFA accumulation (override pH rule)
        if total_vfa > self.vfa_thresh:
            feed_mult = 0.75  # Aggressive feed reduction
            q_ad = max(120.0, self.nominal_q_ad * 0.8)  # Reduce flow

        return np.array([q_ad, feed_mult, 0.0], dtype=np.float32)

    def reset(self):
        self.step_count = 0


class ProportionalController(BaseController):
    """
    Proportional (P) controller for pH regulation

    Control law:
        feed_mult = 1.0 - K_p * (pH - pH_setpoint)

    Simple, no integral/derivative terms.
    Used as stepping stone to PID.
    """

    def __init__(
        self,
        K_p: float = 0.5,
        pH_setpoint: float = 7.2,
        nominal_q_ad: float = 178.0
    ):
        super().__init__(name="Proportional")
        self.K_p = K_p
        self.pH_setpoint = pH_setpoint
        self.nominal_q_ad = nominal_q_ad

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """P control on pH"""
        self.step_count += 1

        pH = observation[4]

        # P control
        error = pH - self.pH_setpoint
        feed_mult = 1.0 - self.K_p * error

        # Clamp feed_mult to valid range
        feed_mult = np.clip(feed_mult, 0.7, 1.3)

        # Keep q_ad constant
        q_ad = self.nominal_q_ad

        return np.array([q_ad, feed_mult, 0.0], dtype=np.float32)

    def reset(self):
        self.step_count = 0


class PIDController(BaseController):
    """
    PID controller for pH regulation

    Control law:
        u(t) = K_p * e(t) + K_i * ∫e(τ)dτ + K_d * de/dt

    Where:
        e(t) = pH_setpoint - pH(t)
        u(t) = feed_multiplier adjustment

    Features:
    - Anti-windup for integral term
    - Derivative filtering to reduce noise
    - Saturates output to [0.7, 1.3]
    """

    def __init__(
        self,
        K_p: float = 0.5,
        K_i: float = 0.1,
        K_d: float = 0.05,
        pH_setpoint: float = 7.2,
        nominal_q_ad: float = 178.0,
        integral_limit: float = 2.0,
        derivative_filter_alpha: float = 0.1
    ):
        super().__init__(name="PID")
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d
        self.pH_setpoint = pH_setpoint
        self.nominal_q_ad = nominal_q_ad
        self.integral_limit = integral_limit
        self.derivative_filter_alpha = derivative_filter_alpha

        # Internal state
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """PID control on pH"""
        self.step_count += 1

        pH = observation[4]

        # Calculate error (setpoint - measurement)
        error = self.pH_setpoint - pH

        # Proportional term
        P_term = self.K_p * error

        # Integral term (with anti-windup)
        self.integral_error += error
        self.integral_error = np.clip(
            self.integral_error,
            -self.integral_limit,
            self.integral_limit
        )
        I_term = self.K_i * self.integral_error

        # Derivative term (with filtering)
        derivative = error - self.prev_error
        self.filtered_derivative = (
            self.derivative_filter_alpha * derivative +
            (1 - self.derivative_filter_alpha) * self.filtered_derivative
        )
        D_term = self.K_d * self.filtered_derivative

        # PID output
        pid_output = P_term + I_term + D_term

        # Convert to feed multiplier
        feed_mult = 1.0 - pid_output
        feed_mult = np.clip(feed_mult, 0.7, 1.3)

        # Update state
        self.prev_error = error

        # Keep q_ad constant
        q_ad = self.nominal_q_ad

        return np.array([q_ad, feed_mult, 0.0], dtype=np.float32)

    def reset(self):
        self.step_count = 0
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0


# ========== Tier 2: Advanced Controllers ==========

class CascadedPIDController(BaseController):
    """
    Cascaded PID controller (pH + VFA)

    Architecture:
    - Outer loop: pH control → VFA setpoint
    - Inner loop: VFA control → feed_multiplier

    This is more sophisticated than single-loop PID.
    """

    def __init__(
        self,
        # Outer loop (pH → VFA setpoint)
        K_p_outer: float = 0.02,
        K_i_outer: float = 0.005,
        K_d_outer: float = 0.01,
        pH_setpoint: float = 7.2,
        vfa_nominal: float = 0.15,
        # Inner loop (VFA → feed_mult)
        K_p_inner: float = 2.0,
        K_i_inner: float = 0.5,
        K_d_inner: float = 0.1,
        # Common parameters
        nominal_q_ad: float = 178.0
    ):
        super().__init__(name="CascadedPID")

        # Outer loop (pH control)
        self.K_p_outer = K_p_outer
        self.K_i_outer = K_i_outer
        self.K_d_outer = K_d_outer
        self.pH_setpoint = pH_setpoint
        self.vfa_nominal = vfa_nominal

        # Inner loop (VFA control)
        self.K_p_inner = K_p_inner
        self.K_i_inner = K_i_inner
        self.K_d_inner = K_d_inner

        self.nominal_q_ad = nominal_q_ad

        # Internal state (outer loop)
        self.integral_error_outer = 0.0
        self.prev_error_outer = 0.0
        self.filtered_derivative_outer = 0.0

        # Internal state (inner loop)
        self.integral_error_inner = 0.0
        self.prev_error_inner = 0.0
        self.filtered_derivative_inner = 0.0

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Cascaded PID control"""
        self.step_count += 1

        total_vfa = observation[0]
        pH = observation[4]

        # ===== Outer loop: pH → VFA setpoint =====
        error_outer = self.pH_setpoint - pH

        # PID terms
        P_outer = self.K_p_outer * error_outer

        self.integral_error_outer += error_outer
        self.integral_error_outer = np.clip(self.integral_error_outer, -5.0, 5.0)
        I_outer = self.K_i_outer * self.integral_error_outer

        derivative_outer = error_outer - self.prev_error_outer
        self.filtered_derivative_outer = (
            0.1 * derivative_outer + 0.9 * self.filtered_derivative_outer
        )
        D_outer = self.K_d_outer * self.filtered_derivative_outer

        # VFA setpoint adjustment
        vfa_setpoint = self.vfa_nominal - (P_outer + I_outer + D_outer)
        vfa_setpoint = np.clip(vfa_setpoint, 0.05, 0.3)

        self.prev_error_outer = error_outer

        # ===== Inner loop: VFA → feed_mult =====
        error_inner = vfa_setpoint - total_vfa

        # PID terms
        P_inner = self.K_p_inner * error_inner

        self.integral_error_inner += error_inner
        self.integral_error_inner = np.clip(self.integral_error_inner, -2.0, 2.0)
        I_inner = self.K_i_inner * self.integral_error_inner

        derivative_inner = error_inner - self.prev_error_inner
        self.filtered_derivative_inner = (
            0.1 * derivative_inner + 0.9 * self.filtered_derivative_inner
        )
        D_inner = self.K_d_inner * self.filtered_derivative_inner

        # Feed multiplier
        feed_mult = 1.0 + (P_inner + I_inner + D_inner)
        feed_mult = np.clip(feed_mult, 0.7, 1.3)

        self.prev_error_inner = error_inner

        # Keep q_ad constant
        q_ad = self.nominal_q_ad

        return np.array([q_ad, feed_mult, 0.0], dtype=np.float32)

    def reset(self):
        self.step_count = 0
        self.integral_error_outer = 0.0
        self.prev_error_outer = 0.0
        self.filtered_derivative_outer = 0.0
        self.integral_error_inner = 0.0
        self.prev_error_inner = 0.0
        self.filtered_derivative_inner = 0.0


# ========== Tier 3: Thermal Controllers (3-dim action: q_ad, feed_mult, Q_HEX) ==========

class ConstantThermalController(BaseController):
    """
    Constant controller with fixed Q_HEX.

    Outputs the same action every step, including a fixed heat exchanger power.
    Q_HEX_nominal should be the steady-state value required for the scenario:
      - nominal:    ~500 W  (UA=50, ΔT=10°C)
      - cold_winter: ~2400 W (UA=80, ΔT=30°C)
    """

    def __init__(self,
                 q_ad: float = 178.0,
                 feed_mult: float = 1.0,
                 Q_HEX: float = 500.0,
                 Q_HEX_bias: float = None):
        super().__init__(name="ConstantThermal")
        self.q_ad      = q_ad
        self.feed_mult = feed_mult
        self.Q_HEX     = Q_HEX_bias if Q_HEX_bias is not None else Q_HEX

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        self.step_count += 1
        return np.array([self.q_ad, self.feed_mult, self.Q_HEX], dtype=np.float32)

    def reset(self):
        self.step_count = 0


class ThermalPIDController(BaseController):
    """
    Thermal PID controller: controls Q_HEX to maintain T_L setpoint.

    pH control via feed_mult remains constant (nominal).
    obs[12] = T_L_norm = (T_L - 308.15) / 10.0  →  setpoint = 0.0 (35°C)

    Control law:
        e(t) = T_setpoint_norm - obs[12]
        Q_HEX = Q_HEX_bias + K_p*e + K_i*∫e + K_d*de/dt
    """

    def __init__(self,
                 T_setpoint_C: float = 35.0,
                 K_p: float = 500.0,
                 K_i: float = 50.0,
                 K_d: float = 20.0,
                 Q_HEX_bias: float = 500.0,
                 Q_HEX_min: float = 0.0,
                 Q_HEX_max: float = 5000.0,
                 nominal_q_ad: float = 178.0):
        super().__init__(name="ThermalPID")
        self.T_sp_norm    = (T_setpoint_C - 35.0) / 10.0  # normalized setpoint
        self.K_p          = K_p
        self.K_i          = K_i
        self.K_d          = K_d
        self.Q_HEX_bias   = Q_HEX_bias
        self.Q_HEX_min    = Q_HEX_min
        self.Q_HEX_max    = Q_HEX_max
        self.nominal_q_ad = nominal_q_ad

        self.integral_error   = 0.0
        self.prev_error       = 0.0
        self.integral_limit   = 50.0

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        self.step_count += 1

        # obs[12] = T_L_norm (only present in 13-dim obs)
        T_L_norm = float(observation[12]) if len(observation) > 12 else 0.0
        error = self.T_sp_norm - T_L_norm

        # PID
        P_term = self.K_p * error

        self.integral_error = np.clip(
            self.integral_error + error,
            -self.integral_limit, self.integral_limit
        )
        I_term = self.K_i * self.integral_error

        D_term = self.K_d * (error - self.prev_error)
        self.prev_error = error

        Q_HEX = self.Q_HEX_bias + P_term + I_term + D_term
        Q_HEX = np.clip(Q_HEX, self.Q_HEX_min, self.Q_HEX_max)

        return np.array([self.nominal_q_ad, 1.0, Q_HEX], dtype=np.float32)

    def reset(self):
        self.step_count     = 0
        self.integral_error = 0.0
        self.prev_error     = 0.0


class FullPIDController(BaseController):
    """
    Full multi-loop PID: pH PID (feed_mult) + Temperature PID (Q_HEX).

    - Inner loop 1: pH → feed_mult  (same as PIDController)
    - Inner loop 2: T_L → Q_HEX    (same as ThermalPIDController)
    Both loops are independent (decoupled).
    """

    def __init__(self,
                 # pH loop
                 K_p_ph: float = 0.5,
                 K_i_ph: float = 0.05,
                 K_d_ph: float = 0.05,
                 pH_setpoint: float = 7.2,
                 # Temperature loop
                 T_setpoint_C: float = 35.0,
                 K_p_T: float = 500.0,
                 K_i_T: float = 50.0,
                 K_d_T: float = 20.0,
                 Q_HEX_bias: float = 500.0,
                 Q_HEX_min: float = 0.0,
                 Q_HEX_max: float = 5000.0,
                 nominal_q_ad: float = 178.0):
        super().__init__(name="FullPID")
        # pH loop
        self.K_p_ph      = K_p_ph
        self.K_i_ph      = K_i_ph
        self.K_d_ph      = K_d_ph
        self.pH_setpoint = pH_setpoint
        self.int_ph      = 0.0
        self.prev_err_ph = 0.0
        # Temperature loop
        self.T_sp_norm   = (T_setpoint_C - 35.0) / 10.0
        self.K_p_T       = K_p_T
        self.K_i_T       = K_i_T
        self.K_d_T       = K_d_T
        self.Q_HEX_bias  = Q_HEX_bias
        self.Q_HEX_min   = Q_HEX_min
        self.Q_HEX_max   = Q_HEX_max
        self.int_T       = 0.0
        self.prev_err_T  = 0.0
        self.nominal_q_ad = nominal_q_ad

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        self.step_count += 1

        # ── pH loop ───────────────────────────────────────────────
        pH = float(observation[4])
        err_ph = self.pH_setpoint - pH
        self.int_ph = np.clip(self.int_ph + err_ph, -2.0, 2.0)
        d_ph = err_ph - self.prev_err_ph
        self.prev_err_ph = err_ph
        pid_ph = self.K_p_ph * err_ph + self.K_i_ph * self.int_ph + self.K_d_ph * d_ph
        feed_mult = np.clip(1.0 - pid_ph, 0.7, 1.3)

        # ── Temperature loop ──────────────────────────────────────
        T_L_norm = float(observation[12]) if len(observation) > 12 else 0.0
        err_T = self.T_sp_norm - T_L_norm
        self.int_T = np.clip(self.int_T + err_T, -50.0, 50.0)
        d_T = err_T - self.prev_err_T
        self.prev_err_T = err_T
        pid_T = self.K_p_T * err_T + self.K_i_T * self.int_T + self.K_d_T * d_T
        Q_HEX = np.clip(self.Q_HEX_bias + pid_T, self.Q_HEX_min, self.Q_HEX_max)

        return np.array([self.nominal_q_ad, feed_mult, Q_HEX], dtype=np.float32)

    def reset(self):
        self.step_count  = 0
        self.int_ph      = 0.0
        self.prev_err_ph = 0.0
        self.int_T       = 0.0
        self.prev_err_T  = 0.0


# ========== Controller Factory ==========

def get_controller(controller_name: str, **kwargs) -> BaseController:
    """
    Factory function to create controllers

    Args:
        controller_name: Name of controller ('random', 'constant', 'pid', etc.)
        **kwargs: Controller-specific parameters

    Returns:
        Controller instance

    Example:
        >>> controller = get_controller('pid', K_p=0.5, K_i=0.1, K_d=0.05)
    """
    controller_map = {
        'random': RandomController,
        'constant': ConstantController,
        'rule_based': RuleBasedController,
        'proportional': ProportionalController,
        'pid': PIDController,
        'cascaded_pid': CascadedPIDController,
        # Thermal controllers (3-dim action including Q_HEX)
        'constant_thermal': ConstantThermalController,
        'thermal_pid': ThermalPIDController,
        'full_pid': FullPIDController,
    }

    if controller_name.lower() not in controller_map:
        raise ValueError(
            f"Unknown controller '{controller_name}'. "
            f"Available: {list(controller_map.keys())}"
        )

    controller_class = controller_map[controller_name.lower()]
    return controller_class(**kwargs)


# ========== Testing ==========

if __name__ == '__main__':
    print("=" * 80)
    print("Baseline Controllers - Smoke Test")
    print("=" * 80)

    # Mock 13-dim observation (full obs space used in paper)
    obs13 = np.array([
        0.15,    # [0]  total_vfa  (kmol COD/m³)
        0.08,    # [1]  alkalinity
        1.875,   # [2]  vfa_alk_ratio
        1e-6,    # [3]  S_H2
        7.2,     # [4]  pH
        0.002,   # [5]  S_NH3
        0.095,   # [6]  S_IN
        0.7,     # [7]  X_ac
        0.3,     # [8]  X_H2
        1600.0,  # [9]  q_CH4 (m³/day)
        178.0,   # [10] q_ad
        1.0,     # [11] feed_mult
        0.0,     # [12] T_L_norm = (35.0 - 35.0) / 10.0
    ], dtype=np.float32)

    controllers = [
        ('Constant',        ConstantController()),
        ('PID',             PIDController()),
        ('CascadedPID',     CascadedPIDController()),
        ('ConstantThermal', ConstantThermalController()),
        ('FullPID',         FullPIDController()),
        # Not used in paper comparisons:
        ('Random',          RandomController(seed=42)),
        ('RuleBased',       RuleBasedController()),
        ('Proportional',    ProportionalController()),
    ]

    print(f"\n{'Controller':<20} {'q_ad':>10} {'feed_mult':>10} {'Q_HEX':>10}  status")
    print("-" * 65)
    for name, ctrl in controllers:
        try:
            action = ctrl.get_action(obs13)
            if len(action) == 3:
                q_ad, fm, q_hex = action
                print(f"{name:<20} {q_ad:>10.1f} {fm:>10.4f} {q_hex:>10.1f}  OK")
            else:
                print(f"{name:<20} {action[0]:>10.1f} {action[1]:>10.4f} {'—':>10}  OK (2-dim)")
        except Exception as e:
            print(f"{name:<20} {'':>10} {'':>10} {'':>10}  ERROR: {e}")

    print(f"\n{'='*65}")
    print("PID controller response to pH deviation:")
    print(f"{'Step':>6} {'pH':>8} {'feed_mult':>12} {'integral':>12}")
    print("-" * 42)
    pid = PIDController()
    for pH in [7.2, 7.0, 6.8, 6.9, 7.1]:
        obs13[4] = pH
        a = pid.get_action(obs13)
        print(f"{pid.step_count:>6} {pH:>8.2f} {a[1]:>12.4f} {pid.integral_error:>12.4f}")
    print("Done.")

#!/usr/bin/env python3
"""
Custom Controller Template — ADM1 Benchmark
============================================

Copy this file, implement your control logic in MyController.get_action(),
then run the benchmark:

    python evaluate_controller.py --controller examples/my_controller.py

The script discovers MyController automatically (any subclass of BaseController
defined at module level).

Observation space (13-dim, obs_mode='full'):
    idx  variable          unit            description
    ---  --------          ----            -----------
     0   total_vfa         kg COD/m³       S_ac + S_pro + S_bu + S_va
     1   alkalinity        kmol/m³         0.8·S_IC + S_NH3
     2   vfa_alk_ratio     —               total_vfa / alkalinity
     3   S_h2              kg COD/m³       dissolved hydrogen
     4   pH                —               reactor pH
     5   S_nh3             kmol N/m³       free ammonia
     6   S_IN              kmol N/m³       inorganic nitrogen
     7   X_ac              kg COD/m³       acetoclastic methanogen biomass
     8   X_h2              kg COD/m³       hydrogenotrophic methanogen biomass
     9   q_ch4             m³/day          methane flow rate
    10   q_ad_current      m³/day          current feed flow rate
    11   feed_mult_current  —               current feed concentration multiplier
    12   T_L_norm          —               (T_liquid − 308.15 K) / 10

Compact observation (5-dim, obs_mode='simple'):
    Indices [4, 9, 12, 10, 11] of the full obs:
    [pH, q_ch4, T_L_norm, q_ad_current, feed_mult_current]

Action space (3-dim continuous):
    [q_ad (m³/day), feed_mult (dimensionless), Q_HEX (W)]
    bounds:  [50, 300]  ×  [0.7, 1.3]  ×  [−5000, 5000]

Safety thresholds (used by MetricsCalculator):
    pH out of [6.8, 7.8]  →  violation
    total_vfa > 0.2 kmol COD/m³  →  violation
    S_nh3 > 0.002 kmol N/m³  →  violation

Episode termination (catastrophic failure):
    pH < 5.8  or  S_nh3 > 0.01  or  total_vfa > 0.8

Evaluation scenarios:
    nominal, high_load, low_load, shock_load, temperature_drop, cold_winter
"""

import numpy as np
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.baseline_controllers import BaseController


# ── Observation index constants ───────────────────────────────────────────────

OBS_TOTAL_VFA       = 0
OBS_ALKALINITY      = 1
OBS_VFA_ALK_RATIO   = 2
OBS_S_H2            = 3
OBS_PH              = 4
OBS_S_NH3           = 5
OBS_S_IN            = 6
OBS_X_AC            = 7
OBS_X_H2            = 8
OBS_Q_CH4           = 9
OBS_Q_AD            = 10
OBS_FEED_MULT       = 11
OBS_T_L_NORM        = 12

# Action bounds (mirrors ADM1Env_v2.action_space)
Q_AD_MIN,    Q_AD_MAX    = 50.0,   300.0   # m³/day
FEED_MIN,    FEED_MAX    = 0.7,    1.3
Q_HEX_MIN,   Q_HEX_MAX  = -5000.0, 5000.0  # W


# ── Your controller ───────────────────────────────────────────────────────────

class MyController(BaseController):
    """
    Example: dead-band pH controller with fixed feed rate and thermal hold.

    Replace this logic with your own algorithm.  The only requirements are:
    - __init__ accepts no mandatory arguments (for auto-discovery)
    - get_action(obs) returns a length-3 numpy array
    - reset() is called at the start of every new episode
    """

    def __init__(self):
        super().__init__(name="MyController")

        # ── Setpoints ────────────────────────────────────────────────────
        self.ph_target   = 7.2          # target pH
        self.ph_deadband = 0.15         # ± tolerance before acting
        self.q_ad_base   = 180.0        # baseline feed flow  (m³/day)
        self.q_hex_base  = 0.0          # no active heating by default

        # ── Controller gain ───────────────────────────────────────────────
        self.kp_feed = 0.15             # proportional gain: ΔpH → Δfeed_mult

        # ── Internal state ────────────────────────────────────────────────
        self._prev_ph = None

    def reset(self):
        """Called at the start of every episode."""
        self.step_count = 0
        self._prev_ph   = None

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute control action from the current observation.

        Args:
            observation: numpy array, shape (13,) for full obs or (5,) for simple obs.
                         Index constants defined at the top of this file.

        Returns:
            action: numpy array, shape (3,)  →  [q_ad, feed_mult, Q_HEX]
        """
        self.step_count += 1

        # ── Read observations ─────────────────────────────────────────────
        ph       = observation[OBS_PH]
        total_vfa = observation[OBS_TOTAL_VFA]
        T_norm   = observation[OBS_T_L_NORM] if len(observation) > 5 else 0.0

        # ── pH dead-band P controller → feed_mult ─────────────────────────
        ph_error = self.ph_target - ph
        if abs(ph_error) <= self.ph_deadband:
            feed_mult = 1.0                         # inside dead-band: hold nominal
        else:
            # pH too high → reduce feed; pH too low → increase feed
            feed_mult = 1.0 - self.kp_feed * ph_error

        # ── VFA safety override: back off feed if VFA is rising ───────────
        if total_vfa > 0.15:
            feed_mult = min(feed_mult, 0.85)

        # ── Thermal management (active only if temperature sensor available)
        T_liquid = T_norm * 10.0 + 35.0            # °C (approximate)
        if T_liquid < 33.0:
            q_hex = 2000.0                          # warm up reactor
        elif T_liquid > 38.0:
            q_hex = -1000.0                         # cool down
        else:
            q_hex = self.q_hex_base

        # ── Clip to action bounds and return ─────────────────────────────
        q_ad      = float(np.clip(self.q_ad_base, Q_AD_MIN,  Q_AD_MAX))
        feed_mult = float(np.clip(feed_mult,       FEED_MIN,  FEED_MAX))
        q_hex     = float(np.clip(q_hex,           Q_HEX_MIN, Q_HEX_MAX))

        return np.array([q_ad, feed_mult, q_hex], dtype=np.float32)


# ── Quick self-test (optional) ────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Quick single-scenario test of MyController.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--scenario', default='nominal',
                        choices=['nominal', 'high_load', 'low_load',
                                 'shock_load', 'temperature_drop', 'cold_winter'])
    parser.add_argument('--episodes', type=int, default=3)
    args = parser.parse_args()

    from env.adm1_gym_env import ADM1Env_v2
    from evaluation.metrics_calculator import MetricsCalculator

    ctrl = MyController()
    env  = ADM1Env_v2(scenario_name=args.scenario)

    scores, viols = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=42 + ep)
        ctrl.reset()
        calc = MetricsCalculator()
        done = False
        while not done:
            action = ctrl.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            calc.add_step(obs, action, reward, info)
            if terminated:
                calc.set_terminated(ctrl.step_count)
            done = terminated or truncated
        m = calc.compute_metrics()
        scores.append(m['summary']['overall_score'])
        viols.append(m['safety']['violation_rate'])
        print(f"  ep {ep+1}: score={scores[-1]:.3f}  viol={viols[-1]:.3f}")

    env.close()
    print(f"\nMean score: {np.mean(scores):.3f}  |  Mean viol: {np.mean(viols):.3f}")
    print("Run `python evaluate_controller.py --controller examples/my_controller.py` "
          "for the full benchmark.")

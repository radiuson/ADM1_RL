#!/usr/bin/env python3
"""
SCADA Calibration Layer — ADM1_RL
===================================

Maps real-plant SCADA / LAB measurements to the simulator's observation space
so that a policy trained on BSM2 can be applied directly to a real digester.

Grounded in: Muscatine WRRF dataset (Hunter & Schroer 2023, n=861 lab days).

Observation layout (5-dim SCADA obs):
    obs[0]  vfa_alk_ratio   ← Dig1-FOS-TAC  (daily LAB)
    obs[1]  pH              ← Dig1-pH        (daily LAB, same units — no conversion)
    obs[2]  q_ch4           ← biogas × 0.65  (SCADA real-time, needs volume scale)
    obs[3]  q_ad_current    ← Q-TWAS_GPM     (SCADA real-time, needs flow scale)
    obs[4]  feed_mult       ← relative input, dimensionless — no conversion needed

Calibration design
------------------
FOS/TAC ↔ vfa_alk_ratio: linear, two-anchor method.

    Anchor 1 (nominal):  real FOS/TAC = 0.250  ↔  sim ratio = 1.52
    Anchor 2 (alarm):    real FOS/TAC = 0.400  ↔  sim ratio = 2.00
        Derived from: Muscatine p90 = 0.370, p95 = 0.440; literature alarm = 0.40
        Simulator alarm threshold: VFA > 0.30 kg COD/m³ → vfa_alk_ratio ≈ 2.00

    Linear map:  sim_ratio = A × fos_tac + B
        A = (2.00 − 1.52) / (0.40 − 0.25) = 3.200
        B = 1.52 − 3.200 × 0.25           = 0.720

    Verified:
        f(0.250) = 1.520  (nominal)
        f(0.400) = 2.000  (alarm)
        f(0.930) = 3.696  (Muscatine Sep-2022 crisis max)

q_ch4 / q_ad: z-score normalised then rescaled to simulator operating window.
    Requires plant-specific statistics (mean and std) — see PlantProfile.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── FOS/TAC ↔ vfa_alk_ratio calibration ──────────────────────────────────────

# Calibration anchors (derived from Muscatine data + BSM2 steady-state)
_REAL_NOMINAL  = 0.250   # Muscatine mean FOS/TAC
_REAL_ALARM    = 0.400   # Nordmann alarm, Muscatine p90=0.37, p95=0.44
_SIM_NOMINAL   = 1.520   # BSM2 vfa_alk_ratio at nominal (post bug-fix)
_SIM_ALARM     = 2.000   # BSM2 safety penalty threshold

_A = (_SIM_ALARM  - _SIM_NOMINAL) / (_REAL_ALARM - _REAL_NOMINAL)  # 3.200
_B = _SIM_NOMINAL - _A * _REAL_NOMINAL                              # 0.720


def fos_tac_to_sim(fos_tac: float) -> float:
    """
    Map real plant FOS/TAC (dimensionless, Nordmann) → simulator vfa_alk_ratio.

    Args:
        fos_tac: Real FOS/TAC measurement (e.g. 0.25 at nominal, 0.40+ alarm)

    Returns:
        Equivalent simulator obs[0] value (e.g. 1.52 at nominal, 2.00+ alarm)
    """
    return float(_A * fos_tac + _B)


def sim_to_fos_tac(sim_ratio: float) -> float:
    """
    Inverse map: simulator vfa_alk_ratio → real plant FOS/TAC.

    Useful for interpreting simulator results in terms readable to plant operators.
    Clipped to [0.11, 5.0] — the Muscatine observed minimum is 0.11; below the
    calibration intercept the linear extrapolation returns negatives (not physical).
    """
    return float(np.clip((sim_ratio - _B) / _A, 0.11, 5.0))


# ── Plant profile for flow / gas calibration ─────────────────────────────────

@dataclass
class PlantProfile:
    """
    Plant-specific statistics for z-score calibration of q_ch4 and q_ad.

    Compute from your plant's historical data:
        mean_q_ch4 = df['your_biogas_col'].mean() × ch4_fraction
        std_q_ch4  = df['your_biogas_col'].std()  × ch4_fraction

    Muscatine defaults (from Hunter & Schroer 2023, LAB/LABS-raw.csv):
        Biogas mean ≈ 103.5  (units depend on SCADA — verify before deploying)
        Assumed CH4 fraction ≈ 0.65
    """
    # Real plant statistics (fill with your own data)
    mean_q_ch4:  float = 103.5 * 0.65   # m³/d or local unit — update to match yours
    std_q_ch4:   float = 41.5  * 0.65

    mean_q_ad:   float = 178.0           # nominal feed flow (same unit as action space)
    std_q_ad:    float = 30.0

    # Simulator operating window
    sim_mean_q_ch4: float = 1556.0       # BSM2 nominal CH4 flow (m³/d)
    sim_std_q_ch4:  float = 400.0
    sim_mean_q_ad:  float = 178.0
    sim_std_q_ad:   float = 50.0

    # CH4 fraction of biogas (if your sensor measures total biogas)
    ch4_fraction: float = 0.65

    def q_ch4_to_sim(self, q_ch4_real: float) -> float:
        """Real biogas/CH4 flow → simulator q_ch4 (obs[2]).
        Note: clips to [0, 4000] — the obs-space bound of 600 in ADM1Env_Std
        is incorrect (nominal BSM2 value is ~1556 m³/d); use raw obs.
        """
        q_ch4_real = q_ch4_real * self.ch4_fraction if q_ch4_real > 10 else q_ch4_real
        z = (q_ch4_real - self.mean_q_ch4) / max(self.std_q_ch4, 1e-6)
        return float(np.clip(z * self.sim_std_q_ch4 + self.sim_mean_q_ch4, 0.0, 4000.0))

    def q_ad_to_sim(self, q_ad_real: float) -> float:
        """Real feed flow → simulator q_ad (obs[3])."""
        z = (q_ad_real - self.mean_q_ad) / max(self.std_q_ad, 1e-6)
        return float(np.clip(z * self.sim_std_q_ad + self.sim_mean_q_ad, 50.0, 300.0))


# ── Main calibration entry point ──────────────────────────────────────────────

class SCADACalibration:
    """
    Converts real SCADA / LAB measurements to the 5-dim simulator obs vector.

    Usage (inference on real plant):
        cal = SCADACalibration(PlantProfile(...))

        obs = cal.to_sim_obs(
            fos_tac   = 0.28,   # Dig1-FOS-TAC from daily LAB
            ph        = 7.21,   # Dig1-pH
            q_ch4     = 98.0,   # biogas flow (total, SCADA)
            q_ad      = 185.0,  # feed flow (SCADA)
            feed_mult = 1.05,   # relative feed setting
        )
        action = policy.predict(obs, deterministic=True)
    """

    def __init__(self, plant: Optional[PlantProfile] = None):
        self.plant = plant or PlantProfile()

    def to_sim_obs(
        self,
        fos_tac:   float,
        ph:        float,
        q_ch4:     float,
        q_ad:      float,
        feed_mult: float = 1.0,
    ) -> np.ndarray:
        """
        Convert real measurements to 5-dim simulator obs.

        Args:
            fos_tac:   Real FOS/TAC ratio (Nordmann, dimensionless)
            ph:        Digester pH
            q_ch4:     Biogas or CH4 flow (same unit as plant profile)
            q_ad:      Feed (sludge) flow rate
            feed_mult: Relative feed multiplier (1.0 = nominal; dimensionless)

        Returns:
            5-dim np.float32 array matching ADM1Env_Std SCADA obs:
            [vfa_alk_ratio, pH, q_ch4_sim, q_ad_sim, feed_mult]
        """
        return np.array([
            fos_tac_to_sim(fos_tac),
            float(np.clip(ph, 5.5, 8.5)),
            self.plant.q_ch4_to_sim(q_ch4),
            self.plant.q_ad_to_sim(q_ad),
            float(np.clip(feed_mult, 0.7, 1.3)),
        ], dtype=np.float32)

    def action_to_real(self, action: np.ndarray):
        """
        Convert simulator action [q_ad_sim, feed_mult] back to real plant setpoints.

        Returns:
            q_ad_real:   Feed flow in real plant units
            feed_mult:   Feed multiplier (already dimensionless, no conversion)
        """
        q_ad_sim, feed_mult = float(action[0]), float(action[1])
        z = (q_ad_sim - self.plant.sim_mean_q_ad) / max(self.plant.sim_std_q_ad, 1e-6)
        q_ad_real = z * self.plant.std_q_ad + self.plant.mean_q_ad
        return float(q_ad_real), float(np.clip(feed_mult, 0.7, 1.3))

    def describe(self):
        """Print calibration summary."""
        print("=== SCADA Calibration Summary ===")
        print(f"  FOS/TAC → vfa_alk_ratio:  ratio = {_A:.3f} × FOS/TAC + {_B:.3f}")
        print(f"    f(0.250) = {fos_tac_to_sim(0.250):.3f}  (nominal)")
        print(f"    f(0.400) = {fos_tac_to_sim(0.400):.3f}  (alarm)")
        print(f"    f(0.930) = {fos_tac_to_sim(0.930):.3f}  (Sep-2022 crisis max)")
        print(f"  Inverse: FOS/TAC = (ratio − {_B:.3f}) / {_A:.3f}")
        print(f"    g(1.52) = {sim_to_fos_tac(1.52):.3f}")
        print(f"    g(2.00) = {sim_to_fos_tac(2.00):.3f}")
        print()
        print(f"  q_ch4:  z-score, plant mean={self.plant.mean_q_ch4:.1f} → sim mean={self.plant.sim_mean_q_ch4:.0f}")
        print(f"  q_ad:   z-score, plant mean={self.plant.mean_q_ad:.1f} → sim mean={self.plant.sim_mean_q_ad:.0f}")
        print(f"  pH:     passthrough (same units)")
        print(f"  feed_mult: passthrough (dimensionless)")


# ── Quick validation ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    cal = SCADACalibration()
    cal.describe()

    print()
    print("=== Example: Muscatine daily reading →  simulator obs ===")
    cases = [
        ("Normal",    0.23, 7.27, 103.5, 178.0, 1.00),
        ("Elevated",  0.35, 7.15, 88.0,  180.0, 0.90),
        ("Alarm",     0.41, 7.05, 75.0,  160.0, 0.80),
        ("Crisis",    0.93, 6.85, 42.0,  140.0, 0.70),
    ]
    print(f"  {'Case':<12} {'FOS/TAC':>8} {'pH':>6} → {'obs[0]':>8} {'obs[1]':>6} {'obs[2]':>8} {'obs[3]':>8} {'obs[4]':>6}")
    for name, fos, ph, q_ch4, q_ad, fm in cases:
        obs = cal.to_sim_obs(fos, ph, q_ch4, q_ad, fm)
        print(f"  {name:<12} {fos:>8.3f} {ph:>6.2f} → {obs[0]:>8.3f} {obs[1]:>6.2f} {obs[2]:>8.1f} {obs[3]:>8.1f} {obs[4]:>6.2f}")

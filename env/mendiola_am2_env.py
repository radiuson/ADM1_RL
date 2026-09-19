"""
Single-stage AD model of Mendiola-Rodriguez & Ricardez-Sandoval (2022),
*Digital Chemical Engineering* 3:100023, Section 3.1 - a four-state AM2-type
CSTR treating tequila vinasses.

    dz1/dt = -D*alpha*z1 + mu1*z1
    dz2/dt = -D*alpha*z2 + mu2*z2
    dz3/dt =  D*(S1_in - z3) - mu1*z1
    dz4/dt =  D*(S2_in - z4) - mu2*z2 + gamma1*mu1*z1

    mu1 = mu1max * S1 / (ks1 + S1)                    Monod, acidogenesis
    mu2 = mu2max * S2 / (ks2 + S2 + (S2/kI2)^2)       Haldane, methanogenesis

States: z1 acidogenic biomass (g/L), z2 methanogenic biomass (mmol/L),
z3 = S1 substrate COD (g COD/L), z4 = S2 VFA (mmol VFA/L).
Action: dilution rate D in [0.05, 0.56] 1/d.

Parameter values, initial conditions, disturbance schedule and the eight
uncertainty realisations are taken from that paper (Tables 1-4).  The reported
objective is accumulated COD reduction over a 365-day horizon.

This environment exists so the present work can be run under a published
AD-RL problem setting, not to reproduce that paper's numbers.
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import scipy.integrate
from gymnasium import spaces

# Table 1 - fixed parameters and initial conditions
KS2, KI2, GAMMA1 = 36.468, 16.773, 2.6584
Z0 = np.array([3.002, 143.496, 16.0, 60.0], dtype=np.float64)
S1_IN_NOM, S2_IN_NOM = 16.0, 60.0
D_LOW, D_HIGH = 0.05, 0.56

# Table 4 - uncertainty realisations; index 4 is the nominal case
UNCERTAINTY = [
    dict(mu1max=0.63, mu2max=0.75, ks1=6.20, alpha=0.40),
    dict(mu1max=0.95, mu2max=1.03, ks1=4.53, alpha=0.32),
    dict(mu1max=1.20, mu2max=0.88, ks1=5.71, alpha=0.47),
    dict(mu1max=0.48, mu2max=0.97, ks1=7.30, alpha=0.50),
    dict(mu1max=0.7999, mu2max=0.7357, ks1=5.207, alpha=0.458),   # nominal
    dict(mu1max=0.55, mu2max=0.65, ks1=3.27, alpha=0.55),
    dict(mu1max=1.05, mu2max=0.81, ks1=3.85, alpha=0.49),
    dict(mu1max=0.87, mu2max=0.92, ks1=3.12, alpha=0.53),
]

# Scenario 2 - step disturbances on the inlet concentrations
DISTURBANCES = [(15, -0.25), (101, -0.15), (247, +0.20), (320, +0.25)]


class MendiolaAM2Env(gym.Env):
    """Gymnasium wrapper around the four-state model above."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        realisation: int = 4,
        disturbances: bool = False,
        horizon_days: int = 365,
        step_days: float = 1.0,
        vfa_soft_limit: float = 90.0,   # mmol VFA/L; see note in step()
    ):
        super().__init__()
        p = UNCERTAINTY[realisation]
        self.mu1max, self.mu2max = p["mu1max"], p["mu2max"]
        self.ks1, self.alpha = p["ks1"], p["alpha"]
        self.realisation = realisation
        self.disturbances = disturbances
        self.horizon_days = horizon_days
        self.step_days = step_days
        self.max_steps = int(horizon_days / step_days)
        self.vfa_soft_limit = vfa_soft_limit

        self.action_space = spaces.Box(D_LOW, D_HIGH, shape=(1,), dtype=np.float32)
        # Observation: the four states plus the applied dilution rate.
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, D_LOW], dtype=np.float32),
            high=np.array([50.0, 500.0, 40.0, 400.0, D_HIGH], dtype=np.float32),
            dtype=np.float32,
        )
        self.z = Z0.copy()
        self.t_days = 0.0
        self.current_step = 0
        self.D = float(np.mean([D_LOW, D_HIGH]))
        self.cod_removed = 0.0

    # ── dynamics ─────────────────────────────────────────────────────────────
    def _inlet(self) -> Tuple[float, float]:
        s1, s2 = S1_IN_NOM, S2_IN_NOM
        if self.disturbances:
            for day, frac in DISTURBANCES:
                if self.t_days >= day:
                    s1, s2 = S1_IN_NOM * (1 + frac), S2_IN_NOM * (1 + frac)
        return s1, s2

    def _ode(self, _t, z, D, s1_in, s2_in):
        z1, z2, S1, S2 = np.maximum(z, 0.0)
        mu1 = self.mu1max * S1 / (self.ks1 + S1 + 1e-12)
        mu2 = self.mu2max * S2 / (KS2 + S2 + (S2 / KI2) ** 2 + 1e-12)
        return [
            -D * self.alpha * z1 + mu1 * z1,
            -D * self.alpha * z2 + mu2 * z2,
            D * (s1_in - S1) - mu1 * z1,
            D * (s2_in - S2) - mu2 * z2 + GAMMA1 * mu1 * z1,
        ]

    # ── gym API ──────────────────────────────────────────────────────────────
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.z = Z0.copy()
        self.t_days = 0.0
        self.current_step = 0
        self.D = float(np.mean([D_LOW, D_HIGH]))
        self.cod_removed = 0.0
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        return np.array([*self.z, self.D], dtype=np.float32)

    def step(self, action):
        self.D = float(np.clip(np.asarray(action).ravel()[0], D_LOW, D_HIGH))
        s1_in, s2_in = self._inlet()

        sol = scipy.integrate.solve_ivp(
            self._ode, [0, self.step_days], self.z,
            args=(self.D, s1_in, s2_in), method="LSODA", rtol=1e-6, atol=1e-8,
        )
        self.z = np.maximum(sol.y[:, -1], 0.0)
        self.t_days += self.step_days
        self.current_step += 1

        # Objective of Eq. (15) in the source paper: minimise the sum of the
        # effluent COD z3 over the horizon, i.e. maximise -sum(z3).  The paper
        # reports this sum ("accumulated COD") as 109.0748 g COD/L for the
        # optimised nominal case.
        cod_step = float(self.z[2]) * self.step_days
        self.cod_removed += cod_step

        # The source paper optimises COD reduction alone and reports no safety
        # metric.  VFA accumulation is tracked here so the same run can also be
        # scored on constraint violation, which is what this work adds.
        vfa = float(self.z[3])
        violated = vfa > self.vfa_soft_limit
        reward = -cod_step        # maximise -sum(z3)

        washout = self.z[0] < 1e-3 or self.z[1] < 1e-3
        terminated = bool(washout)
        truncated = self.current_step >= self.max_steps

        info = {
            "cod_sum_z3": self.cod_removed,
            "cod_step": cod_step,
            "vfa": vfa,
            "violated": violated,
            "washout": washout,
            "S1": float(self.z[2]),
            "z1": float(self.z[0]),
            "z2": float(self.z[1]),
        }
        return self._obs(), float(reward), terminated, truncated, info

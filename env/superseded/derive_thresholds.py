"""Derive the simulator's VFA constraint thresholds from the plant record.

The plant's monitoring limits cannot be imported into the simulator directly:
simulated VFA runs about an order of magnitude below the Muscatine record, so
the 1500 mg/L alarm would sit at the simulator's 99.9th percentile and leave
the constraint inactive.

Percentile alignment (quantile mapping) is the usual remedy, but it is not
sound here.  The two distributions differ in shape, not only in position: the
simulated upper tail reaches 9.3x its routine level where the plant record
reaches only 2.1x.  Matching percentiles would therefore give the simulated
hard limit roughly four times the margin the plant actually operates against.

What the two systems do share is a routine operating level.  Each plant limit
is expressed as a multiple of that level, and the multiple is transferred:

    threshold_sim = (limit_plant / routine_plant) * routine_sim

Two choices define `routine_sim` and are fixed here so the result is
reproducible:

1.  Anchor scenario.  Routine means the nominal scenario.  Designed stress
    scenarios are excluded: admitting them moves the anchor by a factor of two
    (0.068 to 0.135), whereas routine-operation anchors agree to within 11 %
    (0.089 to 0.099).
2.  Constraint-driven termination is disabled while sampling.  The environment
    terminates on the very thresholds being derived, so leaving it active makes
    the derivation self-referential - the distribution would be truncated at
    whatever limit happened to be in force.

Run:  python3 env/derive_thresholds.py
"""

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.adm1_gym_env_std import ADM1Env_Std  # noqa: E402

# Muscatine Dig1, 1103 daily observations 2020-01 to 2023-03
# (Hunter & Schroer 2023).  VFA as mg/L acetic acid equivalent.
PLANT_ROUTINE_MGL = 1178.0          # median of the record
PLANT_LIMITS = {
    "soft": (1500.0, "routine monitoring alarm"),
    "hard": (2471.0, "Sep-2022 excursion peak"),
}

ANCHOR_SCENARIO = "nominal"
# (feed rate m3/d, influent strength multiplier)
ACTION_GRID = [(150, 0.85), (220, 1.00), (300, 1.30)]
HORIZON_DAYS = 60


def routine_level():
    """Median VFA over the nominal scenario, constraint termination disabled."""
    values = []
    for q_ad, mult in ACTION_GRID:
        env = ADM1Env_Std(ANCHOR_SCENARIO, obs_mode="scada", step_size=1.0)
        env.VFA_HARD_KGCOD_M3 = 1e9      # instance-level; avoids self-reference
        env.NH3_HARD_KMOL_M3 = 1e9
        env.PH_HARD_LO, env.PH_HARD_HI = 0.0, 14.0
        env.reset(seed=0)
        action = np.array([q_ad, mult], dtype=np.float32)
        for _ in range(HORIZON_DAYS):
            _, _, terminated, truncated, _ = env.step(action)
            values.append(env._calculate_total_vfa(env.current_state))
            if terminated or truncated:
                break
    return float(np.median(values))


def main():
    routine = routine_level()
    print(f"plant routine (median)      {PLANT_ROUTINE_MGL:8.0f} mg/L")
    print(f"simulated routine (median)  {routine:8.3f} kg COD/m3\n")
    print(f"{'limit':<6}{'plant mg/L':>12}{'multiple':>10}{'threshold':>12}  basis")
    for name, (mgl, basis) in PLANT_LIMITS.items():
        ratio = mgl / PLANT_ROUTINE_MGL
        print(f"{name:<6}{mgl:>12.0f}{ratio:>9.3f}x{ratio * routine:>12.3f}  {basis}")


if __name__ == "__main__":
    main()

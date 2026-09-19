#!/usr/bin/env python3
"""
Reward Configuration Presets for ADM1Env_v2
============================================

Reward configurations used in the paper experiments.  The primary
configuration is ``safety_first`` (linear + constant penalty structure).
The ablation variants ``sf_linear_only`` and ``sf_constant_only`` are used
in the reward ablation study (Section IV-C).

Usage:
    from training.reward_configs import REWARD_CONFIGS
    env = ADM1Env_v2(reward_config=REWARD_CONFIGS['safety_first'])
"""

# Conservative: Safety-first configuration
# High penalties for violations, encourages large safety margins
CONSERVATIVE = {
    'production_scale': 2000.0,
    'ph_penalty_scale': 5.0,      # 2.5× default (2.0)
    'vfa_penalty_scale': 15.0,    # 5× default (3.0)
    'nh3_penalty_scale': 100.0,   # 2× default (50.0)
    'energy_penalty_max': 0.2,
    'stability_penalty_max': 0.1,
}

# Balanced: Default configuration (unchanged)
# Moderate penalties, allows some violations
BALANCED = {
    'production_scale': 2000.0,
    'ph_penalty_scale': 2.0,
    'vfa_penalty_scale': 3.0,
    'nh3_penalty_scale': 50.0,
    'energy_penalty_max': 0.2,
    'stability_penalty_max': 0.1,
}

# Aggressive: Production-first configuration
# Low penalties, maximizes CH4 production
AGGRESSIVE = {
    'production_scale': 2000.0,
    'ph_penalty_scale': 1.0,      # 0.5× default
    'vfa_penalty_scale': 1.5,     # 0.5× default
    'nh3_penalty_scale': 25.0,    # 0.5× default
    'energy_penalty_max': 0.1,    # Lower energy penalty
    'stability_penalty_max': 0.05, # Lower stability penalty
}

# Research: For RL research (intermediate penalties)
# Designed to encourage RL exploration while maintaining safety
RESEARCH = {
    'production_scale': 2000.0,
    'ph_penalty_scale': 3.0,      # 1.5× default
    'vfa_penalty_scale': 8.0,     # 2.7× default
    'nh3_penalty_scale': 75.0,    # 1.5× default
    'energy_penalty_max': 0.2,
    'stability_penalty_max': 0.1,
}

# Warning zones: Two-tier penalty system
# Soft warning (0.15-0.2) + hard penalty (>0.2)
WARNING_ZONES = {
    'production_scale': 2000.0,
    'ph_penalty_scale': 2.0,
    'vfa_penalty_scale': 3.0,
    'vfa_warning_threshold': 0.15,  # Warning zone
    'vfa_warning_scale': 5.0,       # Linear penalty in warning zone
    'vfa_danger_scale': 20.0,       # Quadratic penalty in danger zone
    'nh3_penalty_scale': 50.0,
    'energy_penalty_max': 0.2,
    'stability_penalty_max': 0.1,
}

# Exponential penalties: Exponentially increasing penalties
# More aggressive than quadratic for large violations
EXPONENTIAL = {
    'production_scale': 2000.0,
    'ph_penalty_scale': 2.0,
    'vfa_penalty_scale': 10.0,     # Exponential base
    'vfa_penalty_type': 'exponential',  # Use exp instead of quadratic
    'nh3_penalty_scale': 50.0,
    'energy_penalty_max': 0.2,
    'stability_penalty_max': 0.1,
}

# Safety-First: Linear + Constant penalty structure
# Fixes reward imbalance problem: production reward ~1.3 vs quadratic penalty ~0.001
#
# With linear+constant mode:
#   VFA = 0.21 (5% over): -1.0 (constant) - 0.01*10 = -1.10  → net ~ +0.2 (barely profitable)
#   VFA = 0.25 (25% over): -1.0 (constant) - 0.05*10 = -1.50 → net ~ -0.2 (UNPROFITABLE)
#   No violation: production ~ 1.0-1.5, no fixed cost
#
# This forces the agent to learn safe operation is more profitable.
SAFETY_FIRST = {
    'production_scale': 2000.0,
    'penalty_type': 'linear+constant',
    'ph_penalty_scale': 5.0,         # Linear pH penalty per unit deviation
    'ph_constant_penalty': 0.8,      # Fixed penalty for any pH violation
    'vfa_penalty_scale': 10.0,       # Linear VFA penalty per unit excess
    'vfa_constant_penalty': 1.0,     # Fixed penalty for any VFA violation
    'nh3_penalty_scale': 100.0,      # Linear NH3 penalty per unit excess
    'nh3_constant_penalty': 1.5,     # Fixed penalty for any NH3 violation
    'energy_penalty_max': 0.2,
    'stability_penalty_max': 0.1,
}

# Curriculum variant: Safety-First + F_K thermal quality shaping
# Adds an immediate gradient for temperature management before VFA/pH violations.
# F_K = exp(-(T_L-T_a)^2 / 2σ²): reward +(F_K-0.8)*0.3 so the agent proactively
# keeps T_L close to T_a during τₐ curriculum (τₐ slow → T_a lags → F_K drops).
SAFETY_FIRST_CURRICULUM = {
    **SAFETY_FIRST,
    'fk_bonus_scale': 0.3,
}

# Ablation: Safety-First with ONLY linear penalty (no constant term)
# Removes the fixed "entry cost" for any violation — tests if constant term is necessary
SF_LINEAR_ONLY = {
    'production_scale': 2000.0,
    'penalty_type': 'linear+constant',
    'ph_penalty_scale': 5.0,
    'ph_constant_penalty': 0.0,          # ablated out
    'vfa_penalty_scale': 10.0,
    'vfa_constant_penalty': 0.0,         # ablated out
    'nh3_penalty_scale': 100.0,
    'nh3_constant_penalty': 0.0,         # ablated out
    'energy_penalty_max': 0.2,
    'stability_penalty_max': 0.1,
}

# Ablation: Safety-First with ONLY constant penalty (no linear proportional term)
# Removes the proportional "how bad" signal — tests if linear term is necessary
SF_CONSTANT_ONLY = {
    'production_scale': 2000.0,
    'penalty_type': 'linear+constant',
    'ph_penalty_scale': 0.0,             # ablated out
    'ph_constant_penalty': 0.8,
    'vfa_penalty_scale': 0.0,            # ablated out
    'vfa_constant_penalty': 1.0,
    'nh3_penalty_scale': 0.0,            # ablated out
    'nh3_constant_penalty': 1.5,
    'energy_penalty_max': 0.2,
    'stability_penalty_max': 0.1,
}

# Safety-Target: graduated linear-only penalties, calibrated to allow
# exploration near (but below) the VFA threshold.
#
# Key insight: safety_first's vfa_constant_penalty=-1.0 makes ANY violation
# unprofitable (net reward ≈ 0), so the policy learns to stay far below the
# VFA threshold → over-conservative, low CH4.
#
# This config removes the constant "cliff" and instead uses a stronger linear
# scale, so small overshoots are affordable while large violations are MORE
# costly than safety_first:
#
#   VFA = 0.315 (5% over):  safety_first -1.15,  safety_target -0.30  → net +0.85
#   VFA = 0.375 (25% over): safety_first -1.75,  safety_target -1.50  → net +0.25
#   VFA = 0.450 (50% over): safety_first -2.50,  safety_target -3.00  → net -0.50
#   VFA = 0.525 (75% over): safety_first -3.25,  safety_target -4.50  → net -1.25
#
# The policy can freely explore up to ~VFA 0.32-0.33 and still earn positive
# reward, but is strongly punished for letting VFA drift above 0.45.
SAFETY_TARGET = {
    'production_scale': 2000.0,
    'penalty_type': 'linear+constant',
    'ph_penalty_scale': 8.0,             # 1.6× stronger than safety_first
    'ph_constant_penalty': 0.0,          # no binary cliff
    'vfa_penalty_scale': 20.0,           # 2× stronger linear (compensates for no constant)
    'vfa_constant_penalty': 0.0,         # KEY CHANGE: no binary cliff
    'nh3_penalty_scale': 150.0,          # 1.5× stronger linear
    'nh3_constant_penalty': 0.0,         # no binary cliff
    'energy_penalty_max': 0.2,
    'stability_penalty_max': 0.1,
}

# ---------------------------------------------------------------------------
# VFA penalty sweep — Plan A (w/c grid, ST-SAC pH/NH3 base)
#
# Base identical to SAFETY_TARGET except vfa_penalty_scale and
# vfa_constant_penalty, so only the VFA reward axis changes.
#
# Axis 1 (c_VFA = 0): w_VFA ∈ {5,10,15,20,25,30,40}
#   → w=20 is the reference (SAFETY_TARGET / ST-SAC)
#
# Axis 2 (w_VFA = 20): c_VFA ∈ {0.25,0.5,1.0,1.5,2.0}
#   → c=0 is the reference (SAFETY_TARGET / ST-SAC)
# ---------------------------------------------------------------------------
_SWEEP_BASE = {
    'production_scale':    2000.0,
    'penalty_type':        'linear+constant',
    # ST-SAC pH/NH3 params (consistent base across all sweep points)
    'ph_penalty_scale':    8.0,
    'ph_constant_penalty': 0.0,
    'nh3_penalty_scale':   150.0,
    'nh3_constant_penalty': 0.0,
    'energy_penalty_max':  0.2,
    'stability_penalty_max': 0.1,
}

# Axis 1 — c_VFA = 0, varying w_VFA  (ST-SAC w=20 is reference, not repeated here)
# Low end
SWEEP_W2  = {**_SWEEP_BASE, 'vfa_penalty_scale':  2.0, 'vfa_constant_penalty': 0.0}
SWEEP_W3  = {**_SWEEP_BASE, 'vfa_penalty_scale':  3.0, 'vfa_constant_penalty': 0.0}
SWEEP_W5  = {**_SWEEP_BASE, 'vfa_penalty_scale':  5.0, 'vfa_constant_penalty': 0.0}
SWEEP_W10 = {**_SWEEP_BASE, 'vfa_penalty_scale': 10.0, 'vfa_constant_penalty': 0.0}
# Dense zone w=11–14: transition from high-VR to low-VR regime
SWEEP_W11 = {**_SWEEP_BASE, 'vfa_penalty_scale': 11.0, 'vfa_constant_penalty': 0.0}
SWEEP_W12 = {**_SWEEP_BASE, 'vfa_penalty_scale': 12.0, 'vfa_constant_penalty': 0.0}
SWEEP_W13 = {**_SWEEP_BASE, 'vfa_penalty_scale': 13.0, 'vfa_constant_penalty': 0.0}
SWEEP_W14 = {**_SWEEP_BASE, 'vfa_penalty_scale': 14.0, 'vfa_constant_penalty': 0.0}
SWEEP_W15 = {**_SWEEP_BASE, 'vfa_penalty_scale': 15.0, 'vfa_constant_penalty': 0.0}
# Mid zone w=16, 18: between w=15 and w=20
SWEEP_W16 = {**_SWEEP_BASE, 'vfa_penalty_scale': 16.0, 'vfa_constant_penalty': 0.0}
SWEEP_W18 = {**_SWEEP_BASE, 'vfa_penalty_scale': 18.0, 'vfa_constant_penalty': 0.0}
# w=20 → SAFETY_TARGET (reference)
SWEEP_W22 = {**_SWEEP_BASE, 'vfa_penalty_scale': 22.0, 'vfa_constant_penalty': 0.0}
SWEEP_W25 = {**_SWEEP_BASE, 'vfa_penalty_scale': 25.0, 'vfa_constant_penalty': 0.0}
SWEEP_W30 = {**_SWEEP_BASE, 'vfa_penalty_scale': 30.0, 'vfa_constant_penalty': 0.0}
SWEEP_W40 = {**_SWEEP_BASE, 'vfa_penalty_scale': 40.0, 'vfa_constant_penalty': 0.0}
# High end
SWEEP_W50 = {**_SWEEP_BASE, 'vfa_penalty_scale': 50.0, 'vfa_constant_penalty': 0.0}
SWEEP_W60 = {**_SWEEP_BASE, 'vfa_penalty_scale': 60.0, 'vfa_constant_penalty': 0.0}

# Axis 2 — w_VFA = 20, varying c_VFA  (ST-SAC c=0 is reference, not repeated here)
SWEEP_W20C025 = {**_SWEEP_BASE, 'vfa_penalty_scale': 20.0, 'vfa_constant_penalty': 0.25}
SWEEP_W20C050 = {**_SWEEP_BASE, 'vfa_penalty_scale': 20.0, 'vfa_constant_penalty': 0.50}
SWEEP_W20C100 = {**_SWEEP_BASE, 'vfa_penalty_scale': 20.0, 'vfa_constant_penalty': 1.00}
SWEEP_W20C150 = {**_SWEEP_BASE, 'vfa_penalty_scale': 20.0, 'vfa_constant_penalty': 1.50}
SWEEP_W20C200 = {**_SWEEP_BASE, 'vfa_penalty_scale': 20.0, 'vfa_constant_penalty': 2.00}

# Legacy (old MS-SAC base, kept for backward compatibility)
SWEEP_C025 = {**_SWEEP_BASE, 'vfa_penalty_scale': 10.0, 'vfa_constant_penalty': 0.25}
SWEEP_C050 = {**_SWEEP_BASE, 'vfa_penalty_scale': 10.0, 'vfa_constant_penalty': 0.50}
SWEEP_C150 = {**_SWEEP_BASE, 'vfa_penalty_scale': 10.0, 'vfa_constant_penalty': 1.50}

# Collection of all configs

# ── Penalty-structure sweep (2026-09) ────────────────────────────────────────
# Two families traced on the production-safety plane.  Both share the pH and
# NH3 penalties; they differ only in how the VFA penalty responds to the
# magnitude of an exceedance.
#   PS_*  risk-proportional: penalty scales with excess, no fixed cost
#   PC_*  uniform: a fixed cost on any exceedance, plus a weak linear term
def _sweep(scale, const):
    return {
        'production_scale':      2000.0,
        'penalty_type':          'linear+constant',
        'ph_penalty_scale':      5.0,
        'ph_constant_penalty':   0.0,
        'vfa_penalty_scale':     scale,
        'vfa_constant_penalty':  const,
        'nh3_penalty_scale':     100.0,
        'nh3_constant_penalty':  0.0,
        'energy_penalty_max':    0.2,
        'stability_penalty_max': 0.1,
    }

PS_W3, PS_W6, PS_W12, PS_W25, PS_W50 = (_sweep(w, 0.0) for w in (3., 6., 12., 25., 50.))
PC_C05, PC_C1, PC_C2, PC_C4 = (_sweep(5.0, c) for c in (0.5, 1.0, 2.0, 4.0))

# ── 2-D penalty grid (2026-09): both w and c non-zero ───────────────────────
# Fills the region between the pure-linear family (c=0) and the fixed-w=5
# constant family, to test whether a larger linear coefficient with a smaller
# fixed cost dominates the reverse at matched violation rates.
PW10_C05, PW10_C2 = _sweep(10.0, 0.5), _sweep(10.0, 2.0)
PW20_C05, PW20_C1, PW20_C2 = _sweep(20.0, 0.5), _sweep(20.0, 1.0), _sweep(20.0, 2.0)
PW40_C05, PW40_C1, PW40_C2 = _sweep(40.0, 0.5), _sweep(40.0, 1.0), _sweep(40.0, 2.0)

REWARD_CONFIGS = {
    'pw10_c05': PW10_C05, 'pw10_c2': PW10_C2,
    'pw20_c05': PW20_C05, 'pw20_c1': PW20_C1, 'pw20_c2': PW20_C2,
    'pw40_c05': PW40_C05, 'pw40_c1': PW40_C1, 'pw40_c2': PW40_C2,
    'ps_w3': PS_W3, 'ps_w6': PS_W6, 'ps_w12': PS_W12, 'ps_w25': PS_W25, 'ps_w50': PS_W50,
    'pc_c05': PC_C05, 'pc_c1': PC_C1, 'pc_c2': PC_C2, 'pc_c4': PC_C4,
    'conservative': CONSERVATIVE,
    'balanced': BALANCED,
    'aggressive': AGGRESSIVE,
    'research': RESEARCH,
    'warning_zones': WARNING_ZONES,
    'exponential': EXPONENTIAL,
    'safety_first': SAFETY_FIRST,
    'safety_first_curriculum': SAFETY_FIRST_CURRICULUM,
    'sf_linear_only': SF_LINEAR_ONLY,
    'sf_constant_only': SF_CONSTANT_ONLY,
    'safety_target': SAFETY_TARGET,
    # VFA sweep — Axis 1 (c=0, varying w)
    'sweep_w2':   SWEEP_W2,
    'sweep_w3':   SWEEP_W3,
    'sweep_w5':   SWEEP_W5,
    'sweep_w10':  SWEEP_W10,
    'sweep_w11':  SWEEP_W11,
    'sweep_w12':  SWEEP_W12,
    'sweep_w13':  SWEEP_W13,
    'sweep_w14':  SWEEP_W14,
    'sweep_w15':  SWEEP_W15,
    'sweep_w16':  SWEEP_W16,
    'sweep_w18':  SWEEP_W18,
    # 'sweep_w20' == 'safety_target' (reference, not duplicated)
    'sweep_w22':  SWEEP_W22,
    'sweep_w25':  SWEEP_W25,
    'sweep_w30':  SWEEP_W30,
    'sweep_w40':  SWEEP_W40,
    'sweep_w50':  SWEEP_W50,
    'sweep_w60':  SWEEP_W60,
    # VFA sweep — Axis 2 (w=20, varying c)
    # 'sweep_w20c0' == 'safety_target' (reference, not duplicated)
    'sweep_w20c025': SWEEP_W20C025,
    'sweep_w20c050': SWEEP_W20C050,
    'sweep_w20c100': SWEEP_W20C100,
    'sweep_w20c150': SWEEP_W20C150,
    'sweep_w20c200': SWEEP_W20C200,
    # Legacy keys (kept for backward compat)
    'sweep_c025': SWEEP_C025,
    'sweep_c050': SWEEP_C050,
    'sweep_c150': SWEEP_C150,
}


def get_reward_config(name: str):
    """
    Get reward configuration by name

    Args:
        name: Configuration name

    Returns:
        Reward config dict

    Example:
        >>> config = get_reward_config('conservative')
    """
    if name not in REWARD_CONFIGS:
        raise ValueError(
            f"Unknown reward config '{name}'. "
            f"Available: {list(REWARD_CONFIGS.keys())}"
        )
    return REWARD_CONFIGS[name].copy()


def compare_configs():
    """Print comparison of all reward configurations"""
    print("=" * 80)
    print("Reward Configuration Comparison")
    print("=" * 80)

    print(f"\n{'Config':<20} {'pH Penalty':<15} {'VFA Penalty':<15} {'NH3 Penalty':<15}")
    print("-" * 80)

    for name, config in REWARD_CONFIGS.items():
        if 'vfa_penalty_type' in config:
            vfa_penalty = f"{config['vfa_penalty_scale']:.1f} (exp)"
        else:
            vfa_penalty = f"{config['vfa_penalty_scale']:.1f}"

        print(f"{name:<20} {config['ph_penalty_scale']:<15.1f} "
              f"{vfa_penalty:<15} {config['nh3_penalty_scale']:<15.1f}")

    print("\n" + "=" * 80)
    print("Penalty Scale Interpretation:")
    print("=" * 80)
    print("  Conservative: 5× stronger VFA penalty (15.0 vs 3.0)")
    print("  Balanced:     Default (baseline)")
    print("  Aggressive:   0.5× weaker penalties (maximize production)")
    print("  Research:     2.7× stronger VFA penalty (good for RL)")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    compare_configs()

    # Example: penalty at VFA = 0.21 kmol COD/m³ (5% over the 0.2 threshold)
    print("\n" + "=" * 80)
    print("Example penalty at VFA = 0.21 kmol COD/m³ (excess = 0.01)")
    print("=" * 80)

    vfa = 0.21
    vfa_excess = vfa - 0.2  # 0.01

    for name, config in REWARD_CONFIGS.items():
        ptype = config.get('penalty_type', 'quadratic')
        scale = config.get('vfa_penalty_scale', 0.0)
        const = config.get('vfa_constant_penalty', 0.0)
        if ptype == 'linear+constant':
            penalty = -(scale * vfa_excess + const)
        else:
            # Standard quadratic penalty
            penalty = -(vfa_excess ** 2) * scale
        print(f"  {name:<20} [{ptype:<16}]  penalty = {penalty:.4f}")

    print("=" * 80)


# ── (w, c) grid at the scale the trade-off actually occupies ─────────────────
# The production term is q_ch4 / 2000 and reaches about 1.9 at the highest
# feed this environment allows, so a penalty w*excess only competes with it
# while w*excess is of that order.  Excess reaches roughly 1.5 kg COD/m3 at
# full feed under the heaviest loading, which puts the contested range at
# w of order 1: below 0.05 the optimum is full feed in every scenario, above
# 3 it stops moving.  The constant term is scaled the same way - the marginal
# production gain from the feed step that first causes a violation is about
# 0.46, so c is swept over a comparable range rather than the unit scale.
for _w in (0.1, 0.25, 0.5, 1.0, 2.0, 3.0):
    REWARD_CONFIGS[f'lw{_w:g}'.replace('.', 'p')] = _sweep(_w, 0.0)
for _w, _c in ((0.5, 0.05), (0.5, 0.15), (0.5, 0.4),
               (1.0, 0.05), (1.0, 0.15), (1.0, 0.4)):
    REWARD_CONFIGS[f'lw{_w:g}c{_c:g}'.replace('.', 'p')] = _sweep(_w, _c)

# Fill the low-weight corner of the grid so the constant family is sampled
# wherever the pure-linear family is; without these the highest-production
# region is covered on one side only.
for _w, _c in ((0.1, 0.05), (0.1, 0.15), (0.25, 0.05), (0.25, 0.15)):
    REWARD_CONFIGS[f'lw{_w:g}c{_c:g}'.replace('.', 'p')] = _sweep(_w, _c)


# ── Extended sweep: the two ends of the frontier, plus the pure-constant form ─
# The middle of the grid showed no structural effect, so the extension targets
# where one is predicted: as w approaches zero the linear term stops deterring
# anything and only a constant can, and near zero violation a fixed cost may
# reach the same safety as a large linear weight at less production cost.  The
# pure-constant form (w = 0) is the opposite extreme to the risk-proportional
# one and had not been tested at all.
for _w in (0.02, 0.05, 5.0, 10.0, 20.0):
    REWARD_CONFIGS[f'lw{_w:g}'.replace('.', 'p')] = _sweep(_w, 0.0)
for _w, _c in ((0.02, 0.05), (0.02, 0.15), (0.05, 0.05), (0.05, 0.15),
               (0.5, 0.8), (0.5, 1.5), (1.0, 0.8), (1.0, 1.5),
               (2.0, 0.4), (2.0, 0.8)):
    REWARD_CONFIGS[f'lw{_w:g}c{_c:g}'.replace('.', 'p')] = _sweep(_w, _c)
for _c in (0.15, 0.4, 0.8):
    REWARD_CONFIGS[f'lc{_c:g}'.replace('.', 'p')] = _sweep(0.0, _c)


# Weights for the action-level sensitivity study.  The analytic pre-check puts
# the contested range at w = 0.1 to 10 for every candidate level, so the same
# weights are used throughout and only the level changes.
for _w in (0.1, 0.5, 2.0, 5.0):
    REWARD_CONFIGS[f'sw{_w:g}'.replace('.', 'p')] = _sweep(_w, 0.0)


# Denser weights for the action-level sensitivity study.
for _w in (0.25, 1.0, 3.0, 10.0):
    REWARD_CONFIGS[f'sw{_w:g}'.replace('.', 'p')] = _sweep(_w, 0.0)

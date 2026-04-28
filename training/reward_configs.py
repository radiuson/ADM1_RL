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

# Collection of all configs
REWARD_CONFIGS = {
    'conservative': CONSERVATIVE,
    'balanced': BALANCED,
    'aggressive': AGGRESSIVE,
    'research': RESEARCH,
    'warning_zones': WARNING_ZONES,
    'exponential': EXPONENTIAL,
    'safety_first': SAFETY_FIRST,
    'sf_linear_only': SF_LINEAR_ONLY,
    'sf_constant_only': SF_CONSTANT_ONLY,
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

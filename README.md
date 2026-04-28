# ADM1_RL — Safety-First RL Control of Anaerobic Digestion under Thermal Stress

Code accompanying the paper:

> **Safety-First Control of Anaerobic Digestion Under Thermal Stress Using Reinforcement Learning and a Temperature-Extended ADM1 Model**

---

## Overview

This repository implements a Soft Actor-Critic (SAC) reinforcement learning controller for anaerobic digestion (AD) biogas plants. The environment wraps a temperature-extended ADM1 ODE model and exposes six evaluation scenarios covering nominal operation, load disturbances, and thermal stress.

Key features:
- Temperature-extended ADM1 solver (40-state ODE: 38 biological/chemical states + liquid temperature T_L + microbial adaptation temperature T_a)
- Safety-first reward with linear + constant penalty structure
- Six evaluation scenarios: `nominal`, `high_load`, `low_load`, `shock_load`, `temperature_drop`, `cold_winter`
- Full (13-dim) and compact (5-dim) observation modes
- Engineering baselines: Constant, PID, Cascaded PID, MPC, NMPC (oracle)
- Reward ablation study: safety-first vs. linear-only vs. constant-only penalties

---

## Installation

```bash
# Clone the repository
git clone https://github.com/radiuson/ADM1_RL.git
cd ADM1_RL

# Install in editable mode
pip install -e .
```

Requirements: Python ≥ 3.9, PyTorch ≥ 2.0, stable-baselines3 ≥ 2.0, gymnasium ≥ 0.29.

---

## Repository Structure

```
env/                    # ADM1 Gymnasium environment
  adm1_gym_env.py       # ADM1Env_v2 (main environment class)
  adm1_solver.py        # Temperature-extended ADM1 ODE solver
  scenario_manager.py   # Scenario loading and disturbance injection
  scenarios.yaml        # Scenario definitions
  data/                 # Influent time-series data

training/               # SAC training
  train_sac.py          # Standalone training script
  run_experiment.py     # Full experiment runner (train + eval + figures)
  reward_configs.py     # Reward configuration presets
  configs/              # YAML experiment configs

baselines/              # Engineering baseline controllers
  baseline_controllers.py
  mpc_controller.py
  nmpc_controller.py
  evaluate_baselines.py
  tune_pid.py

evaluation/             # Evaluation and metrics
  metrics_calculator.py # Per-episode metrics (production, safety, stability)
  evaluate_rl_policy.py
  full_evaluation.py
  eval_ablation.py

analysis/               # Figure generation scripts
  plot_combo.py      # Main results figure (score + violation rate)
  plot_generalization.py  # Cross-scenario generalisation figure
  plot_ablation.py   # Reward ablation figure

scripts/                # Re-evaluation utilities
  rerun_eval.py         # Re-evaluate SAC per-run JSONs
  rerun_baseline_eval.py
  rerun_multi_eval.py
```

---

## Quick Start

### Training

```bash
# Train SAC on nominal scenario (paper default)
python training/train_sac.py --scenario nominal --seed 42

# Full paper training run: 6 scenarios × 3 seeds
for scenario in nominal high_load low_load shock_load temperature_drop cold_winter; do
    for seed in 42 123 456; do
        python training/train_sac.py --scenario $scenario --seed $seed
    done
done
```

### Evaluation

```bash
# Full experiment pipeline (train + eval + baselines + figures)
python training/run_experiment.py --config training/configs/experiment_config.yaml --mode full

# Evaluate engineering baselines
python evaluation/full_evaluation.py

# Generate paper figures
python analysis/plot_combo.py --results-dir /path/to/results
python analysis/plot_generalization.py --results-dir /path/to/results
python analysis/plot_ablation.py --results-dir /path/to/results
```

---

## Environment Details

| Parameter | Value |
|-----------|-------|
| Observation space (full) | 13-dim continuous |
| Observation space (compact) | 5-dim: pH, q_CH4, T_L_norm, q_ad, feed_mult |
| Action space | 3-dim: q_ad ∈ [50,300] m³/d, feed_mult ∈ [0.7,1.3], Q_HEX ∈ [−5000,5000] W |
| Episode length | 2880 steps (30 days at 15-min intervals) |
| Safety thresholds | pH ∈ [6.8, 7.8], VFA ≤ 0.2 kg COD/m³, NH3 ≤ 0.002 kmol/m³ |
| Termination (catastrophic) | pH < 5.8 or > 8.8, VFA > 0.8, NH3 > 0.01 |

---

## Reward Function

The safety-first reward combines:
- **Production**: q_CH4 / 2000 (normalized methane flow)
- **Safety penalty**: linear + constant penalty for pH, VFA, and NH3 violations
- **Energy penalty**: −(q_ad / 300)² × 0.2
- **Stability penalty**: −|Δq_CH4| / 100 × 0.1

---

## License

MIT

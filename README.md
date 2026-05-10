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

## Reproducing the Paper

The full reproduction runs in five stages. Set `RESULTS` to your preferred
output directory before starting.

```bash
export RESULTS=./results
```

### Stage 1 — Main SAC training (6 scenarios × 3 seeds × full + compact obs.)

```bash
python training/run_experiment.py \
    --config training/configs/experiment_config.yaml \
    --mode train \
    --output-dir $RESULTS/single_scenario
```

### Stage 2 — Extra seeds for high-variance scenarios (High Load, Cold Winter, Shock Load)

```bash
python training/run_experiment.py \
    --config training/configs/train_extra_seeds.yaml \
    --output-dir $RESULTS/single_scenario
```

### Stage 3 — Reward ablation training (linear-only and constant-only variants)

```bash
python training/run_experiment.py \
    --config training/configs/reward_ablation.yaml \
    --output-dir $RESULTS/ablation_linear_only

python training/run_experiment.py \
    --config training/configs/reward_ablation_constant_only.yaml \
    --output-dir $RESULTS/ablation_constant_only
```

### Stage 4 — Multi-scenario (domain-randomised) training

```bash
python training/run_experiment.py \
    --config training/configs/multiscenario.yaml \
    --output-dir $RESULTS/sac_multi_scenario
```

### Stage 5 — Evaluation and figures

```bash
# Re-evaluate all SAC results with the fixed MetricsCalculator
python scripts/rerun_eval.py --results-dir $RESULTS

# Re-evaluate reward ablation results
python evaluation/eval_ablation.py --results-dir $RESULTS

# Re-evaluate multi-scenario results
python scripts/rerun_multi_eval.py --results-dir $RESULTS

# Re-evaluate engineering baselines (Constant, PID, Cascaded PID)
python scripts/rerun_baseline_eval.py --results-dir $RESULTS

# Generate paper figures
python analysis/plot_combo.py         --results-dir $RESULTS --output-dir $RESULTS/figures
python analysis/plot_generalization.py --results-dir $RESULTS --output-dir $RESULTS/figures
python analysis/plot_ablation.py      --results-dir $RESULTS --output-dir $RESULTS/figures
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

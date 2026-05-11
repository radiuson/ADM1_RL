# ADM1_RL — Safety-First RL Control of Anaerobic Digestion under Thermal Stress

Code and results accompanying the paper:

> **Safety-First Control of Anaerobic Digestion Under Thermal Stress Using Reinforcement Learning and a Temperature-Extended ADM1 Model**

---

## Overview

This repository implements a Soft Actor-Critic (SAC) reinforcement learning controller for anaerobic digestion (AD) biogas plants. The environment wraps a temperature-extended ADM1 ODE model and exposes six evaluation scenarios covering nominal operation, persistent load changes, transient shocks, and thermal stress.

Key features:
- Temperature-extended ADM1 solver (40-state ODE: 38 biological/chemical states + liquid temperature T_L + microbial adaptation temperature T_a)
- Safety-first reward with linear + constant penalty structure
- Six evaluation scenarios: `nominal`, `high_load`, `low_load`, `shock_load`, `temperature_drop`, `cold_winter`
- Full (13-dim) and compact (5-dim) observation modes, plus 36 optional extra state variables via `obs_extra`
- Engineering baselines: Constant, PID, Cascaded PID, MPC, NMPC
- Reward ablation study: safety-first vs. linear-only vs. constant-only
- Pre-computed results and TensorBoard logs included (no retraining required for analysis)

---

## Installation

```bash
git clone https://github.com/radiuson/ADM1_RL.git
cd ADM1_RL
pip install -e .
```

Requirements: Python ≥ 3.9, PyTorch ≥ 2.0, stable-baselines3 ≥ 2.0, gymnasium ≥ 0.29.

---

## Repository Structure

```
env/                         # ADM1 Gymnasium environment
  adm1_gym_env.py            #   ADM1Env_v2 (main environment class)
  adm1_solver.py             #   Temperature-extended ADM1 ODE solver (40-state)
  scenario_manager.py        #   Scenario loading and disturbance injection
  scenarios.yaml             #   Scenario definitions and thermal parameters
  data/                      #   Influent time-series data (15-min resolution)

training/                    # SAC training
  train_sac.py               #   Standalone single-run training script
  run_experiment.py          #   Full experiment runner (train + eval + figures)
  reward_configs.py          #   Reward configuration presets
  configs/                   #   YAML experiment configurations

baselines/                   # Engineering baseline controllers
  baseline_controllers.py    #   Constant, PID, CascadedPID (BaseController ABC)
  mpc_controller.py
  nmpc_controller.py
  evaluate_baselines.py

evaluation/                  # Evaluation and metrics
  metrics_calculator.py      #   Per-episode metrics accumulator
  evaluate_rl_policy.py      #   Evaluate a trained SB3 model
  eval_ablation.py           #   Re-evaluate ablation reward variants

analysis/                    # Figure generation scripts
  plot_combo.py              #   Main results figure (score + violation rate)
  plot_generalization.py     #   Cross-scenario generalisation heatmap
  plot_ablation.py           #   Reward ablation figure
  plot_learning_curves.py    #   Training curves from TensorBoard logs

examples/                    # Custom controller examples
  my_controller.py           #   Documented template (BaseController subclass)
  custom_controller_tutorial.ipynb  # End-to-end tutorial notebook

scripts/                     # Re-evaluation and data utilities
  rerun_eval.py
  rerun_baseline_eval.py
  rerun_multi_eval.py

evaluate_controller.py       # Benchmark any custom controller on all scenarios

results/                     # Pre-computed evaluation data (included in repo)
  sac_single_scenario/       #   SAC single-scenario results (5 seeds, 2 obs modes)
  sac_multi_scenario/        #   SAC multi-scenario (domain-randomised) results
  ablation_linear_only/      #   Ablation: linear penalty only
  ablation_constant_only/    #   Ablation: constant penalty only
```

---

## Quick Start

### Option A — Use the pre-computed results directly

All per-run evaluation JSONs and TensorBoard training curves are included in `results/`.
You can regenerate all paper figures without retraining:

```bash
export RESULTS=./results

python analysis/plot_combo.py          --results-dir $RESULTS --output-dir $RESULTS/figures
python analysis/plot_generalization.py --results-dir $RESULTS --output-dir $RESULTS/figures
python analysis/plot_ablation.py       --results-dir $RESULTS --output-dir $RESULTS/figures
python analysis/plot_learning_curves.py --training-dir $RESULTS/sac_single_scenario/training \
    --seeds 42 123 456 789 1234 --output-dir $RESULTS/figures
```

### Option B — Test your own controller

Implement `BaseController`, run it against all six scenarios, and get a results table:

```bash
# 1. Copy and edit the template
cp examples/my_controller.py my_controller.py

# 2. Run the benchmark
python evaluate_controller.py --controller my_controller.py

# 3. Save results to JSON
python evaluate_controller.py --controller my_controller.py \
    --episodes 10 --output results/my_controller.json
```

See [`examples/my_controller.py`](examples/my_controller.py) for a fully documented template and
[`examples/custom_controller_tutorial.ipynb`](examples/custom_controller_tutorial.ipynb) for an
end-to-end notebook covering implementation, trajectory visualisation, SAC training, and comparison.

### Option C — Retrain from scratch

See [Reproducing the Paper](#reproducing-the-paper) below.

---

## Environment

### Observation space

**Full mode** (`obs_mode='full'`, 13-dim):

| idx | variable | unit | description |
|-----|----------|------|-------------|
| 0 | `total_vfa` | kg COD/m³ | S_ac + S_pro + S_bu + S_va |
| 1 | `alkalinity` | kmol/m³ | 0.8·S_IC + S_NH3 |
| 2 | `vfa_alk_ratio` | — | VFA / alkalinity |
| 3 | `S_h2` | kg COD/m³ | dissolved hydrogen |
| 4 | `pH` | — | reactor pH |
| 5 | `S_nh3` | kmol N/m³ | free ammonia |
| 6 | `S_IN` | kmol N/m³ | inorganic nitrogen |
| 7 | `X_ac` | kg COD/m³ | acetoclastic methanogen biomass |
| 8 | `X_h2` | kg COD/m³ | hydrogenotrophic methanogen biomass |
| 9 | `q_ch4` | m³/day | methane flow rate |
| 10 | `q_ad_current` | m³/day | feed flow (previous step) |
| 11 | `feed_mult_current` | — | feed multiplier (previous step) |
| 12 | `T_L_norm` | — | (T_liquid − 35 °C) / 10 |

**Compact mode** (`obs_mode='simple'`, 5-dim): indices `[4, 9, 12, 10, 11]` → `[pH, q_ch4, T_L_norm, q_ad, feed_mult]`.

**Extended observations** (`obs_extra`): append any of 36 additional ADM1 state variables:

```python
env = ADM1Env_v2(
    scenario_name='nominal',
    obs_extra=['S_ac', 'S_pro', 'S_va', 'S_bu',   # individual VFAs
               'q_co2',                              # CO2 outflow rate
               'T_a',                               # microbial adaptation temperature
               'S_gas_ch4', 'S_hco3_ion'],          # gas phase / acid-base
)
# observation shape: (13 + 8,) = (21,)
```

Full list of accepted `obs_extra` keys (call `ADM1Env_v2.AVAILABLE_EXTRA_OBS` at runtime):
individual VFAs (`S_va/bu/pro/ac`), degradable substrates (`S_su/aa/fa`), dissolved gases,
acid-base ions, gas-phase concentrations, intermediate biomass groups (`X_su/aa/fa/c4/pro`),
particulate matter (`X_xc/ch/pr/li/I`), temperatures (`T_L`, `T_a`), CO2 flow (`q_co2`).

### Action space

| dim | variable | unit | bounds |
|-----|----------|------|--------|
| 0 | `q_ad` | m³/day | [50, 300] |
| 1 | `feed_mult` | — | [0.7, 1.3] |
| 2 | `Q_HEX` | W | [−5000, 5000] |

Actions outside these bounds are clipped by the environment.

### Safety thresholds

| variable | soft violation | termination |
|----------|---------------|-------------|
| pH | < 6.8 or > 7.8 | < 5.8 |
| total_vfa | > 0.2 kg COD/m³ | > 0.8 |
| S_nh3 | > 0.002 kmol N/m³ | > 0.01 |

### Scenarios

| name | description |
|------|-------------|
| `nominal` | Steady-state baseline, no disturbances |
| `high_load` | Persistently elevated feed concentration |
| `low_load` | Persistently reduced feed concentration |
| `shock_load` | Step-increase in feed followed by recovery |
| `temperature_drop` | Gradual ambient temperature decline |
| `cold_winter` | Severe sustained cold (thermal stress) |

---

## Writing a Custom Controller

Subclass `BaseController` and implement two methods:

```python
from baselines.baseline_controllers import BaseController
import numpy as np

class MyController(BaseController):
    def __init__(self):
        super().__init__(name="MyController")

    def reset(self):
        """Called at the start of every episode. Clear integrators here."""
        self.step_count = 0

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        obs: shape (13,) for full mode, (5,) for simple mode
             + extra dims if obs_extra was set.
        Returns: [q_ad, feed_mult, Q_HEX]  (clipped automatically)
        """
        ph  = obs[4]    # pH
        vfa = obs[0]    # total VFA

        feed_mult = 1.0 - 0.2 * (7.2 - ph)   # simple P controller
        if vfa > 0.15:
            feed_mult = min(feed_mult, 0.85)   # safety clamp

        return np.array([180.0, feed_mult, 0.0], dtype=np.float32)
```

Then benchmark it:

```bash
python evaluate_controller.py --controller my_controller.py --episodes 10
```

---

## Reward Function

The safety-first reward combines:

- **Production**: q_CH4 / 2000 (normalised methane flow)
- **Safety penalty**: linear magnitude term + constant event-level penalty for pH, VFA, NH3 violations
- **Energy penalty**: −(q_ad / 300)² × 0.2
- **Stability penalty**: −|Δq_CH4| / 100 × 0.1

Ablation variants (`sf_linear_only`, `sf_constant_only`) remove one penalty term each.
All configurations are in [`training/reward_configs.py`](training/reward_configs.py).

---

## Reproducing the Paper

Set `RESULTS` to your preferred output directory:

```bash
export RESULTS=./results
```

### Stage 1 — Main SAC training (6 scenarios × 3–5 seeds × 2 obs modes)

```bash
python training/run_experiment.py \
    --config training/configs/experiment_config.yaml \
    --mode train --output-dir $RESULTS/sac_single_scenario

# Extra seeds for High Load, Shock Load, Cold Winter
python training/run_experiment.py \
    --config training/configs/train_extra_seeds.yaml \
    --output-dir $RESULTS/sac_single_scenario
```

### Stage 2 — Reward ablation training

```bash
python training/run_experiment.py \
    --config training/configs/reward_ablation.yaml \
    --output-dir $RESULTS/ablation_linear_only

python training/run_experiment.py \
    --config training/configs/reward_ablation_constant_only.yaml \
    --output-dir $RESULTS/ablation_constant_only
```

### Stage 3 — Multi-scenario (domain-randomised) training

```bash
python training/run_experiment.py \
    --config training/configs/multiscenario.yaml \
    --output-dir $RESULTS/sac_multi_scenario
```

### Stage 4 — Evaluation

```bash
# SAC per-run evaluation
python scripts/rerun_eval.py --results-dir $RESULTS

# Ablation evaluation
python evaluation/eval_ablation.py --results-dir $RESULTS

# Multi-scenario evaluation
python scripts/rerun_multi_eval.py --results-dir $RESULTS

# Engineering baselines
python scripts/rerun_baseline_eval.py --results-dir $RESULTS
```

### Stage 5 — Figures

```bash
python analysis/plot_combo.py          --results-dir $RESULTS --output-dir $RESULTS/figures
python analysis/plot_generalization.py --results-dir $RESULTS --output-dir $RESULTS/figures
python analysis/plot_ablation.py       --results-dir $RESULTS --output-dir $RESULTS/figures
python analysis/plot_learning_curves.py \
    --training-dir $RESULTS/sac_single_scenario/training \
    --seeds 42 123 456 789 1234 --output-dir $RESULTS/figures
```

---

## Pre-Computed Results

The `results/` directory contains all evaluation outputs used in the paper (model weights are not included):

| directory | contents |
|-----------|----------|
| `results/sac_single_scenario/evaluation/per_run/` | 324 per-run JSON files (SAC + MPC + NMPC, 5 seeds) |
| `results/sac_single_scenario/evaluation/baselines/` | 24 baseline controller JSON files |
| `results/sac_single_scenario/training/*/tensorboard/` | TensorBoard event files for all 48 SAC training runs |
| `results/sac_multi_scenario/evaluation/per_run/` | 60 multi-scenario SAC evaluation files |
| `results/ablation_linear_only/evaluation/per_run/` | 27 linear-only ablation files |
| `results/ablation_constant_only/evaluation/per_run/` | 27 constant-only ablation files |

---

## License

MIT

#!/bin/bash
# =============================================================================
# Paper Experiment Runner — ADM1_RL (Standard env, no thermal)
# =============================================================================
#
# Execution order:
#   Phase 1: Train 3 RL variants × 3 seeds  (≈ 3–5 hours total on GPU)
#   Phase 2: Evaluate all controllers × 6 scenarios
#   Phase 3: Counterfactual crisis comparison
#
# Usage:
#   cd /home/ihpc/code/biogas/ADM1_RL
#   bash scripts/run_paper_experiments.sh            # full run
#   bash scripts/run_paper_experiments.sh --eval-only  # skip training
#   bash scripts/run_paper_experiments.sh --phase1     # training only
#
# Output:
#   models_std_paper/   — trained SAC models
#   results/paper_eval/ — evaluation CSVs and JSONs
# =============================================================================

set -e   # exit on first error

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MODELS_DIR="models_std_paper_v2"
RESULTS_DIR="results/paper_eval_v2"
SEEDS="42 123 456"
OBS_MODE="scada"
TIMESTEPS=1000000

PHASE1=true
PHASE1_PPO=true
PHASE2=true
PHASE3=true

for arg in "$@"; do
  case $arg in
    --eval-only)   PHASE1=false; PHASE1_PPO=false ;;
    --phase1)      PHASE2=false; PHASE3=false; PHASE1_PPO=false ;;
    --phase1-ppo)  PHASE1=false; PHASE2=false; PHASE3=false ;;
    --phase2)      PHASE1=false; PHASE1_PPO=false; PHASE3=false ;;
    --phase3)      PHASE1=false; PHASE1_PPO=false; PHASE2=false ;;
    --no-ppo)      PHASE1_PPO=false ;;
  esac
done

echo "============================================================"
echo "  ADM1_RL Paper Experiments"
echo "  Working dir: $REPO_ROOT"
echo "  Models dir:  $MODELS_DIR"
echo "  Results dir: $RESULTS_DIR"
echo "  Obs mode:    $OBS_MODE"
echo "  Timesteps:   $TIMESTEPS"
echo "============================================================"

# ─── Phase 1: Training ────────────────────────────────────────────────────────
if $PHASE1; then
  echo ""
  echo "=== Phase 1: RL Training ==="
  echo "    3 configs × 3 seeds = 9 runs (~${TIMESTEPS} steps each)"
  echo ""

  for CONFIG in safety_first safety_target balanced; do
    for SEED in $SEEDS; do
      echo "--- Training: ${CONFIG}  seed=${SEED} ---"
      python training/train_sac_std_cur.py \
        --reward-config "$CONFIG" \
        --seed          "$SEED" \
        --timesteps     "$TIMESTEPS" \
        --obs-mode      "$OBS_MODE" \
        --output-dir    "$MODELS_DIR" \
        --verbose       1
      echo ""
    done
  done

  echo "Phase 1 complete. Models saved to: $MODELS_DIR"
fi

# ─── Phase 1-PPO: PPO Training ────────────────────────────────────────────────
if $PHASE1_PPO; then
  echo ""
  echo "=== Phase 1-PPO: PPO Training ==="
  echo "    3 configs × 3 seeds = 9 runs (~${TIMESTEPS} steps each)"
  echo "    (on-policy baseline; run after SAC Phase 1 completes)"
  echo ""

  for CONFIG in safety_first safety_target balanced; do
    for SEED in $SEEDS; do
      echo "--- PPO Training: ${CONFIG}  seed=${SEED} ---"
      python training/train_ppo_std_cur.py \
        --reward-config "$CONFIG" \
        --seed          "$SEED" \
        --timesteps     "$TIMESTEPS" \
        --obs-mode      "$OBS_MODE" \
        --output-dir    "$MODELS_DIR" \
        --verbose       1
      echo ""
    done
  done

  echo "Phase 1-PPO complete. PPO models saved to: $MODELS_DIR"
fi

# ─── Phase 2: Full Evaluation ─────────────────────────────────────────────────
if $PHASE2; then
  echo ""
  echo "=== Phase 2: Full Evaluation ==="
  echo "    Controllers: Constant, RuleBased, PID, CascadedPID, NMPC, Balanced-RL, SF-SAC, ST-SAC"
  echo "    Scenarios:   nominal, high_load, low_load, shock_load, high_load_real, pre_stressed"
  echo ""

  python evaluation/full_paper_eval.py \
    --output-dir   "$RESULTS_DIR" \
    --models-dir   "$MODELS_DIR" \
    --seeds        42 123 456 \
    --obs-mode     "$OBS_MODE" \
    --include-nmpc \
    --nmpc-horizon 8

  echo ""
  echo "Phase 2 complete. Results saved to: $RESULTS_DIR"
fi

# ─── Phase 3: Counterfactual Crisis Comparison ────────────────────────────────
if $PHASE3; then
  echo ""
  echo "=== Phase 3: Counterfactual Crisis Comparison ==="
  echo "    Scenario: sep2022_crisis (Muscatine Sep 2022 analog)"
  echo ""

  python evaluation/full_paper_eval.py \
    --output-dir    "${RESULTS_DIR}/crisis" \
    --models-dir    "$MODELS_DIR" \
    --scenario      sep2022_crisis \
    --seeds         42 \
    --obs-mode      "$OBS_MODE" \
    --include-nmpc \
    --nmpc-horizon  8 \
    --include-crisis

  echo ""
  echo "Phase 3 complete. Crisis results: ${RESULTS_DIR}/crisis"
fi

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  All phases complete."
echo "  Paper Table II data: ${RESULTS_DIR}/paper_eval_summary.csv"
echo "  Counterfactual data: ${RESULTS_DIR}/crisis/paper_eval_summary.csv"
echo "============================================================"

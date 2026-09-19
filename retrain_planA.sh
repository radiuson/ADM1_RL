#!/usr/bin/env bash
# =============================================================================
# Plan A Retraining Script
# Thresholds: soft VFA=0.30 kg COD/m³, hard VFA=1.20 kg COD/m³
#
# Phase 1: MS-SAC + ST-SAC  (6 runs × 500k steps, parallel)
# Phase 2: Ablation          (18 runs × 300k steps, parallel)
# Phase 3: Single-scenario   (36 runs × 300k steps, parallel batches)
# Phase 4: Baselines         (re-evaluate, fast)
#
# Usage (from ADM1_RL/ directory):
#   bash retrain_planA.sh            # all phases
#   bash retrain_planA.sh --phase1   # MS-SAC + ST-SAC only
#   bash retrain_planA.sh --phase2   # ablation only
#   bash retrain_planA.sh --phase3   # single-scenario only
#   bash retrain_planA.sh --phase4   # baselines only
# =============================================================================

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
LOG_DIR="${ROOT}/logs/planA_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "======================================================="
echo "  Plan A Retraining  (soft=0.30, hard=1.20 kg COD/m³)"
echo "  Log dir: $LOG_DIR"
echo "======================================================="

# ── Parse --phaseN args ──────────────────────────────────────────────────────
RUN_ALL=true
RUN_P1=false; RUN_P2=false; RUN_P3=false; RUN_P4=false
for arg in "$@"; do
    case "$arg" in
        --phase1) RUN_P1=true; RUN_ALL=false ;;
        --phase2) RUN_P2=true; RUN_ALL=false ;;
        --phase3) RUN_P3=true; RUN_ALL=false ;;
        --phase4) RUN_P4=true; RUN_ALL=false ;;
    esac
done
if $RUN_ALL; then RUN_P1=true; RUN_P2=true; RUN_P3=true; RUN_P4=true; fi

wait_for_pids() {
    local label="$1"; shift
    local pids=("$@")
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "  [FAIL] PID $pid failed" >&2
            failed=$((failed + 1))
        fi
    done
    if [[ $failed -gt 0 ]]; then
        echo "  [WARN] $failed job(s) failed in '$label'" >&2
    else
        echo "  [OK]   '$label' complete"
    fi
}

# =============================================================================
# Phase 1: MS-SAC (safety_first) + ST-SAC (safety_target), uniform_random, 500k
# =============================================================================
if $RUN_P1; then
    echo ""
    echo "### Phase 1: MS-SAC + ST-SAC (6 parallel runs × 500k steps) ###"
    pids=()
    for reward in safety_first safety_target; do
        for seed in 42 123 456; do
            tag="${reward}_seed${seed}"
            logf="${LOG_DIR}/p1_${tag}.log"
            echo "  Launching ${tag} → $logf"
            $PY training/train_sac_scenario_cur.py \
                --reward-config "$reward" \
                --stages uniform_random \
                --timesteps 500000 \
                --seed "$seed" \
                --output-dir models \
                --device auto \
                >"$logf" 2>&1 &
            pids+=($!)
        done
    done
    wait_for_pids "Phase 1" "${pids[@]}"
    echo "### Phase 1 done ###"
fi

# =============================================================================
# Phase 2: Reward ablation (sf_linear_only + sf_constant_only)
# 3 scenarios × 3 seeds × 2 configs = 18 runs at 300k steps each
# =============================================================================
if $RUN_P2; then
    echo ""
    echo "### Phase 2: Ablation (18 parallel runs × 300k steps) ###"
    # Run both ablation configs via run_experiment.py (handles scenario loop internally)
    pids=()
    for cfg in reward_ablation.yaml reward_ablation_constant_only.yaml; do
        logf="${LOG_DIR}/p2_${cfg%.yaml}.log"
        echo "  Launching $cfg → $logf"
        $PY training/run_experiment.py \
            --config "training/configs/$cfg" \
            --mode train_and_eval \
            >"$logf" 2>&1 &
        pids+=($!)
    done
    wait_for_pids "Phase 2" "${pids[@]}"
    echo "### Phase 2 done ###"
fi

# =============================================================================
# Phase 3: Single-scenario SAC (6 scenarios × 3 seeds × 2 obs = 36 runs × 300k)
# Run in batches of 12 to avoid over-saturating 24 cores
# =============================================================================
if $RUN_P3; then
    echo ""
    echo "### Phase 3: Single-scenario SAC (36 runs in batches) ###"
    SCENARIOS=(nominal high_load shock_load low_load temperature_drop cold_winter)
    SEEDS=(42 123 456)
    OBS_MODES=(full simple)
    BATCH_SIZE=12
    pids=()
    count=0
    for scenario in "${SCENARIOS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for obs in "${OBS_MODES[@]}"; do
                tag="${scenario}_seed${seed}_${obs}"
                logf="${LOG_DIR}/p3_${tag}.log"
                echo "  Launching $tag → $logf"
                obs_suffix=""
                [[ "$obs" == "simple" ]] && obs_suffix="--obs-mode simple"
                # shellcheck disable=SC2086
                $PY training/train_sac.py \
                    --scenario "$scenario" \
                    --seed "$seed" \
                    --reward-config safety_first \
                    --timesteps 300000 \
                    --output-dir results/sac_single_scenario/training \
                    $obs_suffix \
                    >"$logf" 2>&1 &
                pids+=($!)
                count=$((count + 1))
                if [[ $count -ge $BATCH_SIZE ]]; then
                    wait_for_pids "Phase 3 batch" "${pids[@]}"
                    pids=()
                    count=0
                fi
            done
        done
    done
    if [[ ${#pids[@]} -gt 0 ]]; then
        wait_for_pids "Phase 3 final batch" "${pids[@]}"
    fi
    echo "### Phase 3 done ###"
fi

# =============================================================================
# Phase 4: Baseline re-evaluation (deterministic, fast)
# =============================================================================
if $RUN_P4; then
    echo ""
    echo "### Phase 4: Baseline re-evaluation ###"
    logf="${LOG_DIR}/p4_baselines.log"
    $PY evaluation/full_evaluation.py \
        --output-dir results/baselines_planA \
        >"$logf" 2>&1
    echo "  Baselines saved to results/baselines_planA"
    echo "### Phase 4 done ###"
fi

echo ""
echo "======================================================="
echo "  All requested phases complete."
echo "  Logs: $LOG_DIR"
echo "======================================================="

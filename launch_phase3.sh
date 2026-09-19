#!/usr/bin/env bash
# Waits for Phase 1+2 PIDs to finish, then launches Phase 3 single-scenario SAC
# Run as:  bash launch_phase3.sh [phase1_pid1 phase1_pid2 ...] &

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=python3
LOG_DIR="${ROOT}/logs/planA_$(date +%Y%m%d)"
mkdir -p "$LOG_DIR"

# Wait for all PIDs passed as args (Phase 1 + Phase 2 jobs)
if [[ $# -gt 0 ]]; then
    echo "Waiting for ${#} background PIDs to finish before Phase 3..."
    for pid in "$@"; do
        wait "$pid" 2>/dev/null || true
    done
    echo "All watched PIDs done. Starting Phase 3."
else
    # If no PIDs given, wait until the 24 known jobs finish
    echo "No PIDs given — waiting until fewer than 5 training jobs remain..."
    while true; do
        n=$(ps aux | grep "train_sac" | grep -v grep | wc -l || echo 0)
        if [[ "$n" -lt 5 ]]; then break; fi
        echo "  $(date +%H:%M) — $n jobs still running..."
        sleep 300
    done
    echo "Training jobs finished. Starting Phase 3."
fi

echo ""
echo "### Phase 3: Single-scenario SAC (36 parallel runs × 300k steps) ###"
SCENARIOS=(nominal high_load shock_load low_load temperature_drop cold_winter)
SEEDS=(42 123 456)
OBS_MODES=(full simple)

pids=()
for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for obs in "${OBS_MODES[@]}"; do
            tag="${scenario}_seed${seed}_${obs}"
            logf="${LOG_DIR}/p3_${tag}.log"
            echo "  Launching $tag"
            obs_flag=""
            [[ "$obs" == "simple" ]] && obs_flag="--obs-mode simple"
            # shellcheck disable=SC2086
            nohup $PY training/train_sac.py \
                --scenario "$scenario" \
                --seed "$seed" \
                --reward-config safety_first \
                --timesteps 300000 \
                --output-dir results/sac_single_scenario_planA/training \
                $obs_flag \
                --device auto \
                >"$logf" 2>&1 &
            pids+=($!)
        done
    done
done

echo "Phase 3: ${#pids[@]} jobs launched. Waiting..."
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=$((failed + 1))
done
[[ $failed -gt 0 ]] && echo "[WARN] $failed Phase 3 jobs failed" || echo "[OK] Phase 3 complete"

echo ""
echo "### Phase 4: Baseline re-evaluation ###"
$PY evaluation/full_evaluation.py \
    --output-dir results/baselines_planA \
    >"${LOG_DIR}/p4_baselines.log" 2>&1
echo "Baselines saved → results/baselines_planA"

echo "All phases done."

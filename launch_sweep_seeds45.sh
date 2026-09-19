#!/usr/bin/env bash
# Add seeds 4 and 5 to all existing sweep configs (22 configs × 2 seeds = 44 runs)
# Purpose: reduce seed variance for cold_winter and other noisy scenarios.

set -euo pipefail
cd "$(dirname "$0")"

LOGDIR="logs/sweep"
OUTDIR="models_sweep"
STAGES="uniform_random"
TIMESTEPS=300000
DEVICE="cpu"
MAX_JOBS=6

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$LOGDIR" "$OUTDIR"

ALL_CONFIGS=(
    sweep_w2
    sweep_w3
    sweep_w5
    sweep_w10
    sweep_w11
    sweep_w12
    sweep_w13
    sweep_w14
    sweep_w15
    sweep_w16
    sweep_w18
    sweep_w22
    sweep_w25
    sweep_w30
    sweep_w40
    sweep_w50
    sweep_w60
    sweep_w20c025
    sweep_w20c050
    sweep_w20c100
    sweep_w20c150
    sweep_w20c200
)

SEEDS=(4 5)

run_one() {
    local cfg="$1" seed="$2"
    local logfile="$LOGDIR/${cfg}_seed${seed}.log"
    # Skip if final model already exists
    local run_dir="$OUTDIR/sac_scenario_cur_${cfg}_uniform_random_seed${seed}"
    if [ -f "$run_dir/final_model.zip" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP   ${cfg} seed${seed} (already done)"
        return
    fi
    echo "[$(date '+%H:%M:%S')] START  ${cfg} seed${seed}"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python3 -u training/train_sac_scenario_cur.py \
        --reward-config "$cfg" \
        --seed "$seed" \
        --stages "$STAGES" \
        --timesteps "$TIMESTEPS" \
        --output-dir "$OUTDIR" \
        --device "$DEVICE" \
        --verbose 1 \
        --eval-freq 50000 \
        --n-eval-episodes 3 \
        > "$logfile" 2>&1
    echo "[$(date '+%H:%M:%S')] DONE   ${cfg} seed${seed}"
}
export -f run_one
export LOGDIR OUTDIR STAGES TIMESTEPS DEVICE

JOBS=()
for cfg in "${ALL_CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        JOBS+=("${cfg}:${seed}")
    done
done

echo "==== Sweep Seeds 4+5 ===="
echo "  Configs : ${#ALL_CONFIGS[@]}"
echo "  Seeds   : 4, 5"
echo "  Total   : ${#JOBS[@]} runs"
echo "  Parallel: $MAX_JOBS"
echo "  Steps   : $TIMESTEPS"
echo "  Started : $(date)"
echo "========================="

ACTIVE=0
for job in "${JOBS[@]}"; do
    cfg="${job%%:*}"
    seed="${job##*:}"
    while (( ACTIVE >= MAX_JOBS )); do
        wait -n 2>/dev/null && ACTIVE=$((ACTIVE-1)) || true
    done
    run_one "$cfg" "$seed" &
    ACTIVE=$((ACTIVE+1))
done

wait
echo ""
echo "==== All ${#JOBS[@]} runs complete ==== $(date)"

#!/usr/bin/env bash
# VFA penalty hyperparameter sweep — train all (w, c) grid points
# Axis 1: c_VFA=0, w_VFA ∈ {2,3,5,10,15,20(ref),25,30,40,50,60}
# Axis 2: w_VFA=20, c_VFA ∈ {0.25,0.5,1.0,1.5,2.0}
# 15 new configs × 3 seeds = 45 runs (reference safety_target already trained)
# Max 8 parallel jobs.

set -euo pipefail
cd "$(dirname "$0")"

LOGDIR="logs/sweep"
OUTDIR="models_sweep"
STAGES="uniform_random"
TIMESTEPS=300000
DEVICE="cpu"
MAX_JOBS=6

# Prevent numpy/scipy/PyTorch from spawning per-process thread pools
# that over-subscribe CPUs and deadlock under parallel training.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$LOGDIR" "$OUTDIR"

# ---- configs to sweep (must match keys in REWARD_CONFIGS) ----
W_CONFIGS=(
    sweep_w2
    sweep_w3
    sweep_w5
    sweep_w10
    sweep_w15
    sweep_w25
    sweep_w30
    sweep_w40
    sweep_w50
    sweep_w60
)
C_CONFIGS=(
    sweep_w20c025
    sweep_w20c050
    sweep_w20c100
    sweep_w20c150
    sweep_w20c200
)
ALL_CONFIGS=("${W_CONFIGS[@]}" "${C_CONFIGS[@]}")

SEEDS=(1 2 3)

run_one() {
    local cfg="$1" seed="$2"
    local logfile="$LOGDIR/${cfg}_seed${seed}.log"
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

# Build job list
JOBS=()
for cfg in "${ALL_CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        JOBS+=("${cfg}:${seed}")
    done
done

echo "==== VFA Sweep Training ===="
echo "  Configs : ${#ALL_CONFIGS[@]}  (${#W_CONFIGS[@]} w-axis + ${#C_CONFIGS[@]} c-axis)"
echo "  Seeds   : ${#SEEDS[@]}"
echo "  Total   : ${#JOBS[@]} runs"
echo "  Parallel: $MAX_JOBS"
echo "  Steps   : $TIMESTEPS"
echo "  Started : $(date)"
echo "============================"

ACTIVE=0
PIDS=()

for job in "${JOBS[@]}"; do
    cfg="${job%%:*}"
    seed="${job##*:}"

    # Throttle
    while (( ACTIVE >= MAX_JOBS )); do
        wait -n 2>/dev/null && ACTIVE=$((ACTIVE-1)) || true
    done

    run_one "$cfg" "$seed" &
    PIDS+=($!)
    ACTIVE=$((ACTIVE+1))
done

# Wait for remaining
wait
echo ""
echo "==== All ${#JOBS[@]} sweep runs complete ==== $(date)"

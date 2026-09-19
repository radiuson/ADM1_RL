#!/usr/bin/env bash
# Dense w-axis sweep: w ∈ {11,12,13,14,16,18,22}
# Fill transition zone (w=10→15) and mid zone (w=15→25) for curve fitting.
# 7 configs × 3 seeds = 21 runs, max 6 parallel.

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

DENSE_CONFIGS=(
    sweep_w11
    sweep_w12
    sweep_w13
    sweep_w14
    sweep_w16
    sweep_w18
    sweep_w22
)

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

JOBS=()
for cfg in "${DENSE_CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        JOBS+=("${cfg}:${seed}")
    done
done

echo "==== Dense w-axis Sweep ===="
echo "  Configs : ${#DENSE_CONFIGS[@]}  (w=11,12,13,14,16,18,22)"
echo "  Seeds   : ${#SEEDS[@]}"
echo "  Total   : ${#JOBS[@]} runs"
echo "  Parallel: $MAX_JOBS"
echo "  Steps   : $TIMESTEPS"
echo "  Started : $(date)"
echo "============================"

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
echo "==== All ${#JOBS[@]} dense sweep runs complete ==== $(date)"

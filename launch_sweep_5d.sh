#!/usr/bin/env bash
# 5D (simple obs) sweep — same 22 configs × 5 seeds as 13D sweep
# obs_mode: simple (5-dim: pH, q_ch4, T_L_norm, q_ad, feed_mult)
# Output: models_sweep_5d/

set -euo pipefail
cd "$(dirname "$0")"

LOGDIR="logs/sweep_5d"
OUTDIR="models_sweep_5d"
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

SEEDS=(1 2 3 4 5)

run_one() {
    local cfg="$1" seed="$2"
    local run_dir="$OUTDIR/sac_scenario_cur_${cfg}_uniform_random_seed${seed}"
    local logfile="$LOGDIR/${cfg}_seed${seed}.log"
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
        --obs-mode simple \
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

echo "==== 5D Sweep Training ===="
echo "  obs_mode: simple (5-dim)"
echo "  Configs : ${#ALL_CONFIGS[@]}"
echo "  Seeds   : ${SEEDS[*]}"
echo "  Total   : ${#JOBS[@]} runs"
echo "  Parallel: $MAX_JOBS"
echo "  Steps   : $TIMESTEPS"
echo "  Output  : $OUTDIR"
echo "  Started : $(date)"
echo "==========================="

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
echo "==== All ${#JOBS[@]} 5D sweep runs complete ==== $(date)"

#!/usr/bin/env bash
# start_ablation_when_ready.sh
# Called by cron every 20 min. Starts ablation once Experiment 3 finishes.
#
# Experiment 3 is done when:
#   - run_experiment3.py process (PID 151349) is gone, AND
#   - 36 model dirs exist (18 PPO + 18 SAC-curriculum)
#
# Ablation is already running if run_ablation.py process exists.
# Ablation is done if flag file /tmp/ablation_started.flag exists.

set -euo pipefail

MODELS_DIR="/home/ihpc/code/biogas/ADM1_RL/models"
ADM1_RL_DIR="/home/ihpc/code/biogas/ADM1_RL"
LOG="/tmp/ablation_auto.log"
FLAG="/tmp/ablation_started.flag"

echo "[$(date '+%H:%M:%S')] Cron check fired" >> "$LOG"

# ── 1. Skip if ablation already started ──────────────────────────────────────
if [ -f "$FLAG" ]; then
    echo "[$(date '+%H:%M:%S')] Already started (flag exists) — skipping" >> "$LOG"
    exit 0
fi

# ── 2. Skip if ablation already running ──────────────────────────────────────
if pgrep -f "run_ablation.py" > /dev/null 2>&1; then
    echo "[$(date '+%H:%M:%S')] run_ablation.py already running — skipping" >> "$LOG"
    touch "$FLAG"
    exit 0
fi

# ── 3. Check if Experiment 3 is still running ─────────────────────────────────
if pgrep -f "run_experiment3.py" > /dev/null 2>&1; then
    # Count completed models so far
    PPO_COUNT=$(ls "$MODELS_DIR" 2>/dev/null | grep -c "^ppo_" || true)
    CUR_COUNT=$(ls "$MODELS_DIR" 2>/dev/null | grep -c "_curriculum$" || true)
    echo "[$(date '+%H:%M:%S')] Experiment 3 still running (ppo=$PPO_COUNT/18, curriculum=$CUR_COUNT/18)" >> "$LOG"
    exit 0
fi

# ── 4. Experiment 3 process gone — verify completeness ───────────────────────
PPO_COUNT=$(ls "$MODELS_DIR" 2>/dev/null | grep -c "^ppo_" || true)
CUR_COUNT=$(ls "$MODELS_DIR" 2>/dev/null | grep -c "_curriculum$" || true)

if [ "$PPO_COUNT" -lt 18 ] || [ "$CUR_COUNT" -lt 18 ]; then
    echo "[$(date '+%H:%M:%S')] Experiment 3 process gone but incomplete (ppo=$PPO_COUNT, curriculum=$CUR_COUNT) — waiting" >> "$LOG"
    exit 0
fi

# ── 5. Experiment 3 complete → start ablation ────────────────────────────────
echo "[$(date '+%H:%M:%S')] Experiment 3 COMPLETE (ppo=$PPO_COUNT, curriculum=$CUR_COUNT)" >> "$LOG"
echo "[$(date '+%H:%M:%S')] Starting ablation training..." >> "$LOG"

touch "$FLAG"

source /home/ihpc/anaconda3/etc/profile.d/conda.sh
conda activate biosim

cd "$ADM1_RL_DIR"
nohup python3 training/run_ablation.py --device cuda \
    >> /tmp/ablation.log 2>&1 &

echo "[$(date '+%H:%M:%S')] Ablation started (PID=$!)" >> "$LOG"

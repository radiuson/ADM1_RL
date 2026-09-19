#!/usr/bin/env bash
# 每20分钟由cron调用，ablation完成后自动开始评估

MODELS_DIR="/home/ihpc/code/biogas/ADM1_RL/models"
ADM1_DIR="/home/ihpc/code/biogas/ADM1_RL"
LOG="/tmp/eval_auto.log"
FLAG="/tmp/eval_started.flag"

echo "[$(date '+%H:%M:%S')] cron check" >> "$LOG"

[ -f "$FLAG" ] && { echo "[$(date '+%H:%M:%S')] 已完成，跳过" >> "$LOG"; exit 0; }

if pgrep -f "run_ablation.py" > /dev/null 2>&1; then
    PH3=$(ls "$MODELS_DIR" | grep 'safety_first_seed[0-9]*_curriculum$' | wc -l)
    PH4=$(ls "$MODELS_DIR" | grep 'safety_first_curriculum_seed[0-9]*$' | wc -l | tr -d ' ')
    echo "[$(date '+%H:%M:%S')] ablation仍在运行 (phase3=$PH3/18, phase4=$PH4/18)" >> "$LOG"
    exit 0
fi

PH3=$(ls "$MODELS_DIR" | grep 'safety_first_seed[0-9]*_curriculum$' | wc -l | tr -d ' ')
PH4=$(ls "$MODELS_DIR" | grep 'safety_first_curriculum_seed[0-9]*$' | wc -l | tr -d ' ')

if [ "$PH3" -lt 18 ] || [ "$PH4" -lt 18 ]; then
    echo "[$(date '+%H:%M:%S')] ablation未完成 (phase3=$PH3, phase4=$PH4)" >> "$LOG"
    exit 0
fi

echo "[$(date '+%H:%M:%S')] ablation完成 (phase3=$PH3, phase4=$PH4)，启动评估..." >> "$LOG"
touch "$FLAG"

cd "$ADM1_DIR"

# 顺序执行：phase3 → phase4 → ppo_cur_fk（避免并行写JSON）
nohup bash -c '
    cd /home/ihpc/code/biogas/ADM1_RL
    python3 scripts/eval_all_new_models.py --only phase3 >> /tmp/eval_phase3.log 2>&1
    echo "[DONE] phase3 eval" >> /tmp/eval_auto.log
    python3 scripts/eval_all_new_models.py --only phase4 >> /tmp/eval_phase4.log 2>&1
    echo "[DONE] phase4 eval" >> /tmp/eval_auto.log
    python3 scripts/eval_all_new_models.py --only ppo_cur_fk >> /tmp/eval_ppo_cur_fk.log 2>&1
    echo "[DONE] ppo_cur_fk eval" >> /tmp/eval_auto.log
' >> /tmp/eval_chain.log 2>&1 &

echo "[$(date '+%H:%M:%S')] 评估链已启动 (phase3→phase4→ppo_cur_fk)" >> "$LOG"

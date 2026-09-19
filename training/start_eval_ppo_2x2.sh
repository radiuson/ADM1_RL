#!/usr/bin/env bash
# 训练完成后自动触发 PPO 2x2 评估（ppo_cur_only → ppo_fk_only）

MODELS_DIR="/home/ihpc/code/biogas/ADM1_RL/models"
ADM1_DIR="/home/ihpc/code/biogas/ADM1_RL"
LOG="/tmp/eval_ppo2x2.log"
FLAG="/tmp/eval_ppo2x2.flag"

echo "[$(date '+%H:%M:%S')] cron check" >> "$LOG"

[ -f "$FLAG" ] && { echo "[$(date '+%H:%M:%S')] 已完成，跳过" >> "$LOG"; exit 0; }

# 检查训练进程
if pgrep -f "run_ppo_cur_only\|run_ppo_fk_only" > /dev/null 2>&1; then
    CUR=$(ls "$MODELS_DIR" | grep 'ppo_.*safety_first_seed[0-9]*_curriculum$' | wc -l | tr -d ' ')
    FK=$(ls "$MODELS_DIR"  | grep 'ppo_.*safety_first_curriculum_seed[0-9]*$'  | wc -l | tr -d ' ')
    echo "[$(date '+%H:%M:%S')] 训练仍在运行 (cur_only=$CUR/18, fk_only=$FK/18)" >> "$LOG"
    exit 0
fi

CUR=$(ls "$MODELS_DIR" | grep 'ppo_.*safety_first_seed[0-9]*_curriculum$' | wc -l | tr -d ' ')
FK=$(ls "$MODELS_DIR"  | grep 'ppo_.*safety_first_curriculum_seed[0-9]*$'  | wc -l | tr -d ' ')

if [ "$CUR" -lt 18 ] || [ "$FK" -lt 18 ]; then
    echo "[$(date '+%H:%M:%S')] 训练未完成 (cur_only=$CUR/18, fk_only=$FK/18)" >> "$LOG"
    exit 0
fi

echo "[$(date '+%H:%M:%S')] 训练完成 (cur_only=$CUR, fk_only=$FK)，启动评估..." >> "$LOG"
touch "$FLAG"

cd "$ADM1_DIR"
nohup bash -c '
    cd /home/ihpc/code/biogas/ADM1_RL
    python3 scripts/eval_all_new_models.py --only ppo_cur_only >> /tmp/eval_ppo_cur_only.log 2>&1
    echo "[DONE] ppo_cur_only eval" >> /tmp/eval_ppo2x2.log
    python3 scripts/eval_all_new_models.py --only ppo_fk_only >> /tmp/eval_ppo_fk_only.log 2>&1
    echo "[DONE] ppo_fk_only eval" >> /tmp/eval_ppo2x2.log
' >> /tmp/eval_ppo2x2_chain.log 2>&1 &

echo "[$(date '+%H:%M:%S')] 评估链已启动 (ppo_cur_only→ppo_fk_only)" >> "$LOG"

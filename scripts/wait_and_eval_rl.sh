#!/usr/bin/env bash
# PPO 완료(72개 "Model saved →") 감지 후 RL 전체 평가 자동 실행
# 사용법: bash scripts/wait_and_eval_rl.sh  (ADM1_RL/ 디렉토리에서)

LOG_FILE=/tmp/train_ppo_parallel.log
EVAL_LOG=/tmp/eval_rl_models.log
TARGET=72
CHECK_INTERVAL=300   # 5분마다 확인

echo "[WAIT] PPO 완료 대기 시작: $(date)" | tee "$EVAL_LOG"
echo "[WAIT] 목표: $TARGET 개 'Model saved →' in $LOG_FILE" | tee -a "$EVAL_LOG"

while true; do
    if [ ! -f "$LOG_FILE" ]; then
        echo "[WAIT] 로그 파일 없음: $LOG_FILE, 대기 중..." | tee -a "$EVAL_LOG"
        sleep "$CHECK_INTERVAL"
        continue
    fi

    COUNT=$(grep "Model saved →" "$LOG_FILE" | grep -v "Best model" | wc -l)
    echo "[WAIT] $(date '+%H:%M:%S') — 완료된 PPO 작업: $COUNT / $TARGET" | tee -a "$EVAL_LOG"

    if [ "$COUNT" -ge "$TARGET" ]; then
        echo "[WAIT] PPO 훈련 완료! ($COUNT 개) → RL 평가 시작: $(date)" | tee -a "$EVAL_LOG"
        break
    fi

    sleep "$CHECK_INTERVAL"
done

cd "$(dirname "$0")/.." || exit 1

echo "[EVAL] RL 모델 전체 평가 시작: $(date)" | tee -a "$EVAL_LOG"
python scripts/eval_all_new_models.py --only all --results-dir results/evaluation_60d 2>&1 | tee -a "$EVAL_LOG"
echo "[EVAL] RL 평가 완료: $(date)" | tee -a "$EVAL_LOG"

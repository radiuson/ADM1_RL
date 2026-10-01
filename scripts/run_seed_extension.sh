#!/bin/bash
# 扩充种子到 61（共 20 个）。每轮跑完立即评估。
# 所有产物写在项目内：日志 logs_campaign/，评估 results_ext/。
# 论文数据仍是 results_frozen_20260915/，本脚本不触碰它。
# 停止：touch <项目>/STOP_EXTENSION
cd "$(dirname "$0")/.."
ROOT=$(pwd)
BIO=/home/ihpc/anaconda3/envs/biosim/bin/python3
SAFE=/home/ihpc/anaconda3/envs/adm1_safe/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs_campaign results_ext/evres results_ext/evcmdp

for SEED in $(seq 53 61); do
  [ -f "$ROOT/STOP_EXTENSION" ] && { echo "[$(date '+%m-%d %H:%M')] 停止于种子 $SEED 前"; break; }
  $BIO scripts/gen_seed_round.py $SEED "$ROOT/round_$SEED.txt"
  n=$(wc -l < "$ROOT/round_$SEED.txt"); [ "$n" -eq 0 ] && continue
  echo "[$(date '+%m-%d %H:%M')] 种子 $SEED 开始, $n 个作业"
  xargs -P 22 -d "\n" -I{} bash -c 'eval "$1"' _ {} < "$ROOT/round_$SEED.txt"
  echo "[$(date '+%m-%d %H:%M')] 种子 $SEED 训练完成，评估中"

  : > "$ROOT/ev_$SEED.txt"
  for d in models_seed/*/ models_mem/*/; do
    o="results_ext/evres/$(echo ${d} | tr '/' '_').json"
    [ -f "$d/final_model.zip" ] && [ ! -f "$o" ] \
      && echo "$BIO evaluation/eval_sb3.py $d/final_model.zip $o" >> "$ROOT/ev_$SEED.txt"
  done
  for D in models_cmdp/*/seed-*/; do
    mx=$(ls "$D"/torch_save/ 2>/dev/null | grep -oP 'epoch-\K\d+' | sort -n | tail -1)
    [ "${mx:-0}" -lt 50 ] && continue
    tag=$(echo "$D" | md5sum | cut -c1-10)
    [ -f "results_ext/evcmdp/h$tag.json" ] && continue
    echo "$SAFE evaluation/eval_cmdp.py '$D' results_ext/evcmdp/h$tag.json" >> "$ROOT/ev_$SEED.txt"
  done
  [ -s "$ROOT/ev_$SEED.txt" ] && xargs -P 22 -d "\n" -I{} bash -c 'eval "$1"' _ {} < "$ROOT/ev_$SEED.txt"
  echo "[$(date '+%m-%d %H:%M')] 种子 $SEED 完成 — 累计 evres $(ls results_ext/evres/*.json 2>/dev/null|wc -l) evcmdp $(ls results_ext/evcmdp/*.json 2>/dev/null|wc -l)"
done
echo "[$(date '+%m-%d %H:%M')] 扩充结束"

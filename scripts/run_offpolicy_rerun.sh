#!/bin/bash
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
while pgrep -f "train_sac_std_cur|train_omnisafe" > /dev/null; do sleep 120; done
echo "[$(date '+%m-%d %H:%M')] 离策略约束族重跑开始: $(wc -l < /tmp/rerun_off.txt) 个"
xargs -P 22 -d "\n" -I{} bash -c 'eval "$1"' _ {} < /tmp/rerun_off.txt
echo "[$(date '+%m-%d %H:%M')] 训练完成，评估中"
: > /tmp/ev_off.txt
for D in models_cmdp/*Lag-\{ADM1-v0\}/seed-*/ models_cmdp/SACPID-\{ADM1-v0\}/seed-*/; do
  [ -d "$D" ] || continue
  mx=$(ls "$D"/torch_save/ 2>/dev/null | grep -oP 'epoch-\K\d+' | sort -n | tail -1)
  [ "${mx:-0}" -lt 50 ] && continue
  tag=$(echo "$D" | md5sum | cut -c1-10)
  echo "/home/ihpc/anaconda3/envs/adm1_safe/bin/python evaluation/eval_cmdp.py '$D' results_ext/evcmdp/off$tag.json" >> /tmp/ev_off.txt
done
xargs -P 22 -d "\n" -I{} bash -c 'eval "$1"' _ {} < /tmp/ev_off.txt
echo "[$(date '+%m-%d %H:%M')] 完成 $(ls results_ext/evcmdp/off*.json 2>/dev/null|wc -l) 个评估"

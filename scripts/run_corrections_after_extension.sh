#!/bin/bash
# 严格串行：等种子扩充的调度器进程本身退出，再等训练进程清空，才启动修正队列。
# 上一版只检查训练进程，会在扩充的两轮之间穿过去，导致 89 个进程挤 12 个物理核。
cd "$(dirname "$0")/.."
ROOT=$(pwd)
echo "[$(date '+%m-%d %H:%M')] 等待种子扩充调度器退出"
while pgrep -f "[r]un_seed_extension.sh" > /dev/null; do sleep 180; done
echo "[$(date '+%m-%d %H:%M')] 扩充调度器已退出，等待训练进程清空"
while pgrep -f "[t]rain_sac_std_cur|[t]rain_omnisafe" > /dev/null; do sleep 120; done
rm -f "$ROOT/STOP_CORRECTIONS"
echo "[$(date '+%m-%d %H:%M')] 修正队列接管"
exec ./scripts/run_correction_queue.sh

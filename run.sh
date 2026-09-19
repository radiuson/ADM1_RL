#!/bin/bash
# 用正确的 conda 环境跑项目脚本，不依赖 shell 当前激活的环境。
#   ./run.sh py   <script> [args]   -> biosim   (SB3 / 环境 / 图)
#   ./run.sh safe <script> [args]   -> adm1_safe (omnisafe)
C=/home/ihpc/anaconda3/envs
case "$1" in
  py)   shift; exec env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $C/biosim/bin/python3 "$@" ;;
  safe) shift; exec env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $C/adm1_safe/bin/python "$@" ;;
  *) echo "用法: $0 {py|safe} <script> [args]"; exit 1 ;;
esac

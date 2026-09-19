#!/bin/bash
# 配置修正队列：四批串行，每批跑完即评估。
# 等当前种子轮次结束后开始，避免与扩充实验抢核（12 物理核）。
# 停止：touch <项目>/STOP_CORRECTIONS
cd "$(dirname "$0")/.."
ROOT=$(pwd)
BIO=/home/ihpc/anaconda3/envs/biosim/bin/python3
SAFE=/home/ihpc/anaconda3/envs/adm1_safe/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs_campaign results_ext/evres results_ext/evcmdp

echo "[$(date '+%m-%d %H:%M')] 等待当前作业结束"
while pgrep -f "train_sac_std_cur|train_omnisafe" > /dev/null; do sleep 120; done

run_batch () {   # $1=名称  $2=作业文件
  [ -f "$ROOT/STOP_CORRECTIONS" ] && { echo "[$(date '+%m-%d %H:%M')] 收到停止信号"; exit 0; }
  local n=$(wc -l < "$2")
  echo "[$(date '+%m-%d %H:%M')] === $1 开始: $n 个作业 ==="
  xargs -P 20 -d "\n" -I{} bash -c 'eval "$1"' _ {} < "$2"
  echo "[$(date '+%m-%d %H:%M')] === $1 训练完成 ==="
}

# ---- 批 1：离策略约束族，对偶超参对齐到在策略值 ----
: > "$ROOT/q1.txt"; i=0
for A in SACLag DDPGLag TD3Lag SACPID; do for C in 0.6 1.8 3.0 6.0; do for S in $(seq 42 51); do
  echo "sleep $((i*4)); $SAFE training/train_omnisafe.py --algo $A --cost-limit $C --seed $S --timesteps 150000 --output-dir models_cmdp --device cpu > logs_campaign/q1_${A}_c${C}_s${S}.log 2>&1" >> "$ROOT/q1.txt"
  i=$((i+1)); done; done; done
run_batch "批1 离策略约束族(对偶超参对齐)" "$ROOT/q1.txt"

# ---- 批 2：DDPG / TD3 加探索噪声 ----
: > "$ROOT/q2.txt"
for A in ddpg td3; do for W in lw0p5 lw1 lw2 lw5; do for S in $(seq 42 51); do
  echo "$BIO training/train_sac_std_cur.py --reward-config $W --seed $S --normalize --step-size 1.0 --timesteps 150000 --algo $A --output-dir models_noise --device cpu --verbose 0 > logs_campaign/q2_${A}_${W}_s${S}.log 2>&1" >> "$ROOT/q2.txt"
done; done; done
run_batch "批2 DDPG/TD3 加 action_noise" "$ROOT/q2.txt"

# ---- 批 3：A2C 对齐 n_steps 到 2048 ----
: > "$ROOT/q3.txt"
for W in lw0p5 lw1 lw2 lw5; do for S in $(seq 42 51); do
  echo "ADM1_ALIGN_NSTEPS=1 $BIO training/train_sac_std_cur.py --reward-config $W --seed $S --normalize --step-size 1.0 --timesteps 150000 --algo a2c --output-dir models_align --device cpu --verbose 0 > logs_campaign/q3_a2c_${W}_s${S}.log 2>&1" >> "$ROOT/q3.txt"
done; done
run_batch "批3 A2C n_steps=2048" "$ROOT/q3.txt"

# ---- 批 4：CrossQ 原生短预热 ----
: > "$ROOT/q4.txt"
for W in lw0p5 lw1 lw2 lw5; do for S in $(seq 42 51); do
  echo "ADM1_NATIVE_HP=1 $BIO training/train_sac_std_cur.py --reward-config $W --seed $S --normalize --step-size 1.0 --timesteps 150000 --algo crossq --output-dir models_native --device cpu --verbose 0 > logs_campaign/q4_crossq_${W}_s${S}.log 2>&1" >> "$ROOT/q4.txt"
done; done
run_batch "批4 CrossQ learning_starts=100" "$ROOT/q4.txt"

# ---- 统一评估 ----
echo "[$(date '+%m-%d %H:%M')] 评估全部修正批次"
: > "$ROOT/qev.txt"
for d in models_noise/*/ models_align/*/ models_native/*/; do
  [ -f "$d/final_model.zip" ] || continue
  o="results_ext/evres/$(echo ${d} | tr '/' '_').json"
  [ -f "$o" ] || echo "$BIO evaluation/eval_sb3.py $d/final_model.zip $o" >> "$ROOT/qev.txt"
done
for D in models_cmdp/*/seed-*/; do
  mx=$(ls "$D"/torch_save/ 2>/dev/null | grep -oP 'epoch-\K\d+' | sort -n | tail -1)
  [ "${mx:-0}" -lt 50 ] && continue
  tag=$(echo "$D" | md5sum | cut -c1-10)
  [ -f "results_ext/evcmdp/q$tag.json" ] || echo "$SAFE evaluation/eval_cmdp.py '$D' results_ext/evcmdp/q$tag.json" >> "$ROOT/qev.txt"
done
[ -s "$ROOT/qev.txt" ] && xargs -P 20 -d "\n" -I{} bash -c 'eval "$1"' _ {} < "$ROOT/qev.txt"

$BIO scripts/export_hyperparams.py ../ADM1/papers/mypaper/table_hyperparams.tex
echo "[$(date '+%m-%d %H:%M')] 全部完成 — evres $(ls results_ext/evres/*.json 2>/dev/null|wc -l) evcmdp $(ls results_ext/evcmdp/*.json 2>/dev/null|wc -l)"

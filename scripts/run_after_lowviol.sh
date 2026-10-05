#!/bin/bash
# Wait for the lowviol campaign, then evaluate it and start the 600k budget
# test.  The machine is already saturated at 20 workers, so the two cannot
# overlap without both running slowly.
#
# Stop cleanly: touch STOP_CHAIN
#   nohup bash scripts/run_after_lowviol.sh > logs_campaign/chain.log 2>&1 &
cd "$(dirname "$0")/.."
ROOT=$(pwd)
BIO=/home/ihpc/anaconda3/envs/biosim/bin/python3
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs_campaign

say() { echo "[$(date '+%m-%d %H:%M')] $*"; }

say "waiting for lowviol training to finish"
while pgrep -f 'bin/python3 training/train_sac_std_cur.py --reward-config lw10' > /dev/null \
   || pgrep -f 'bin/python3 training/train_sac_std_cur.py --reward-config lw20' > /dev/null; do
  [ -f "$ROOT/STOP_CHAIN" ] && { say "stopped before evaluation"; exit 0; }
  sleep 300
done
say "lowviol training done — $(find models_lowviol -name final_model.zip | wc -l)/168 models"

[ -f "$ROOT/STOP_CHAIN" ] && { say "stopped before evaluation"; exit 0; }
say "evaluating lowviol"
bash scripts/run_lowviol_eval.sh
say "lowviol analysis:"
$BIO scripts/analyse_lowviol.py results_lowviol/evres

[ -f "$ROOT/STOP_CHAIN" ] && { say "stopped before the 600k test"; exit 0; }
n=$(wc -l < "$ROOT/b600k_jobs.txt")
say "starting the 600k budget test — $n runs"
xargs -P 20 -d "\n" -I{} bash -c 'eval "$1"' _ {} < "$ROOT/b600k_jobs.txt"
say "600k training done — $(find models_budget600k -name final_model.zip 2>/dev/null | wc -l)/$n models"

say "evaluating the 600k runs"
mkdir -p results_b600k/evres
: > "$ROOT/b600k_eval.txt"
for d in models_budget600k/*/; do
  [ -f "$d/final_model.zip" ] || continue
  o="results_b600k/evres/$(basename "$d").json"
  [ -f "$o" ] && continue
  echo "$BIO evaluation/eval_sb3.py '$d/final_model.zip' '$o'" >> "$ROOT/b600k_eval.txt"
done
[ -s "$ROOT/b600k_eval.txt" ] && xargs -P 20 -d "\n" -I{} bash -c 'eval "$1"' _ {} < "$ROOT/b600k_eval.txt"
say "chain complete — results in results_lowviol/ and results_b600k/"

#!/bin/bash
# Evaluate the extended-weight runs once their training has finished.
#
# Writes to results_lowviol/evres/ rather than into the paper's archive, so
# the reported numbers are untouched until the new runs have been looked at.
# Skips any record already written, so it is safe to re-run.
#
#   bash scripts/run_lowviol_eval.sh
cd "$(dirname "$0")/.."
ROOT=$(pwd)
BIO=/home/ihpc/anaconda3/envs/biosim/bin/python3
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p results_lowviol/evres logs_campaign

JOBS="$ROOT/lowviol_eval.txt"
: > "$JOBS"
for d in models_lowviol/*/; do
  [ -f "$d/final_model.zip" ] || continue
  o="results_lowviol/evres/$(basename "$d").json"
  [ -f "$o" ] && continue
  echo "$BIO evaluation/eval_sb3.py '$d/final_model.zip' '$o'" >> "$JOBS"
done

n=$(wc -l < "$JOBS")
echo "[$(date '+%m-%d %H:%M')] $n policies to evaluate"
[ "$n" -eq 0 ] && { echo "nothing to do"; exit 0; }

xargs -P 20 -d "\n" -I{} bash -c 'eval "$1"' _ {} < "$JOBS"

got=$(ls results_lowviol/evres/*.json 2>/dev/null | wc -l)
echo "[$(date '+%m-%d %H:%M')] evaluation done — $got records in results_lowviol/evres"
echo "next: python3 scripts/analyse_lowviol.py results_lowviol/evres"

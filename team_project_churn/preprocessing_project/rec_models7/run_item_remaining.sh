#!/usr/bin/env bash
# 남은 item 실탐색: CatBoost(최적값) → Transformer → LogReg. (단축 v4cfg와 별개)
set -u
PY="/c/Users/playdata2/anaconda3/python.exe"
BASE="/c/Users/playdata2/Documents/ml_workspace/team_project_churn/preprocessing_project/rec_models7"
S="$BASE/models_rec_bayes.py"; TR="$BASE/models_rec_transformer.py"; RES="$BASE/results.tsv"
runtab () { echo ">>> $1/$2 (t=$3) $(date +%H:%M:%S)"; "$PY" "$S" "$1" "$2" "$3" 2>&1 | grep -E "RESULTREC|Error|ValueError|Traceback" | tee -a "$RES"; }
runtf  () { echo ">>> $1/Transformer $(date +%H:%M:%S)"; OMP_NUM_THREADS=8 "$PY" "$TR" "$1" 40 2>&1 | grep -E "RESULTREC|Error|Traceback" | tee -a "$RES"; }

runtab item CatBoost 2   # 실제 최적값 탐색(느리지만 진행)
runtf  item
runtab item LogReg   4
echo "=== ITEM REMAINING DONE $(date +%H:%M:%S) ==="

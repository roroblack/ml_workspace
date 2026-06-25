#!/usr/bin/env bash
# CatBoost 경량화(iter≤40, trial1) + cat 먼저. 남은 4개 빠르게.
set -u
PY="/c/Users/playdata2/anaconda3/python.exe"
BASE="/c/Users/playdata2/Documents/ml_workspace/team_project_churn/preprocessing_project/rec_models7"
S="$BASE/models_rec_bayes.py"; TR="$BASE/models_rec_transformer.py"; RES="$BASE/results.tsv"
runtab () { echo ">>> $1/$2 (t=$3) $(date +%H:%M:%S)"; "$PY" "$S" "$1" "$2" "$3" 2>&1 | grep -E "RESULTREC|Error|ValueError|Traceback" | tee -a "$RES"; }
runtf  () { echo ">>> $1/Transformer $(date +%H:%M:%S)"; OMP_NUM_THREADS=8 "$PY" "$TR" "$1" 40 2>&1 | grep -E "RESULTREC|Error|Traceback" | tee -a "$RES"; }

runtab cat  CatBoost 1   # 카테고리 먼저(경량)
runtab item CatBoost 1
runtf  item
runtab item LogReg   4
echo "=== DONE $(date +%H:%M:%S) ==="

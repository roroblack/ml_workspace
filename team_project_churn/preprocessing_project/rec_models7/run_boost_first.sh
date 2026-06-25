#!/usr/bin/env bash
# 부스트 우선(사용자 요청): 남은 부스트 3개 먼저 → 그다음 트랜스포머/LogReg
set -u
PY="/c/Users/playdata2/anaconda3/python.exe"
BASE="/c/Users/playdata2/Documents/ml_workspace/team_project_churn/preprocessing_project/rec_models7"
S="$BASE/models_rec_bayes.py"; TR="$BASE/models_rec_transformer.py"; RES="$BASE/results.tsv"
runtab () { echo ">>> $1/$2 (t=$3) $(date +%H:%M:%S)"; "$PY" "$S" "$1" "$2" "$3" 2>&1 | grep -E "RESULTREC|Error|ValueError|Traceback" | tee -a "$RES"; }
runtf  () { echo ">>> $1/Transformer $(date +%H:%M:%S)"; OMP_NUM_THREADS=8 "$PY" "$TR" "$1" 40 2>&1 | grep -E "RESULTREC|Error|Traceback" | tee -a "$RES"; }

# ① 남은 부스트 3개 먼저
runtab cat  XGBoost  2
runtab item CatBoost 2
runtab cat  CatBoost 2
# ② 그다음 트랜스포머 + LogReg
runtf  item
runtab item LogReg   4
echo "=== DONE $(date +%H:%M:%S) ==="

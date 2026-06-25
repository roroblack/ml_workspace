#!/usr/bin/env bash
# 남은 5개만 단독 실행: cat/XGBoost, cat/CatBoost, item/CatBoost, item/LogReg, item/Transformer
set -u
PY="/c/Users/playdata2/anaconda3/python.exe"
BASE="/c/Users/playdata2/Documents/ml_workspace/team_project_churn/preprocessing_project/rec_models7"
S="$BASE/models_rec_bayes.py"; TR="$BASE/models_rec_transformer.py"; RES="$BASE/results.tsv"
runtab () { echo ">>> $1/$2 (t=$3) $(date +%H:%M:%S)"; "$PY" "$S" "$1" "$2" "$3" 2>&1 | grep -E "RESULTREC|Error|ValueError|Traceback" | tee -a "$RES"; }
runtf  () { echo ">>> $1/Transformer $(date +%H:%M:%S)"; OMP_NUM_THREADS=8 "$PY" "$TR" "$1" 40 2>&1 | grep -E "RESULTREC|Error|Traceback" | tee -a "$RES"; }

runtf  item            # 트랜스포머 item (빠름)
runtab cat  XGBoost  2
runtab item CatBoost 2
runtab cat  CatBoost 2
runtab item LogReg   4  # 다중분류 multinomial이라 느림(마지막)
echo "=== MISSING DONE $(date +%H:%M:%S) ==="

#!/usr/bin/env bash
# 바운스(v4-3) 부스트 3종 — 이진. item 부스트 다음 우선순위.
set -u
PY="/c/Users/playdata2/anaconda3/python.exe"
S="/c/Users/playdata2/Documents/ml_workspace/team_project_churn/preprocessing_project/rec_models7/models_bin_bayes.py"
RES="/c/Users/playdata2/Documents/ml_workspace/team_project_churn/preprocessing_project/rec_models7/results_bin.tsv"
[ -f "$RES" ] || echo -e "dataset\tmodel\tresult" > "$RES"
runb () { echo ">>> $1/$2 (t=$3) $(date +%H:%M:%S)"; "$PY" "$S" "$1" "$2" "$3" 2>&1 | grep -E "RESULTBIN|Error|ValueError|Traceback" | tee -a "$RES"; }
runb bounce XGBoost  3
runb bounce LightGBM 3
runb bounce CatBoost 3
echo "=== BOUNCE BOOST DONE $(date +%H:%M:%S) ==="

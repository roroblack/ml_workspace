#!/usr/bin/env bash
# 사용자 지정 순서: ①v4-2(item) 부스트3종 → ②트랜스포머(v4-1·v4-2) → ③나머지(cat 부스트3종 + DT·LogReg·RF×둘)
set -u
PY="/c/Users/playdata2/anaconda3/python.exe"
BASE="/c/Users/playdata2/Documents/ml_workspace/team_project_churn/preprocessing_project/rec_models7"
S="$BASE/models_rec_bayes.py"; TR="$BASE/models_rec_transformer.py"; RES="$BASE/results.tsv"
echo -e "dataset\tmodel\tresult" > "$RES"
runtab () { echo ">>> $1/$2 (t=$3) $(date +%H:%M:%S)"; "$PY" "$S" "$1" "$2" "$3" 2>&1 | grep -E "RESULTREC|Error|ValueError|Traceback" | tee -a "$RES"; }
runtf  () { echo ">>> $1/Transformer $(date +%H:%M:%S)"; OMP_NUM_THREADS=8 "$PY" "$TR" "$1" 40 2>&1 | grep -E "RESULTREC|Error|Traceback" | tee -a "$RES"; }

# ① v4-2(item) 부스트 3종 (우선)
runtab item XGBoost  2
runtab item LightGBM 2
runtab item CatBoost 2
# ② 트랜스포머 v4-1, v4-2
runtf cat
runtf item
# ③ 나머지: cat 부스트 3종 + DT/LogReg/RF × cat,item
runtab cat XGBoost  2
runtab cat LightGBM 2
runtab cat CatBoost 2
for DS in cat item; do
  runtab $DS DecisionTree 6
  runtab $DS LogReg       6
  runtab $DS RandomForest 5
done
echo "=== ALL DONE $(date +%H:%M:%S) ==="

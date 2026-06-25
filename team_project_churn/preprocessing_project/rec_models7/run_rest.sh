#!/usr/bin/env bash
# 부스팅 3종 완료 후: 트랜스포머(v4-1·v4-2) → 나머지(DT·LogReg·RF) 순.
set -u
PY="/c/Users/playdata2/anaconda3/python.exe"
BASE="/c/Users/playdata2/Documents/ml_workspace/team_project_churn/preprocessing_project/rec_models7"
S="$BASE/models_rec_bayes.py"; TR="$BASE/models_rec_transformer.py"; RES="$BASE/results.tsv"
[ -f "$RES" ] || echo -e "dataset\tmodel\tresult" > "$RES"
runtab () { echo ">>> $1 / $2 (trials=$3) $(date +%H:%M:%S)"; "$PY" "$S" "$1" "$2" "$3" 2>&1 | grep -E "RESULTREC|Error|ValueError|Traceback" | tee -a "$RES"; }
runtf  () { echo ">>> $1 / Transformer $(date +%H:%M:%S)"; OMP_NUM_THREADS=6 "$PY" "$TR" "$1" 40 2>&1 | grep -E "RESULTREC|Error|Traceback" | tee -a "$RES"; }

# 1) 트랜스포머 (v4-1, v4-2)
runtf cat
runtf item
# 2) 나머지 정형 모델 (DT·LogReg·RF) × v4-1, v4-2
for DS in cat item; do
  runtab $DS DecisionTree 6
  runtab $DS LogReg       6
  runtab $DS RandomForest 5
done
echo "=== REST DONE $(date +%H:%M:%S) ==="

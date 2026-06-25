#!/usr/bin/env bash
# 우선순위: 부스팅 3종 × v4-1/v4-2 먼저 산출.
set -u
PY="/c/Users/playdata2/anaconda3/python.exe"
S="/c/Users/playdata2/Documents/ml_workspace/team_project_churn/preprocessing_project/rec_models7/models_rec_bayes.py"
RES="/c/Users/playdata2/Documents/ml_workspace/team_project_churn/preprocessing_project/rec_models7/results.tsv"
[ -f "$RES" ] || echo -e "dataset\tmodel\tresult" > "$RES"
run () {
  echo ">>> $1 / $2 (trials=$3) $(date +%H:%M:%S)"
  "$PY" "$S" "$1" "$2" "$3" 2>&1 | grep -E "RESULTREC|Error|ValueError|Traceback" | tee -a "$RES"
}
for DS in cat item; do
  run $DS XGBoost  3
  run $DS LightGBM 3
  run $DS CatBoost 3
done
echo "=== BOOST DONE $(date +%H:%M:%S) ==="

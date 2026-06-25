#!/usr/bin/env bash
# v4-1/v4-2 추천 6모델(tabular) × 2데이터셋 models7식 최적화 일괄 실행.
# Transformer는 시퀀스 입력이라 이 집계테이블엔 미적용(README 명시).
set -u
PY="/c/Users/playdata2/anaconda3/python.exe"
S="/c/Users/playdata2/Documents/ml_workspace/team_project_churn/preprocessing_project/rec_models7/models_rec_bayes.py"
RES="/c/Users/playdata2/Documents/ml_workspace/team_project_churn/preprocessing_project/rec_models7/results.tsv"
echo -e "dataset\tmodel\tresult" > "$RES"

# 모델별 trial 예산(다중분류 비용 고려, 린)
run () { # $1=dataset $2=model $3=trials
  echo ">>> $1 / $2 (trials=$3) $(date +%H:%M:%S)"
  "$PY" "$S" "$1" "$2" "$3" 2>&1 | grep -E "RESULTREC|Error|Errno|ValueError" | tee -a "$RES"
}

for DS in cat item; do
  run $DS DecisionTree 6
  run $DS LogReg       6
  run $DS RandomForest 5
  run $DS XGBoost      3
  run $DS LightGBM     3
  run $DS CatBoost     3
done
echo "=== ALL DONE $(date +%H:%M:%S) ==="

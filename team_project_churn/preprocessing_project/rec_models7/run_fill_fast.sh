#!/usr/bin/env bash
# 빠른 정형모델(DT·RF·LogReg) × {cat,item} 완주. skip-aware + 모델별 로그분리.
set -u
PY="/c/Users/playdata2/anaconda3/python.exe"
BASE="/c/Users/playdata2/Documents/ml_workspace/team_project_churn/preprocessing_project"
S="$BASE/rec_models7/models_rec_bayes.py"
RES="$BASE/rec_models7/results.tsv"
LOGD="$BASE/rec_models7/logs"; mkdir -p "$LOGD"
[ -f "$RES" ] || printf 'dataset\tmodel\tresult\n' > "$RES"
declare -A DIR=( [cat]="v4-1_rec_category" [item]="v4-2_rec_item" )

run () { # ds model trials
  local ds=$1 m=$2 tr=$3
  local j="$BASE/${DIR[$ds]}/output/$m/prep_${m}_rec.joblib"
  if [ -f "$j" ]; then echo ">>> SKIP $ds/$m (완료) $(date +%H:%M:%S)"; return; fi
  echo ">>> RUN  $ds/$m trials=$tr $(date +%H:%M:%S)"
  "$PY" "$S" "$ds" "$m" "$tr" > "$LOGD/${ds}_${m}.log" 2>&1
  local rc=$?
  [ $rc -ne 0 ] && echo "    FAIL rc=$rc -> logs/${ds}_${m}.log"
  grep -hE "RESULTREC" "$LOGD/${ds}_${m}.log" 2>/dev/null | tee -a "$RES"
}

run item DecisionTree 8
run cat  RandomForest 5
run item RandomForest 5
run cat  LogReg       6
run item LogReg       6
echo "=== FAST DONE $(date +%H:%M:%S) ==="

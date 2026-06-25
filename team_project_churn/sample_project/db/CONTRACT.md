# 모델 → 백엔드 공유 변수 계약 (CONTRACT)

각 모델 담당자가 학습/예측을 **파일로** 끝낸 뒤, 백엔드에 **정확히 이 JSON 형태**로 제출한다. 백엔드는 추론하지 않으므로, 아래 변수들이 화면(Streamlit)·앙상블·리텐션의 유일한 입력이다.

## 1. `POST /models/submit` — 모델 등록 + (옵션) 배치예측
헤더: `x-api-key: <API_KEY>`, `Content-Type: application/json`

```jsonc
{
  "model_name": "CatBoost_ES_Feb",          // (필수) 고유. registry UNIQUE 키
  "model_type": "tree",                      // (필수) tree | linear | sequence | ensemble
  "feature_schema_version": "v2",            // 피처 스키마 버전(기본 v2 = 10피처 canonical ndays)
  "artifact_path": "models/tabular/catboost_es_feb.cbm",  // (필수) 모델 바이너리 파일 경로(DB 밖)
  "dataset_path": "preprocessing_project/v4_model_prep/output/DecisionTree/train.parquet",
  "train_period": "2019-10-01..2020-01-31",
  "preprocessing_config": {                  // 서빙 시 전처리 재현 근거(누수차단)
    "scale": "none",                         // none | log1p+minmax | zscore ...
    "log1p": false,
    "scaler_path": null,                     // linear면 joblib 경로
    "label": "churn",                        // churn | churn_no_purchase
    "feature_order": ["recency_days","tenure_days","ndays","n_events","n_view","n_cart","n_remove_from_cart","n_purchase","avg_price","purch_amt"],
    "extra_features": [],                    // v2 추가피처(있으면 컬럼명 배열)
    "threshold": 0.42,                       // 운영 임계값(F1/비용 기준)
    "calibrator": "isotonic"                 // none | isotonic | platt
  },
  "metrics": { "cv_pr_auc": 0.93, "oot_pr_auc": 0.936, "auc": 0.7902, "brier": 0.11, "ece": 0.03, "threshold": 0.42 },
  "set_active": true,                        // 같은 model_type에서 활성 1개로 전환
  "predictions": [                            // (옵션) 배치 예측 — user별
    { "user_id": "519...", "session_id": null, "churn_probability": 0.87, "risk_level": "high",
      "top_factors": { "recency_days": 9, "n_purchase": 0, "last_cat_id": "1487" } }
  ]
}
```
응답: `{ "registered": {"model_id":1,"mode":"mysql|memory"}, "predictions_logged": N }`

### model_type별 필수 차이
| type | preprocessing_config 추가키 | 비고 |
| --- | --- | --- |
| tree (XGB/LGBM/CatBoost/DT/RF) | `scale:"none"`, `cat_features:[...]`(네이티브 범주형이면) | 스케일 불필요 |
| linear (LogReg) | `scale:"log1p+minmax"`, `scaler_path` | 스케일러 joblib 필수 |
| sequence (Transformer 등) | `seq_granularity`,`seq_len`,`n_channels`,`mask:true` | 시퀀스 npz 경로 |
| ensemble | `members:[{model_name,weight}]` | 멤버 가중 |

## 2. 실시간 예측 표시용 — `POST /predict` (모델 산출 확률 전달)
```jsonc
{ "user_id":"519...", "churn_probability":0.73 }
→ { "user_id":"519...", "churn_probability":0.73, "risk_level":"high", "recommended_action":"쿠폰 발송..." }
```
> 핵심: **백엔드는 확률을 만들지 않는다.** 실시간도 모델파트가 `predictions`로 제출하거나, 모델 사이드카가 Neon pull 이벤트로 점수를 만들어 `/models/submit` 또는 `/predict`로 전달한다.

## 3. 앙상블 — `POST /ensemble`
```jsonc
{ "user_id":"519...", "members":[ {"model_name":"CatBoost","prob":0.81,"weight":2},
                                   {"model_name":"LightGBM","prob":0.77,"weight":2},
                                   {"model_name":"Transformer","prob":0.66,"weight":1} ] }
→ { "prob_ensemble":0.766, "risk_level":"high", "improvement":"...", "members":[...] }
```

## 4. 불변(파일에 두는 것 — DB로 보내지 않음)
- 모델 바이너리(`artifact_path`), 전처리기 joblib(`scaler_path`), 시퀀스 npz, 1.27M 유저 피처/임베딩.
- DB에는 **경로 + 메타 + 운영대상 유저 예측/요약만**.

## 5. 표준 risk 규칙(공유)
`high ≥ 0.65`, `medium ≥ 0.35`, 그 미만 `low` (백엔드 `RISK` 상수와 동일).

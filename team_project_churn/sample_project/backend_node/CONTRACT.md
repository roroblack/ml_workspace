# 모델파트 → 백엔드 제출 계약 (19-1 §7.3)

`POST /models/submit` (헤더 `x-api-key`). 모델파트가 학습/평가를 파일로 끝낸 뒤 결과 변수만 제출.

```json
{
  "model_name": "CatBoost_Churn_v2", "model_type": "tree",
  "feature_schema_version": "v2", "label_name": "churn", "horizon_days": 7,
  "preprocessing_config": { "scale": "none", "count_transform": "none", "feature_order": ["recency_days","tenure_days","ndays","n_events","n_view","n_cart","n_remove_from_cart","n_purchase","avg_price","purch_amt"] },
  "dataset_path": "data/processed/churn/models7/CatBoost_train.parquet",
  "artifact_path": "models/churn/catboost_churn_v2.cbm",
  "metrics": { "roc_auc": 0.791, "pr_auc": 0.9347, "best_threshold": 0.52, "best_f1": 0.918 },
  "evaluation": { "eval_predictions_path": "data/processed/evaluation/eval_predictions.parquet", "shap_summary_path": "data/processed/evaluation/shap_summary.json" },
  "is_active": true
}
```
- 응답: `{ model_id, eval_id, mode }`. `is_active=true`면 동일 type 기존 active 해제.
- 대용량(parquet/cbm/npz)은 **파일+경로**. DB엔 경로만.
- 예측 의미 = **향후 7일 이탈 확률**(churn = 기준 시점 이후 7일 무이벤트).
- 차트 원천 = 학습 종료 시 산출물(`metrics_summary.json`,`curves.json`,`shap_summary.json`) — 백엔드는 재학습 안 함.

# v5_item_recommendation — 상품목록 기반 다음 아이템 예측

목적: 17-3-5의 다음 카테고리 예측을 상품 레벨로 확장한다.

```text
X = 세션 내 직전 L개 이벤트의 상품/카테고리/행동/가격/gap 피처
y = 다음 이벤트의 product_id
```

기본 실행은 빠른 벤치마크용으로 `sample_project/data/raw/events.csv`를 사용한다. 전체 5개월 zip으로 확장할 수 있게 스크립트 인자를 열어 둔다.

## 실행

```bash
cd team_project_churn
python preprocessing_project/v5_item_recommendation/src/build_product_dataset.py --target-products 50 --max-windows 20000
python preprocessing_project/v5_item_recommendation/src/train_product_models.py --max-train 12000
```

v4의 7모델 구성(DecisionTree, RandomForest, LogReg, XGBoost, LightGBM, CatBoost, Transformer)으로 추가 검증:

```bash
python preprocessing_project/v5_item_recommendation/src/train_v4_product_models.py --max-train 12000
```

## 산출물

```text
output/
├── product_nextitem_dataset.parquet
├── product_vocab.json
├── dataset_meta.json
├── benchmark_metrics.json
├── benchmark_predictions.parquet
├── model_registry.json
├── v4_benchmark_metrics.json
├── v4_benchmark_predictions.parquet
├── v4_model_registry.json
└── models/
```

## 지표

- Top-1: 1순위 추천 상품이 실제 다음 상품과 일치하는 비율
- Hit@10: 추천 10개 안에 실제 다음 상품이 들어간 비율
- MRR@10: 실제 다음 상품이 몇 번째에 있는지를 반영한 reciprocal rank 평균

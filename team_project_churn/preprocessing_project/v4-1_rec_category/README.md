# v4-1_rec_category — 추천 전처리 (Y = 다음 카테고리)

v4_model_prep(=churn 7모델 전처리)의 **X를 그대로 재사용**하고, 라벨만 churn(이탈 이진) → **다음 기간 주요 관심 category_id**(다중분류)로 바꾼 **추천용** 데이터셋이다.

## 무엇이 바뀌었나 (v4 대비)
| 항목 | v4_model_prep | **v4-1 (여기)** |
| --- | --- | --- |
| X(피처) | 유저별 22피처(+파생) | **동일**(`processed_eventbase/*_tabular_v2.parquet` 재사용) |
| Y(라벨) | `churn`(이탈 0/1) | **`y_next_category`** = 결과창 최빈 `category_id` |
| 과제 | 이진분류(이탈) | **다중분류(다음 카테고리 추천)** |
| 시간분할 | 관찰→결과(7일) 외삽 | **동일**(누수 차단) |

## 시간 분할(누수 차단, v4와 동일)
- train: 관찰 `2019-10-01~2020-01-25`(X) → 결과 `2020-01-25~02-01`(Y) — `src/2020-Jan.csv.zip`
- test : 관찰 `2020-02-01~02-22`(X) → 결과 `2020-02-22~03-01`(Y) — `src/2020-Feb.csv.zip`

## 산출물 (`output/`)
| 파일 | 내용 |
| --- | --- |
| `train_cat.parquet` | **42,186행** × (`user_id` + 27피처 + `y_next_category`) · 389클래스 |
| `test_cat.parquet`  | **29,205행** · 361클래스(train 클래스 부분집합) |
| `classes_cat.json` | train 카테고리 클래스 목록 |
| `meta_cat_*.json` / `first30_cat.txt` | 메타(커버리지·클래스수)·30행 미리보기 |

- **커버리지**: train 3.3% / test 9.8% (= X 유저 중 결과창에 활동이 있어 추천 타깃이 존재하는 비율). 이탈률이 높아 낮은 게 정상 — 추천 타깃은 "다음 기간에 돌아온 유저"에만 존재.
- Y는 `category_id`(int64) 원값. 모델 단계에서 라벨 인코딩(0..C-1).

## 사용법 (모델, v4 models7와 동일 프레임 · 다중분류로)
```python
import pandas as pd
tr = pd.read_parquet("output/train_cat.parquet"); te = pd.read_parquet("output/test_cat.parquet")
X_cols = [c for c in tr.columns if c not in ("user_id", "y_next_category")]
Xtr, ytr = tr[X_cols], tr["y_next_category"]
# LightGBM/XGBoost/CatBoost: objective=multiclass(num_class=389), top-k 정확도/MRR로 추천 평가
```
- 평가: top-1/top-5 accuracy, MRR, Recall@k (추천 지표).
- 재현: `python src/pp_rec_category.py` (v4 X parquet이 먼저 생성돼 있어야 함).

## 비고
- X에는 `top_category_id`/`last_cat_id`가 포함(과거 관심) → 다음 카테고리의 강한 신호(자기상관). 추천에선 정상.
- `churn`/`churn_no_purchase`는 피처에서 제외(라벨 혼동 방지).
- 시퀀스 기반 다음카테고리는 별도 `scripts/nextcat_1735.py`(17-4-0). 본 v4-1은 **집계피처 기반(v4 models7 정합)**.

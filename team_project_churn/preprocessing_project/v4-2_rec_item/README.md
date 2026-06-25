# v4-2_rec_item — 추천 전처리 (Y = 다음 아이템)

v4_model_prep의 **X를 그대로 재사용**하고, 라벨을 **다음 기간 주요 관심 `product_id`** 로 바꾼 **아이템 추천용** 데이터셋. `product_id`는 카디널리티가 매우 커서(수만) **train 빈도 상위 `TOPK=1000` 아이템만 클래스**로 둔다.

## 무엇이 바뀌었나 (v4 대비)
| 항목 | v4_model_prep | **v4-2 (여기)** |
| --- | --- | --- |
| X(피처) | 유저별 22피처(+파생) | **동일**(`*_tabular_v2.parquet` 재사용) |
| Y(라벨) | `churn` | **`y_next_item`** = 결과창 최빈 `product_id`(상위 1000) |
| 과제 | 이진분류 | **다중분류(다음 아이템 추천, 1000-class)** |

## 시간 분할(v4와 동일, 누수 차단)
- train: 관찰 ~`2020-01-25` → 결과 `01-25~02-01` — `src/2020-Jan.csv.zip`
- test : 관찰 `02-01~02-22` → 결과 `02-22~03-01` — `src/2020-Feb.csv.zip`

## 산출물 (`output/`)
| 파일 | 내용 |
| --- | --- |
| `train_item.parquet` | **20,254행** × (`user_id` + 27피처 + `y_next_item`) · 1000클래스 |
| `test_item.parquet`  | **11,159행** · 966클래스(train 상위1000 부분집합) |
| `classes_item.json` | train 상위 1000 product_id 목록 |
| `meta_item_*.json` / `first30_item.txt` | 메타·미리보기 |

- **커버리지**: 결과창 활성 train 3.3%/test 9.8% 중, **top-1000 아이템에 해당하는 비율 train 48.0% / test 38.2%**. 나머지(롱테일 아이템)는 분류 tractable을 위해 제외 — `meta`에 명시.
- TOPK는 `src/pp_rec_item.py`의 `TOPK`로 조절(↑ 커버리지↑, 클래스↑·난이도↑).

## 사용법 (다중분류)
```python
tr = pd.read_parquet("output/train_item.parquet"); te = pd.read_parquet("output/test_item.parquet")
X_cols = [c for c in tr.columns if c not in ("user_id", "y_next_item")]
# objective=multiclass(num_class=1000) · Recall@k/MRR로 평가(아이템 추천)
```
- 재현: `python src/pp_rec_item.py`

## 비고
- 아이템 추천은 클래스가 많아(1000) 집계피처만으론 한계 — 본격 추천은 시퀀스(SASRec/`product_id` 레벨, 17-4-0 §4) 권장. v4-2는 **v4 models7 프레임 정합 + 베이스라인** 용도.
- `churn` 라벨은 피처에서 제외.

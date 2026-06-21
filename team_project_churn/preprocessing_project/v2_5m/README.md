# v2_5m — 5개월 풀데이터 전처리기 (정형 + 시퀀스 + 7모델 최적본)

REES46 5개월(2019-10~2020-02) 풀데이터에서 **유저 단위 이탈 라벨 테이블(정형)** 과 **주별 시퀀스(DL)** 를 만들고, 7개 모델별 최적 전처리본까지 생성하는 버전.

## 1. 재현 소스 (`src/`)
| 파일 | 역할 | 입력 → 출력 |
| --- | --- | --- |
| `prep_5month.py` | **정본 생성**(풀 1.27M). 월별 순차 집계로 메모리 안전 | `src/2019-*.csv.zip` → `processed_5m/{train,test}_tabular.parquet`,`_seq.npz`,`meta_5m.json` |
| `restore_churn_np.py` | 코호트 parquet에 보조라벨 `churn_no_purchase` 복원 | cohort parquet 갱신 |
| `bayes_5m.py` | ML/DL 전처리 베이지안 최적화(코호트 recency≤7) | → `bayes_5m.json`, 코호트 parquet |
| `finalize_5m_opt.py` | 최적 전처리 적용본 저장 + **스케일러 영속화** | → `*_ML_opt.parquet`, **`ML_opt_scaler.joblib`**, `*_seq_DL_opt.npz` |
| `models7_opt.py` | 7모델별 전처리 베이지안 → 모델별 데이터셋 | cohort → `processed_5m/models7/{Model}_{train,test}.parquet`, `Transformer_*_seq.npz`, `models7_opt.json` |
| `models7_earlystop.py` | 부스팅/Transformer early stopping 재학습·평가 | → `models7_earlystop.json` (최종 **CatBoost Feb AUC 0.7902**) |
| `rec_extract_5m.py` | 추천 전용 데이터셋(user×item) 추출 | → `processed_rec/` |
| `preprocess_bo.py`,`preprocess_bayes.py`,`preprocess_bayes2.py` | (참고) Bank/초기 전처리 베이지안 실험 | → `outputs/` |

재현: `python preprocessing_project/v2_5m/src/prep_5month.py` → `restore_churn_np.py` → `bayes_5m.py` → `finalize_5m_opt.py` → `models7_opt.py` → `models7_earlystop.py` (repo 루트에서 실행, `HERE`는 자동으로 `team_project_churn`).

## 2. 기간 분할 (시간 외삽, 누수 차단)
| 데이터셋 | 관찰기간(피처 X) | 결과기간(7일, 라벨 Y) |
| --- | --- | --- |
| train | 2019-10-01 ~ 2020-01-25 (≈17주) | 2020-01-25 ~ 01-31 |
| test | 2020-02-01 ~ 02-21 (≈3주) | 2020-02-22 ~ 02-28 |

## 3. X 구성 — 정형 10피처
`recency_days, tenure_days, ndays, n_events, n_view, n_cart, n_remove_from_cart, n_purchase, avg_price, purch_amt`
- 카운트 피처(`COUNT_IDX=[ndays,n_events,n_view,n_cart,n_remove_from_cart,n_purchase,purch_amt]`)는 `log1p` 옵션 대상.
- `recency_days` = 관찰종료 − 마지막활동(일). `tenure_days` = 마지막−최초 활동. `avg_price` = sum_price/n_events.

### X 구성 — 시퀀스
- `train_seq.npz` X:[N, 17, 3] / `test_seq.npz` X:[N, 3, 3], step_features=`[view, cart, purchase]` 주별 카운트.
- DL 최적본 `*_seq_DL_opt.npz`: 코호트·최근 4주 [N,4,3] (raw).

## 4. Y 구성 — 라벨 2종
| 라벨 | 정의 | train 비율 |
| --- | --- | --- |
| `churn` (주) | 결과기간 7일간 **어떤 이벤트도 없으면 1** | `meta_5m.json` 참조 |
| `churn_no_purchase` (보조) | 결과기간 7일간 **purchase 없으면 1** | 〃 |
- 코호트 7모델 학습은 `recency≤7` 서브셋·`churn` 사용.

## 5. 결과물 (`output/` + 정본 위치)
- `output/`: `meta_5m.json`, `ML_opt_scaler.joblib`, `bayes_5m.json`, `models7_opt.json`, `models7_earlystop.json`. 상세 대용량 목록은 `output/OUTPUT_INDEX.md`.
- 정본(대용량): `sample_project/data/processed_5m/`(정형 parquet·시퀀스 npz)와 `.../models7/`(모델별 최적본).
- 7모델 성능(Feb AUC, early stopping): **CatBoost 0.7902** / LightGBM 0.7895 / XGBoost 0.7888 / Transformer 0.7878 / LogReg 0.7856 / RandomForest 0.7573 / DecisionTree 0.7616.

## 6. 알려진 호환성 한계 (→ [17-6-5](../../reports/17-6-5_5m_models7_호환성정합_및_개선_계획서.md))
- models7 모델별 parquet은 **모델별 스케일러가 저장되지 않아 서빙 재현 불가**(개선 대상: `prep_{Model}.joblib` + manifest).
- models7는 `churn`만 보존(보조라벨 드롭). 배포본(v1)은 7피처·일별 → v2(10피처·주별)와 스키마 단절.

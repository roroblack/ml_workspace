# v1_daily — 배포본(일별 14일) 전처리기

`sample_project`(운영 배포본)에서 실시간 LSTM 서빙에 쓰는 **일별 시퀀스 + RFM 정형** 전처리. **정본은 `sample_project`** 이며, 본 폴더는 참조 스냅샷 + 명세.

## 1. 재현 소스
- 정본: `sample_project/src/build_features.py` (+ `config.py`, `generate_sample_data.py` / `ingest_rees46.py`)
- 참조 스냅샷: `src/build_features.py` (드리프트 방지용 복사본 — 수정은 정본에서)
- 재현: `python sample_project/src/run_pipeline.py [real|src/2019-Nov.csv.zip]`

## 2. X 구성 — 정형 7피처
`recency_days, n_view, n_cart, n_purchase, n_events, active_days, avg_price`
- `config.OBS_DAYS=14`(관찰), `OUTCOME_DAYS=7`(결과). 기준일 BASE는 데이터 최소일에서 자동 도출.

### X 구성 — 시퀀스
- `sequences.npz` X:[N, 14, 3], step_features=`[view, cart, purchase]` **일별** 카운트. 0 패딩.
- 서빙 스케일러: `models/seq_scaler.npz`(feature별 mean/std).

## 3. Y 구성
- `churn`: 결과기간 7일 무활동=1.

## 4. 결과물
- 정본: `sample_project/data/processed/{features.csv, sequences.npz}`, DB(`churn.db`), `sample_project/models/{lstm.pth, seq_scaler.npz, lstm_meta.json}`
- 본 폴더 `output/feature_schema.yaml` — 스키마 스냅샷.

## 5. v2_5m와의 차이(호환성)
| 축 | v1_daily | v2_5m |
| --- | --- | --- |
| 피처 | 7 (`active_days`) | 10 (`ndays`+tenure/remove/purch_amt) |
| 시퀀스 | 일별 14 | 주별 17/3 |
| 라벨 | churn | churn + churn_no_purchase |
| 모집단 | 샘플 | 전체 1.27M / 코호트 |

→ v2 모델을 v1 배포본에 직접 붙일 수 없음. 통합은 [17-6-5](../../reports/17-6-5_5m_models7_호환성정합_및_개선_계획서.md)의 스키마 v2 단일화 + 어댑터 경유.

# 18. sample_project 작업 현황 점검 리포트

작성일: 2026-06-19

## 1. 결론 요약

`sample_project`는 최종 본 프로젝트가 아니라, `05-5` 비전 명세서의 구조를 로컬에서 검증하기 위한 샌드박스 성격이 강하다. 현재 들어가 있는 범위는 다음과 같다.

- REES46 적재, 피처 생성, ML/DL 학습, SQLite DB 적재, Streamlit 대시보드, 실시간 예측 서비스 골격이 있다.
- 5개월 REES46 전처리 산출물과 세션 단위 실시간/추천 실험 산출물이 별도로 들어 있다.
- Vercel/Neon 연동은 실제 코드가 아니라 `vercel_site/README.md` 수준의 설계 자리만 있다.
- 얼굴 로그인은 실제 OpenCV/InsightFace가 아니라 user_id 기반 로그인 스텁이다.
- 현재 DB에는 이벤트/피처/시퀀스/사용자 테이블은 채워져 있지만, `model_registry`와 `prediction_log`가 비어 있어 대시보드의 모델 성능/최신 예측 화면은 바로 완성 상태로 보기 어렵다.

## 2. 현재 실행 상태

- Streamlit 프로세스가 실행 중이다.
- 실행 명령: `python -m streamlit run frontend_streamlit\app.py --server.port 8502`
- 접속 확인: `http://localhost:8502` HTTP 200 응답 확인.
- 로그 기준 URL:
  - Local: `http://localhost:8502`
  - Network: `http://192.168.0.94:8502`

## 3. 폴더별 구현 상태

| 위치 | 상태 | 설명 |
| --- | --- | --- |
| `src/run_pipeline.py` | 구현됨 | REES46 적재 -> 피처 생성 -> ML/DL 학습 -> 배치 예측 오케스트레이션 |
| `src/ingest_rees46.py` | 구현됨 | `2019-Nov.csv.zip` 기본 적재, 사용자 8,000명 표본으로 `data/raw/events.csv` 생성 |
| `src/build_features.py` | 구현됨 | 14일 관찰 + 7일 결과 라벨, 정형 피처/시퀀스 생성, SQLite 운영 테이블 적재 |
| `src/train_ml.py` | 구현됨 | LogReg/GBM 학습, 최고 모델 저장 |
| `src/train_dl.py` | 구현됨 | LSTM 시퀀스 모델 학습, 모델/스케일러 저장 |
| `frontend_streamlit/app.py` | 구현됨 | 고객/관리자 대시보드, 위험도/퍼널/모델성능 표시 |
| `frontend_streamlit/services/predictor.py` | 구현됨 | DB 최근 이벤트를 LSTM 입력으로 변환해 이탈확률 예측 |
| `frontend_streamlit/services/face_auth.py` | 스텁 | user_id 존재 여부로 로그인, 실제 얼굴 인증은 미연동 |
| `db/db_client.py` | 부분 구현 | SQLite는 동작, Postgres는 접속 함수만 있고 초기화는 명세서 DDL에 위임 |
| `vercel_site/` | 설계만 있음 | 실제 Next.js/API 파일 없음, README만 존재 |
| `notebooks/` | 비어 있음 | Colab 학습 노트북 구조는 아직 sample_project에 없음 |

## 4. 데이터와 산출물 현황

### 4.1 로컬 샘플 파이프라인 산출물

| 파일 | 현재 상태 |
| --- | --- |
| `data/raw/events.csv` | 97,669행, 약 10.56MB |
| `data/processed/features.csv` | 3,911명, 9컬럼, 약 0.21MB |
| `data/processed/sequences.npz` | `X=(3911, 14, 3)`, `user_id=(3911,)`, 약 0.66MB |
| `models/tabular_model.joblib` | 존재 |
| `models/tabular_scaler.joblib` | 존재 |
| `models/lstm.pth` | 존재 |
| `models/seq_scaler.npz` | 존재 |

현재 샘플 기본 윈도우는 `OBS_DAYS=14`, `OUTCOME_DAYS=7`, `SEQ_LEN=14`이다. 즉 본 프로젝트의 5개월 전처리 산출물과 같은 구조가 아니라, 로컬 데모용 짧은 윈도우이다.

### 4.2 5개월 전처리 산출물

`data/processed_5m/`와 `data/processed_5m.zip`에는 5개월 기반 산출물이 들어 있다.

| 산출물 | shape/내용 |
| --- | --- |
| `train_tabular.parquet` | `(1,266,514, 13)` |
| `test_tabular.parquet` | `(298,813, 13)` |
| `train_seq.npz` | `X=(1266514, 17, 3)` |
| `test_seq.npz` | `X=(298813, 4, 3)` |
| `train_tabular_ML_opt.parquet` | `(109,378, 12)` |
| `test_tabular_ML_opt.parquet` | `(109,736, 12)` |
| `train_seq_DL_opt.npz` | `X=(109378, 4, 3)` |
| `test_seq_DL_opt.npz` | `X=(109736, 4, 3)` |

`meta_5m.json` 기준:

- Train: 2019-10-01 ~ 2020-01-25 관찰, 2020-01-25 ~ 2020-02-01 7일 결과
- Test: 2020-02-01 ~ 2020-02-22 관찰, 2020-02-22 ~ 2020-03-01 7일 결과
- 피처: `recency_days`, `tenure_days`, `ndays`, `n_events`, `n_view`, `n_cart`, `n_remove_from_cart`, `n_purchase`, `avg_price`, `purch_amt`

주의: 이 5개월 산출물은 sample_project의 기본 `run_pipeline.py`가 직접 사용하는 기본 데이터는 아니다. 별도 실험 산출물로 보관된 상태에 가깝다.

### 4.3 세션/추천 실험 산출물

| 위치 | 내용 |
| --- | --- |
| `data/processed_full/session_level_full.npz` | REES46 2019-Nov 전체 기반, `X=(263039, 10, 8)` |
| `data/processed_full/recommend_user_interest.parquet` | `(368183, 5)`, user별 관심 카테고리/브랜드 추천 후보 |
| `data/processed_nextcat/nextcat_dataset.npz` | 다음 카테고리 예측, `143,915` window |
| `data/processed_nextcat/nextcat_dataset_fullwin.npz` | 다음 카테고리 전체 window, `883,959` window |

추천 쪽 실험은 이미 REES46 기반으로 의미 있는 형태가 있다. 다만 실제 Streamlit 화면/DB API에 연결된 기능으로는 아직 통합되지 않았다.

## 5. 현재 모델/실험 결과

### 5.1 로컬 샘플 모델

`models/tabular_meta.json`

- Best tabular: `LogReg`
- LogReg AUC: `0.8687`, F1: `0.8269`, Recall: `0.7288`
- GBM AUC: `0.8636`, F1: `0.9011`, Recall: `0.9424`

`models/lstm_meta.json`

- LSTM AUC: `0.8726`, F1: `0.8279`, Recall: `0.7254`
- 시퀀스: 14일 x 3피처

주의: 모델 파일은 존재하지만 현재 SQLite의 `model_registry`는 비어 있다. 즉 모델 파일과 DB 활성 모델 등록 상태가 불일치한다.

### 5.2 5개월 베이지안/전처리 결과

`outputs/realtime/bayes_5m.json`

- Cohort: `recency<=7`
- Label: `7일 무활동 이탈`
- Train: `109,378`, Test: `109,736`
- Baseline LogReg CV AUC: `0.7716`
- ML LogReg: CV AUC `0.7967`, Feb test AUC `0.7854`
- ML GBM: CV AUC `0.8006`, Feb test AUC `0.7896`
- DL LSTM: Val AUC `0.7677`, Feb test AUC `0.7679`

현재 5개월 실험에서는 GBM이 LSTM보다 우세하다. DL이 의미 없다는 뜻은 아니지만, 현재 피처/라벨/윈도우에서는 ML 계열이 더 안정적으로 보인다.

### 5.3 실시간/세션 단위 실험

`outputs/realtime/session_sim.json`

- 세션 6,000개, 이벤트 30,668개
- 이탈/이탈성 label positive rate: `0.381`
- 모델 AUC는 LogReg/LSTM/Transformer가 약 `0.711` 근처로 거의 비슷하다.
- 이벤트 구동 방식이 recall `0.632`, precision `0.762`, median lead `52.3초`로 tick 기반보다 실시간 감지에 유리하게 나온다.

`outputs/realtime/forecast_session_full.json`

- REES46 2019-Nov 전체 기반 세션 부하 예측
- GRU RMSE `0.2959`, overload AUC `0.8661`
- Transformer overload AUC `0.8664`
- 단순 기준선 대비는 확실히 개선됨

`outputs/realtime/nextcat_models.json`

- 다음 카테고리 예측에서 Best: `SASRec`
- SASRec top1 `0.5682`, hit@10 `0.7961`, mrr@10 `0.6436`
- 단순 Last-category도 top1 `0.5772`로 매우 강함. 추천 기능을 넣을 때는 복잡한 DL 추천보다 Last-category/최근 관심 기반 baseline도 반드시 같이 비교해야 한다.

## 6. DB 상태

현재 `data/churn.db` 테이블 행 수:

| 테이블 | 행 수 |
| --- | ---: |
| `realtime_event_log` | 41,813 |
| `feature_user_snapshot` | 3,911 |
| `sequence_snapshot` | 3,911 |
| `face_user` | 3,912 |
| `face_login_log` | 1 |
| `model_registry` | 0 |
| `prediction_log` | 0 |
| `retention_action_log` | 0 |

해석:

- 이벤트/피처/시퀀스/로그인 사용자 적재는 되어 있다.
- 학습 모델 파일은 있으나 DB에는 활성 모델이 없다.
- 예측 로그가 없어서 Streamlit 고객 화면의 “최근 예측”과 관리자 화면의 “위험 고객 Top 20”은 현재 DB 상태만 보면 비어 있을 가능성이 높다.
- `predictor.predict_user()`를 실행하면 모델 파일로 예측은 가능하지만, `model_registry`가 비어 있으므로 `prediction_log.model_id`는 `NULL`로 남을 수 있다.

## 7. 발견된 불일치와 리스크

1. README와 코드의 기본 데이터 설명이 다르다.
   - README는 “기본 DB는 SQLite, 데이터는 합성 생성”이라고 설명한다.
   - 실제 `run_pipeline.py`는 인자가 없으면 REES46를 기본으로 사용하고, 실패하면 중단한다.

2. sample_project 기본 파이프라인과 5개월 산출물이 분리되어 있다.
   - 기본 앱은 `data/processed/features.csv`, `sequences.npz`, `models/lstm.pth`를 사용한다.
   - 5개월 최적화 산출물은 `data/processed_5m/`에 있지만 앱/DB 흐름에 직접 연결되어 있지 않다.

3. DB의 모델 등록/예측 로그가 현재 비어 있다.
   - 데모 시연 전에 `train_ml.py`, `train_dl.py`, `predictor.predict_all(log=True)`를 다시 실행하거나 DB 등록 상태를 복구해야 한다.

4. 얼굴 로그인은 아직 실제 기능이 아니다.
   - `opencv_face_login` 프로젝트의 InsightFace/OpenCV 구현을 가져와 `face_auth.verify(image)`에 연결해야 한다.

5. Vercel/Neon은 설계만 있고 구현은 없다.
   - `/api/events`, `/api/predictions/latest`, `/api/dashboard/summary`, `/api/auth/login-log` 라우트 파일이 없다.

6. Postgres 초기화 경로가 약하다.
   - `db_client.py`는 Postgres 접속은 고려하지만 `init_db()`는 SQLite만 지원하고 Postgres는 `NotImplementedError`를 낸다.

7. 추천 산출물은 있으나 서비스에 연결되지 않았다.
   - `recommend_user_interest.parquet`와 next-category 실험 결과가 Streamlit/DB/API에 아직 붙어 있지 않다.

8. Colab 노트북 폴더가 비어 있다.
   - 본 프로젝트에서 모델 학습을 Colab 중심으로 진행하려면 노트북 템플릿과 산출물 저장 규칙을 추가해야 한다.

## 8. 바로 다음 작업 추천

1. `README.md`를 현재 룰에 맞게 수정
   - “합성 데이터 기본” 설명을 “REES46 기본, synthetic은 데모 옵션”으로 변경.

2. DB 상태 복구
   - `sample_project`에서 `python src/train_ml.py`, `python src/train_dl.py` 실행 후 `predictor.predict_all(log=True)`로 `model_registry`, `prediction_log` 채우기.

3. 5개월 산출물을 앱 기본 모델로 연결
   - 현재 14일 샘플 모델 대신 `processed_5m/*_ML_opt.parquet`, `*_DL_opt.npz` 기반 모델을 Streamlit 예측 경로에 연결.

4. 얼굴 로그인 실제 연동
   - `opencv_face_login`의 얼굴 임베딩 추출/검증 로직을 `frontend_streamlit/services/face_auth.py`에 주입.

5. 추천 기능 1차 연결
   - `recommend_user_interest.parquet`를 기준으로 고객별 top category/top brand 추천을 대시보드에 표시.
   - 이후 next-category 모델 결과를 붙여 DL 추천과 baseline을 비교.

6. Vercel API 최소 구현
   - `/api/events`부터 구현해 Neon의 `realtime_event_log`에 적재.
   - Streamlit은 Neon을 읽어 예측/대시보드를 갱신.

7. Colab 노트북 구조 생성
   - `notebooks/colab/01_preprocess_rees46_5m.ipynb`
   - `notebooks/colab/02_train_ml_bayes.ipynb`
   - `notebooks/colab/03_train_dl_sequence.ipynb`
   - `notebooks/colab/04_export_artifacts.ipynb`

## 9. 최종 판단

`sample_project`는 “시연 가능한 앱 골격 + 실험 산출물 보관소”로는 꽤 많이 진행되어 있다. 하지만 지금 상태 그대로를 최종 프로젝트 구현체로 보기에는 아직 연결이 끊긴 부분이 있다. 특히 `5개월 REES46 최적화 산출물 -> 모델 등록 -> DB 예측 로그 -> Streamlit 화면 -> 추천/개선안` 흐름을 하나로 묶는 작업이 필요하다.

가장 먼저 할 일은 README 정합화와 DB 모델/예측 로그 복구이고, 그 다음이 5개월 산출물을 기본 예측 경로로 승격하는 작업이다.

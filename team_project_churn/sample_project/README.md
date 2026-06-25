# 고객 이탈 예측 — 샘플 프로젝트 (sample_project)

[05-5 비전 명세서](../reports/05-5_프로그램_구조도_및_함수_변수_비전_명세서.md) 아키텍처를 **로컬에서 즉시 실행 가능한 최소 샘플**로 구현한 것입니다. 계획서: [reports/15_샘플프로젝트_계획서.md](../reports/15_샘플프로젝트_계획서.md).

## 빠른 실행
```bash
# 1) 전체 파이프라인 (생성 → 피처 → ML/DL 학습 → DB → 배치 예측)
python src/run_pipeline.py

# 2) 대시보드 (얼굴 로그인 스텁 → 고객/관리자)
streamlit run frontend_streamlit/app.py
#   로그인 user_id:  관리자=admin / 고객=U0001 같은 ID

# (선택) 단건 예측 / 모델 비교 / 라이브 이벤트 시뮬
python src/predict.py U0001
python src/compare.py
python src/simulate_events.py
```
> 외부 의존 없음: 기본 DB는 SQLite(`data/churn.db`), 데이터는 합성 생성. Neon/Vercel/카메라 불필요.

## 동작 검증 결과(예시)
- 이벤트 ~32k / 사용자 1,850 (이탈률 ~80%)
- ML: LogReg AUC **0.869**, GBM 0.863 / DL: LSTM AUC **0.873**
- 배치 예측 1,850명 → `prediction_log` 적재, 활성 모델 등록

## 무엇이 실제 / 무엇이 스텁
| 구성 | 상태 |
| --- | --- |
| 데이터 생성·피처·시퀀스 | ✅ 실제(`src/generate_sample_data.py`, `build_features.py`) |
| ML(LogReg/GBM)·DL(LSTM) | ✅ 실제(`train_ml.py`, `train_dl.py`) |
| DB(운영 데이터) | ✅ SQLite(`db/db_client.py`), `DATABASE_URL` 주면 Postgres |
| 실시간 예측(로컬 torch) | ✅ 실제(`frontend_streamlit/services/predictor.py`) |
| 대시보드(고객/관리자) | ✅ 실제(`frontend_streamlit/app.py`) |
| 이커머스 시뮬(Vercel 대역) | 🔁 로컬 `src/simulate_events.py` |
| 얼굴 로그인 | 🔁 스텁(`services/face_auth.py`) — InsightFace 주입 지점 |
| 실제 REES46 | 🔁 `src/ingest_rees46.py`로 교체 가능 |

## 폴더 구조
- `src/` 데이터·학습 파이프라인 / `frontend_streamlit/` 앱·서비스(L2+)
- `db/` 스키마·클라이언트 / `configs/` 파라미터 / `models/` 모델파일 / `data/` 원본·피처
- `vercel_site/` Thin API(실배포 시) — 샘플은 로컬 시뮬레이터로 대체

## 실데이터/실배포로 확장
1. **REES46**: `python src/ingest_rees46.py <csv>` → `events.csv` 생성 후 `build_features.py`.
2. **Neon Postgres**: `.env`에 `DATABASE_URL` 설정(+psycopg, [05-5] DDL).
3. **얼굴 로그인**: `face_auth.verify(image)`에 InsightFace 구현 주입.
4. **Vercel 시뮬 사이트**: 로컬 시뮬 대신 `/api/events`로 이벤트 적재.

## 한계
- 합성 데이터·짧은 윈도우라 절대 수치는 데모용.
- 실시간 시퀀스는 "최근 N일 상대 윈도우"라 학습(절대 윈도우)과 약간의 train/serve skew 있음 — 실데이터/동일 윈도우로 정합화 가능.

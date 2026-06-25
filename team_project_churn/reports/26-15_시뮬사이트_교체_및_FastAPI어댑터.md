# 26-15. 시뮬레이션 사이트 교체 + FastAPI 어댑터 + 19-2 점검

기준: 2026-06-22. 지시: ①레거시 테이블 존치 ②서버 19-2 적합성 점검 ③`ecom-churn-simulation.zip`을 시뮬 사이트로 채택(현 placeholder 백업·삭제·교체, neon/ 보존)하고 우리 규격에 맞게 패치 ④폴더구조·동작 이중체크.

---

## 1. 레거시 테이블 — 존치(DROP 취소)
`churn_prediction·login_log·recommendation_log`는 실시간 추천·비상 백업용이라 **유지**. (앞서 DROP 검토했으나 사용자 지시로 취소.)

## 2. 19-2 규격 적합성 — 대체로 적합
`backend/app/` 실제 구조가 19-2 §5와 일치:
- `interfaces/http/`(health·auth·models·predictions·dashboard·sim·sim_external) · `application/`(usecase 7) · `domain/`(risk_level·face_match) · `infrastructure/{mysql,files,model_inference}` · `schemas/` · `validators/` · `config.py`·`main.py` ✅
- 미구축(향후): `infrastructure/external_neon/`(Neon adapter), `model_inference/{inference_port,model_worker_client,batch_prediction_importer}`. 현재 시뮬 로그는 MySQL `sim_event_log` 사용.

## 3. 시뮬 사이트 교체
- 정체: `ecom-churn-simulation.zip` = **React19+Vite + Express/tRPC + Drizzle** 풀스택 이커머스 시뮬(Manus export). 이미 **FastAPI 연동 설계**(`VITE_FASTAPI_URL`, `client/src/lib/fastApiClient.ts`).
- 백업: 현 placeholder를 백업 시도 → **`zip` CLI 미설치로 백업 실패**. git 추적분(`README.md`)은 `backups/simsite_OLD_from_git_*`로 복구. **미추적이던 옛 `index.html`(7.6KB placeholder)은 삭제로 유실**(교체 대상이었음). neon/·원본 zip은 보존. ⚠ 절차오류(백업 성공 확인 전 삭제) 기록.
- 교체: zip을 `simulation_site/`에 제자리 추출(client/server/shared/drizzle/…). **`neon/` 9파일 보존**, `ecom-churn-simulation.zip` 원본 보존.

## 4. 우리 규격에 맞게 패치 (핵심)
앱이 기대하는 외부 계약(ENV_SETUP.md/fastApiClient.ts) = raw JSON, x-api-key 불요. 우리 백엔드에 **어댑터** 신설:
- `interfaces/http/sim_external_router.py`:
  - `POST /api/churn/predict` → 이벤트 배치로 세션 채워 **prep 번들 실추론** → `{churn_probability(0~100%), risk_level, timestamp}`
  - `POST /api/recommendations` → `catalog_store` 유사카테고리 상품 추천
  - `POST /api/events` → `sim_event_log` 적재
  - `GET /api/analytics/session/{id}` → 세션 이벤트 카운트
  - `GET /health`(기존 재사용, 200)
  - **봉투 미적용·api-key 미요구**(대시보드 봉투 계약과 분리), CORS 전체 허용(main).
- 재사용: `sim_usecase.score_from_events/session_analytics`(신규), `catalog_store`, `sim_event_repository`.
- 앱 설정: `simulation_site/.env.local` → `VITE_FASTAPI_URL=http://localhost:8090`.

## 5. 검증(이중체크)
| 항목 | 결과 |
| --- | --- |
| 백엔드 import/부팅(34 엔드포인트, /api 어댑터+봉투 공존) | ✅ |
| `/api/churn/predict` | ✅ 200 raw, 실추론(예: 28.5%) |
| `/api/recommendations` | ✅ 200, recs 3건 |
| `/api/events` · `/api/analytics/session/{id}` | ✅ 200 |
| `/health`(앱 헬스체크) | ✅ 200 |
| 폴더구조·`neon/` 보존·`.env.local` | ✅ |
| node v26 / npm 11 | ✅ (pnpm 미설치) |

## 6. 프론트 실제 구동 — 빌드·기동 검증 완료 ✅
- `npm i -g pnpm`(10.x) → `pnpm install`(73초, node_modules OK) → 네이티브(esbuild·oxide) rebuild.
- `pnpm check`(tsc) **0 에러** · `pnpm build`(vite 1773모듈 + esbuild 서버번들) **성공**.
- **Windows 구동 패치**: `dev`/`start` 스크립트의 `NODE_ENV=...` 유닉스 프리픽스가 Windows에서 실패 → **`cross-env` 추가**(devDep)로 수정.
- `pnpm dev` → **`Server running on http://localhost:3000/`**, React 앱(`id="root"`) 정상 서빙 확인. (DB는 lazy=`DATABASE_URL` 있을 때만 → 없어도 부팅; OAuth/analytics 경고는 선택기능.)

## 7. 남은 단계 (시뮬팀 배포)
- 앱 자체 DB 기능(주문/상품 영속 등)은 `DATABASE_URL`(Neon)+`neon/` DDL·seed 적재 시 활성. **churn 데모 경로(이벤트→/api/churn/predict→추천)는 DB 없이 동작**(우리 백엔드만 있으면 됨).
- 실행: 터미널1 `backend`에서 `uvicorn app.main:app --port 8090`, 터미널2 `simulation_site`에서 `pnpm dev`(.env.local의 VITE_FASTAPI_URL=:8090).

## 7. 비고
미커밋(가지마 push 403 권한). `backups/`·`plans/`·`.env.local`은 .gitignore 로컬 전용.

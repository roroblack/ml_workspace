# 가지마 backend (Node, 19-1 클린아키)

운영 API 서버. **추론 안 함** — 모델파트 제출(`/models/submit`) 저장, 평가 산출물 ingest, **대시보드 chart API**, 예측 로그, 추천/리텐션. 운영 DB=MySQL, 시뮬 로그=Neon(외부).

## 레이어 (19-1)
```
src/
├── server.js  config.js
├── interfaces/http/routes.js         라우트→usecase (얇음)
├── application/usecases.js           usecase 오케스트레이션
├── domain/rules.js                   위험등급·리텐션·앙상블(순수)
├── infrastructure/mysql/pool.js      운영 DB repository(미설정 시 메모리)
├── infrastructure/files/artifactStore.js  eval 산출물 reader(차트 원천)
└── validators/schemas.js             제출 검증
db/schema_mysql.sql  seed.sql  migrations/
```
> 19-1의 파일 단위(usecase/repository 다중 파일)를 **데드라인 위해 레이어별 모듈로 압축**. 의존 방향(route→usecase→domain/port→infra)은 동일. 추후 파일 분리 가능.

## 실행
```
node scripts/check.js     # 무설치 검증
node src/server.js        # http://localhost:8090 (MySQL 없으면 메모리 모드, 차트는 파일 원천)
```
운영: `.env`(MySQL_*, API_KEY) 설정 + `npm i mysql2 dotenv` 후 schema_mysql.sql 적용.

## API (x-api-key, /health 제외)
`/health` · `POST /models/submit` · `GET /models` · `GET /dashboard/summary` · `GET /models/:model/charts/:name`(roc/pr/threshold/calibration/shap) · `POST /predict` · `POST /ensemble/run`.

## 원칙
추론·torch 번들 없음. 대용량은 파일+경로. Streamlit은 이 REST만 호출(직접 DB 금지). 차트 원천=학습 종료 시 eval 산출물.

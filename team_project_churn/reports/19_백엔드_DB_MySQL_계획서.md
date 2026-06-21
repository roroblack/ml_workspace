# 19. 백엔드 · DB(MySQL) 계획서

운영 백엔드 = **Node.js 서버**, 운영 DB = **MySQL**. 시뮬 로그 = **Neon(Postgres)**. 대용량(1.27M 유저 피처·시퀀스·모델바이너리)은 **DB 미적재 = 파일 + 경로**. 본 계획서는 [05-6](05-6_프로그램_구조도_및_함수_변수_비전_명세서.md)의 구현 사양이며, 산출물은 `backend/` 스켈레톤이다.

기준: [05-6](05-6_프로그램_구조도_및_함수_변수_비전_명세서.md) · [17-6-1](17-6-1_백엔드_DB_변경사항.md) · [17-6-5](17-6-5_5m_models7_호환성정합_및_개선_계획서.md) · `preprocessing_project/v2_5m/README.md`

---

## 1. 기술 스택 · Node 프레임워크 권고

### 1.1 비교 (NestJS vs Fastify vs Express)

| 축 | Express | **Fastify** | NestJS |
| --- | --- | --- | --- |
| 성능(req/s) | 기준 | **가장 빠름**(약 2배, 저오버헤드) | Express 어댑터 기준 보통(Fastify 어댑터 가능) |
| 스키마 검증 | 수동(미들웨어) | **JSON Schema 내장**(요청/응답 검증·직렬화) | class-validator/DTO(데코레이터) |
| 구조 강제 | 없음(자유) | 플러그인/관례(가벼움) | **강함**(모듈·DI·데코레이터, 대규모 적합) |
| 학습/세팅 비용 | 낮음 | 낮음~중 | **높음**(TS·DI·데코레이터 필수) |
| 보일러플레이트 | 적음 | 적음 | 많음 |
| 발표/데모 적합 | 단순하나 검증 빈약 | **얇고 빠르고 검증 내장** | 풍부하나 무겁다 |

### 1.2 권고 결론 — **Fastify** ✅

- 본 백엔드는 **Thin 운영 서버**다(05-x 원칙: 전송·DB I/O·인증·조합, **추론 없음**). NestJS의 DDD/DI 형식계층은 **과설계**이고(05-5 §0 "풀 DDD 안 만든다"와 일치), Express는 **요청/응답 스키마 검증을 직접 짜야** 해 `/models/submit`처럼 페이로드가 큰 계약에서 실수 위험이 크다.
- Fastify는 **JSON Schema 기반 검증·직렬화가 내장**되어 IO 계약(05-6 §2 제출 계약)을 선언적으로 강제할 수 있고, **성능·콜드스타트가 가볍고**, 플러그인으로 라우트/DB 풀을 깔끔히 분리한다 → 데모 규모(초당 수 건)에 정확히 맞다.
- 따라서 운영 **API 서버 = Fastify(+ mysql2 / pg)** 권고. 단 **구조는 최소**(기존 폴더 방식 유지 + 필요한 파일만): `backend/`에 `server.js`/`db.js`/`check.js`/`schema_mysql.sql`/`seed.sql` **평면 구성**(컨트롤러/레포지토리 등 다층 트리 제거). 스켈레톤은 **무설치 실행을 위해 Node 내장 http**로 동작하며, 운영 전환 시 Fastify로 교체.

### 1.3 레이어 확정 — "서버 선택"과 "웹"을 분리
> 사용자 요청은 **서버(백엔드) 선택**이다. 화면/웹은 **Streamlit**으로 통일한다(수업 기준, 결과도 Streamlit으로 제출).

| 레이어 | 선택 | 비고 |
| --- | --- | --- |
| **웹/화면(UI)** | **Streamlit (둘 다)** | ① 고객·관리자 **대시보드** ② **시뮬 결과 표시** 모두 Streamlit. 백엔드 REST를 호출해 표시 |
| **API 서버(백엔드)** | **Node.js — 권고 Fastify** / 스켈레톤은 내장 http | 추론 없음. JSON Schema 검증(`/models/submit` 계약) |
| 운영 DB | **MySQL 8** (`mysql2/promise`) | 운영 데이터만 |
| 시뮬 로그 DB | **Neon Postgres** (`pg`, SSL) | 시뮬 사이트 활동 로그 |
| 런타임/검증/환경 | Node 18+, CommonJS, `dotenv` | 추론(torch/ML) 의존성 0 |

- **요점**: Fastify는 "웹"이 아니라 **API 서버**다. 사용자에게 화면으로 보이는 **웹은 전부 Streamlit**이고, 그 뒤의 데이터 제공·조합 서버가 Node(Fastify 권고)다.

---

## 2. 무엇을 DB에 / 무엇을 파일에 (용량 경계, 불변)

| 데이터 | 위치 |
| --- | --- |
| 1.27M 유저 피처(parquet)·주별 시퀀스(npz) | **파일** + `*_snapshot.artifact_path` |
| 모델 바이너리(.cbm/.pth/.joblib)·prep_{Model}.joblib | **파일(models/)** + `model_registry.artifact_path` |
| 시뮬 유저활동 로그(대량 append) | **Neon**(시뮬 전용) |
| 운영 데이터(레지스트리·예측·피처스냅샷·추천·리텐션·앙상블·face) | **MySQL** ✅ |

MySQL엔 **운영 데이터만**. 대용량은 경로만.

---

## 3. MySQL ERD / DDL

ERD는 [05-6 §4](05-6_프로그램_구조도_및_함수_변수_비전_명세서.md) 참조. DDL 전문은 `backend/schema_mysql.sql`. 핵심 발췌:

```sql
-- 모델 레지스트리 (17-6-1 확장 컬럼 포함)
CREATE TABLE model_registry (
  model_id              BIGINT AUTO_INCREMENT PRIMARY KEY,
  model_name            VARCHAR(128) NOT NULL,
  model_type            VARCHAR(32)  NOT NULL,           -- tree|linear|sequence|ensemble
  feature_schema_version VARCHAR(16) DEFAULT 'v2',
  preprocessing_config  JSON,                            -- 서빙 전처리 분기 근거
  dataset_path          TEXT,                            -- 파일 경로
  artifact_path         TEXT NOT NULL,                   -- 모델 파일 경로
  train_period          VARCHAR(64),
  metric_cv             DOUBLE, metric_oot DOUBLE,
  metrics_json          JSON,
  is_active             TINYINT(1) DEFAULT 0,
  created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uq_model_name (model_name)
);

CREATE TABLE prediction_log (
  prediction_id   BIGINT AUTO_INCREMENT PRIMARY KEY,
  model_id        BIGINT, user_id VARCHAR(64) NOT NULL, session_id VARCHAR(128),
  churn_probability DOUBLE NOT NULL, risk_level VARCHAR(16) NOT NULL,
  top_factors_json JSON, recommended_action TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_pred_model FOREIGN KEY (model_id) REFERENCES model_registry(model_id)
);

-- 라벨 2종 + 코호트 (17-6-1)
CREATE TABLE feature_user_snapshot (
  snapshot_id BIGINT AUTO_INCREMENT PRIMARY KEY, user_id VARCHAR(64) NOT NULL,
  snapshot_time TIMESTAMP, cohort_flag TINYINT(1),
  churn TINYINT(1), churn_no_purchase TINYINT(1),
  obs_period VARCHAR(64), outcome_period VARCHAR(64),
  feature_json JSON, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE sequence_snapshot (
  snapshot_id BIGINT AUTO_INCREMENT PRIMARY KEY, user_id VARCHAR(64) NOT NULL,
  dataset_tag VARCHAR(32), seq_len INT, n_features INT,
  storage_format VARCHAR(16) DEFAULT 'npz', artifact_path TEXT, row_index INT,
  label INT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 추천(17-6-1 신규)
CREATE TABLE user_interest (
  user_id VARCHAR(64) PRIMARY KEY, top_category_id VARCHAR(64), top_brand VARCHAR(128),
  interest_json JSON, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
CREATE TABLE recommendation (
  rec_id BIGINT AUTO_INCREMENT PRIMARY KEY, user_id VARCHAR(64) NOT NULL, model_id BIGINT,
  rec_items_json JSON, rec_categories_json JSON, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_rec_model FOREIGN KEY (model_id) REFERENCES model_registry(model_id)
);

-- 앙상블
CREATE TABLE ensemble_result (
  ensemble_id BIGINT AUTO_INCREMENT PRIMARY KEY, user_id VARCHAR(64) NOT NULL,
  prob_ensemble DOUBLE NOT NULL, risk_level VARCHAR(16),
  improvement_json JSON, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE ensemble_member (
  member_id BIGINT AUTO_INCREMENT PRIMARY KEY, ensemble_id BIGINT NOT NULL,
  model_id BIGINT, weight DOUBLE, prob DOUBLE,
  CONSTRAINT fk_em_ens FOREIGN KEY (ensemble_id) REFERENCES ensemble_result(ensemble_id)
);
-- + retention_action_log, face_user, face_login_log (schema_mysql.sql 참조)
```

---

## 4. API 엔드포인트 명세

내부 인증: `x-api-key` 헤더(`API_KEY`). 모든 응답 `{ ok, data | error }`.

| Endpoint | Method | 입력 | 출력 | 책임 |
| --- | --- | --- | --- | --- |
| `/health` | GET | — | `{status}` | 헬스체크(DB 없이 200) |
| `/models/submit` | POST | 05-6 §2.1 제출 payload | `{model_id}` | 모델파트 결과 등록 + (옵션)배치 prediction 적재 |
| `/models` | GET | `?type&active` | 모델 목록 | 레지스트리 조회 |
| `/predict` | POST | `{user_id, model_id?}` | 예측 1건 | 단건 점수(스냅샷/제출본 기반) |
| `/predict/realtime` | POST | `{limit?, models?}` | 예측 배치 | Neon pull→스냅샷→분기점수→prediction_log |
| `/recommend` | POST | `{user_id}` | 추천 items/categories | user_interest×모델, view1/cart3/purchase5 가중 |
| `/retention-action` | POST | `{prediction_id}` | action | risk별 쿠폰/리마인드 → retention_action_log |
| `/ensemble` | POST | `{user_id, model_ids?, weights?}` | 앙상블 prob+개선점 | 가중평균/스태킹 → ensemble_result(+member) |

요청/응답 스키마는 Fastify JSON Schema로 라우트에 선언(예: `/models/submit`은 `model_name`,`model_type`,`artifact_path`,`preprocessing_config` 필수).

---

## 5. Neon 연동 (로그 pull / 결과 push)

| 방향 | 트리거 | 동작 |
| --- | --- | --- |
| **pull** | `neonPull.service` worker(`PULL_INTERVAL_MS`=5000) | `SELECT … FROM sim_event_log WHERE received_at > $cursor ORDER BY received_at` → cursor 갱신 → 사용자별 그룹핑 → `/predict/realtime` 입력 |
| **push(여유 시)** | 고위험/추천 생성 후 | `INSERT INTO coupon_push / rec_push (user_id, payload)` → 시뮬 사이트가 노출(이탈예측→이탈방지 루프) |

- Neon은 **SSL 필수 + 풀링 URL**. cursor는 메모리(데모) 또는 `pull_cursor` 테이블(영속).
- Neon 스키마(시뮬 측): `sim_event_log(event_id, user_id, session_id, event_type, product_id, category_code, price, event_time, received_at)` — REES46 스키마.

---

## 6. 폴더 구조

[05-6 §6](05-6_프로그램_구조도_및_함수_변수_비전_명세서.md) 참조. 요약:
**평면 최소**: `backend/{server.js, db.js, check.js, schema_mysql.sql, seed.sql, package.json, .env.example, CONTRACT.md, README.md}` — 단일 폴더, 다층 트리 없음.

---

## 7. 마일스톤

| 단계 | 작업 | 산출 |
| --- | --- | --- |
| M1 스캐폴드 | Fastify 앱·라우트 스텁·DDL·env (DB 없이 기동) | `backend/` ✅(본 작업) |
| M2 DB 연결 | mysql2/pg pool + repositories 실연결 + migrate | 운영 CRUD |
| M3 제출/조회 | `/models/submit`,`/models`,`/predict` 실동작 | 레지스트리·예측 |
| M4 실시간 | Neon pull worker + `/predict/realtime` | prediction_log 적재 |
| M5 추천/앙상블 | `/recommend`,`/retention-action`,`/ensemble` | 방지 루프 |
| M6 Streamlit 연동 | 대시보드를 백엔드 REST로 전환 | 표시 |
| M7 push(옵션) | 결과 Neon push → 사이트 쿠폰 | 시연 루프 닫기 |

---

## 8. 점검 체크리스트 (계획 대비)

| # | 항목 | 계획 | 스켈레톤 반영 | 상태 |
| --- | --- | --- | --- | --- |
| 1 | Node 프레임워크 권고+이유 | Fastify(§1.2) | 본 계획서 §1 | ✅ |
| 2 | MySQL ERD/DDL | §3 + schema_mysql.sql | `backend/schema_mysql.sql` | ✅ |
| 3 | `/models/submit` | §4 + CONTRACT | `server.js` routes + `db.registerModel` | ✅ |
| 4 | `/predict` | §4 | `server.js` | ✅ |
| 5 | `/predict/realtime` | §4 | `server.js` + `db.pullSimEvents` | ✅ |
| 6 | `/recommend` | §4 view1/cart3/purchase5 | `server.js` recommend() | ✅ |
| 7 | `/retention-action` | §4 | `server.js` retentionAction() | ✅ |
| 8 | `/ensemble` | §4 가중평균 | `server.js` ensemble() | ✅ |
| 9 | Neon pull/push | §5 | `db.pullSimEvents`/`pushRetention` | ✅ |
| 10 | registry 확장(전처리 config 등) | 17-6-1 | DDL + `db.registerModel` | ✅ |
| 11 | user_interest/recommendation 신설 | 17-6-1 | DDL | ✅ |
| 12 | 라벨2종/코호트 컬럼 | 17-6-1 | feature_user_snapshot DDL | ✅ |
| 13 | 대용량=파일+경로 | 불변 | artifact_path만 저장(바이너리 X) | ✅ |
| 14 | 추론코드 미포함 | 05-x 원칙 | 백엔드에 torch 의존성 0 | ✅ |
| 15 | env로 비밀 분리 | — | `.env.example` | ✅ |
| 16 | 문법/스모크 체크(무설치) | — | `check.js`(node, 통과) | ✅ |
| 17 | 기존 산출물 불변 + **최소 구조** | 추가형 | `backend/` 평면 8파일만 | ✅ |
| 18 | 모델→서버 공유변수 계약 | 사용자 #2 | `CONTRACT.md` | ✅ |
| 19 | 웹=Streamlit(둘 다) | 사용자 #5 | §1.3 | ✅ |

> 보완 메모: 스켈레톤은 **무설치(Node 내장 http)로 기동·검증까지** 범위. 실DB 연결 시 `db.js`가 `mysql2/pg`를 로드하고 `schema_mysql.sql` 적용이 남는다. 실시간 점수의 "모델 추론"은 **모델파트 제출본(`/models/submit`의 `predictions`) 또는 파이썬 사이드카**가 채운다(백엔드는 추론 X, 비대화 방지).
</content>

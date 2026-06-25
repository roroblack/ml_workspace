# backend — 이탈예측 운영 백엔드 (Node.js, 최소 구성)

**원칙**: 기존 폴더 구조 유지 + **필요한 파일만 추가**. 단일 `backend/` 폴더, 평면 구성(서브폴더 없음).
**역할**: 추론 안 함 — 각 **모델파트가 파일로 학습/예측한 결과 변수만 제출**받아 저장·조합·앙상블하고, **Streamlit**에 자료 제공. 시뮬 사이트 로그는 **Neon**에서 pull.

> 설계: [reports/05-6](../reports/05-6_프로그램_구조도_및_함수_변수_비전_명세서.md), [reports/19](../reports/19_백엔드_DB_MySQL_계획서.md), 계약: [CONTRACT.md](CONTRACT.md)

## 파일 (전부 — 평면)
| 파일 | 역할 |
| --- | --- |
| `server.js` | HTTP 서버 + 전 엔드포인트 + 도메인 로직(추론 X). **무설치 실행**(내장 http) |
| `db.js` | MySQL(운영)·Neon(시뮬로그) 연결. 미설정/미설치 시 **스켈레톤(메모리) 모드** |
| `check.js` | 무설치 스모크(로직 단언) |
| `schema_mysql.sql` / `seed.sql` | 운영 DDL / 데모 시드 |
| `package.json` / `.env.example` / `CONTRACT.md` | 메타 / 환경변수 / 모델→서버 계약 |

## 데이터 흐름
```
모델파트 ── POST /models/submit (공유변수 계약) ─▶ [Node 백엔드] ─▶ [MySQL] 운영데이터
[Sim Site] ─events▶ [Neon] sim_event_log ──pull──▶ [백엔드] ─/predict/realtime
[Streamlit] ◀── REST(JSON) ── [백엔드]  (이탈율·원인·앙상블·추천·리텐션 화면)
(여유 시) 백엔드 ── push ─▶ [Neon] ─▶ Sim Site 쿠폰/추천 (이탈예측→이탈방지)
```

## 실행
```bash
node check.js     # 무설치 검증(로직 단언)
node server.js    # 서버 기동(DB 없으면 스켈레톤 메모리 모드)
# 운영: .env 에 MYSQL_URL / NEON_URL / API_KEY 설정 후  npm i mysql2 pg dotenv
```

## API (x-api-key 필요, /health 제외)
| 경로 | 용도 |
| --- | --- |
| `POST /models/submit` | **모델파트 결과 제출(→ CONTRACT.md)**: registry 등록 + 배치예측 적재 |
| `GET /models` | 등록 모델 목록 |
| `POST /predict` | 모델 산출 확률 → risk/액션 변환 |
| `POST /predict/realtime` | Neon 시뮬 로그 pull → 모델파트 예측과 조인(표시용) |
| `POST /recommend` | view1/cart3/purchase5 가중 카테고리 추천 |
| `POST /retention-action` | risk별 쿠폰/리마인드 (옵션 Neon push) |
| `POST /ensemble` | 가중평균 앙상블 + 개선점 |

## 서버 선택 (사용자 질문 답)
- **스켈레톤 = Node 내장 http**(무설치·데모 즉시 실행).
- **운영 권고 서버 = Fastify**(JSON Schema 검증·직렬화 내장, 경량·고성능 → `/models/submit` 큰 페이로드 계약 검증에 적합). NestJS=과설계, Express=검증 수작업.
- **웹/화면 = Streamlit**(수업 기준): 고객 대시보드 + 시뮬 결과 모두 Streamlit. 백엔드는 REST API만 담당.

## 원칙
- 추론 코드 0. 1.27M 유저 피처·시퀀스·모델바이너리는 **파일+경로**(DB 미적재). 기존 `sample_project`/`preprocessing_project` 불변(추가형).

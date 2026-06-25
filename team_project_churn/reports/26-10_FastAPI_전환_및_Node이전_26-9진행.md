# 26-10. 백엔드 FastAPI 전환 · Node 이전 · 26-9 진행 결과

기준: 2026-06-21. 사용자 지시 3건 — ①백엔드를 FastAPI로 교체(Node는 sample_project에만 보존, 가지마에서 제거) ②프론트(대시보드 REST 교체)는 다른 팀원 몫이라 **백엔드만** 연결, 폴백은 비상시 ③26-9 다음 구현사항 진행. 기준 설계도 [19-2](19-2_최종_백엔드_설계도.md).

---

## 1. 결정 요약
| 항목 | 결정 |
| --- | --- |
| 운영 백엔드 | **FastAPI/Python**(19-2). 가지마 `backend/app/` |
| Node 백엔드 | **제거(가지마)** → `team_project_churn/sample_project/backend_node/`에 보존(백업·테스트) |
| 프론트 연결 | 대시보드 REST 교체는 **프론트 담당자 몫** → 본 작업은 백엔드 API 완성·검증까지 |
| 폴백(파일 직접읽기) | **비상시 구현**(현업 기준: 백엔드 우선, 폴백은 옵션). 지금은 미적용 |

> **"백엔드만 연결, 폴백은 비상시"가 맞다.** 현업에서도 3-tier는 백엔드를 단일 진입점으로 두고, 프론트는 그 계약(REST)만 바라본다. 폴백(백엔드 다운 시 파일 직접 읽기)은 가용성 보강용 옵션이지 1차 구현 대상이 아니다. 프론트팀은 19-2 §8 API 계약만 보면 되고, 백엔드 내부가 Node든 FastAPI든 무관(§7.4 계약 불변).

---

## 2. Node → sample_project 이전 / 가지마 제거
- 복사: `SKN32-2nd_GAJIMA_Dev/backend/*` → `team_project_churn/sample_project/backend_node/`
  - `node_modules`·비밀 `.env` 제외, `package-lock.json`·소스·`db`·`scripts` 보존. `MOVED_NOTE.md` 추가.
- 제거: 가지마 `backend/`의 Node 소스 `git rm`(working tree) + `node_modules` 삭제. 가지마 `backend/`는 이제 FastAPI 전용.
- 커밋: **하지 않음**(가지마 커밋 규칙 — 명령 시 feature/backend에만).

## 3. FastAPI 백엔드 (가지마 `backend/`) — 클린아키텍처
```
backend/
├── requirements.txt · .env.example · README.md · CONTRACT.md
├── db/schema_mysql.sql · migrate.py            # 15테이블, 적용 검증 완료
├── scripts/submit_models.py                    # 7모델 일괄 제출(26-9 P1 #3)
└── app/
    ├── main.py · config.py
    ├── interfaces/http/  health · models · predictions · dashboard (+deps: x-api-key)
    ├── schemas/          model_submit_schema(Pydantic)
    ├── application/      submit_model · dashboard · predict usecase
    ├── domain/           risk_level(위험등급·리텐션·앙상블, 순수)
    ├── infrastructure/mysql/  session(repository + memory 폴백, mysql.connector)
    ├── infrastructure/files/  artifact_store(eval 산출물 reader, SHAP→feature_importance 폴백)
    └── validators/       model_submit_validator
```
- **의존 방향**: router → usecase → domain/repository port → infrastructure adapter. 라우터는 SQL/파일 직접 접근 안 함(19-2 §4).
- **추론 없음**: 점수는 모델파트 제출(`POST /predict`), 백엔드는 위험등급·리텐션·로그·차트만.
- DB 미설정 시 **memory 폴백**으로 무설치 데모 가능. 본 검증은 **MySQL(project2db) 실연동**.
- 비밀 `.env`는 가지마 `.gitignore`로 제외 확인(추적 0).

## 4. 실검증 (uvicorn + 실 MySQL project2db)
서버: `uvicorn app.main:app --port 8090`, `db_mode=mysql` 확인.

| 검증 | 결과 |
| --- | --- |
| `GET /health` | `{ok:true, eval:true, db_mode:"mysql", server:"fastapi"}` |
| 인증 가드 | key 없이 `/models` → **401** |
| `submit_models.py` | **7/7 제출**(registry+evaluation), active=`CatBoost_Churn_v2`(AUC 0.791) |
| `GET /models` | mysql, 7건(tree5·linear1·sequence1) |
| `GET /dashboard/summary` | best=CatBoost 0.791, 7모델 비교표 |
| `GET /models/CatBoost/charts/roc` | ROC fpr/tpr JSON |
| `GET /models/CatBoost/charts/shap` | **feature_importance 폴백**(ndays 0.22·tenure 0.22·recency 0.18) |
| `POST /predict` ×3 | high/medium/low + 리텐션 액션 + prediction_log 적재 |
| `GET /predictions/top-risk?min_prob=0.5` | U_hi(0.91)만(중·저위험 필터) |
| `GET /predictions/latest?user_id=U_hi` | 최신 1건 |

## 5. 26-9 진행 현황
| # | 항목 | 상태 |
| --- | --- | --- |
| P1-1 | 대시보드 → 백엔드 REST 연결 | **프론트 담당자 몫**(백엔드 API는 완성·검증). 본 작업 범위 외 |
| P1-2 | 백엔드 상시 기동 + 검증 | ✅ uvicorn 기동·실 MySQL 검증 |
| P1-3 | 모델 7종 일괄 제출 | ✅ `scripts/submit_models.py` 7/7 |
| P1-4 | top-risk / latest 예측 API | ✅ 구현·검증 |
| P2~P4 | simulation_site·Neon pull·세션 GRU·진짜 SHAP·business_value·Fastify·얼굴 실검증·S1 정렬 | 미진행(잔여) |

## 6. 남은 작업(우선순위)
1. **(프론트팀)** 대시보드 `api_client.py`로 REST 호출 교체 — 백엔드 계약(CONTRACT.md) 제공 완료.
2. P2 실시간 루프: `simulation_site` + Neon `sim_event_log` + 백엔드 pull worker + scoring 어댑터.
3. P3: 세션 GRU 본격화·DL 스태킹·격리 env 진짜 SHAP·viz 3/5·business_value.
4. P4: feature/sequence 스냅샷 적재·얼굴로그인 실검증.

## 7-1. v4 전처리 반영 점검(추가 요청, 2026-06-21)
가지마에 v4 전처리가 반영됐는지 md5 비교로 점검.
- **반영됨(동일 해시)**: 전처리기 6종 `prep_*_v2.joblib`, 정본 `train/test_tabular_v2.parquet`+meta, Transformer `seq.npz`+meta, 리포트 6종, 추천 카탈로그 4종 → 서빙 정본 완비. (v4가 최신: v1→v2→v3→**v4**)
- **갭(미반영)**: per-model 학습입력본 `_v2_train.parquet`(6, 33MB)·베이즈 로그 `*bayes.json`(12)·v1 구버전. 19-2 §7.3 `dataset_path`가 가리키던 `data/processed/churn/models7/`가 부재했음.
- **조치**: per-model `_v2_train.parquet` 6개 + `bayes.json` 12개를 `data/processed/churn/models7/`로 복사(md5 6/6 동일). `submit_models.py`의 `dataset_path`를 per-model 실제 파일로 보정(Transformer는 seq npz). v1 구버전은 v2로 대체되어 미복사.

## 7. 비고(커밋)
가지마는 팀 공유 repo — 본 변경은 **working tree에만** 반영(미커밋). 커밋 명령 시 `feature/backend`에만 반영한다.

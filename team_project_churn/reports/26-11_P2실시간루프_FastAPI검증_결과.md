# 26-11. P2 실시간 이탈예측 루프 + FastAPI 백엔드 실검증 결과

기준: 2026-06-21. 사용자 지시 "FastAPI 교체 진행 + 26-9 다음 구현사항 진행". 설계 [19-2](19-2_최종_백엔드_설계도.md), 직전 [26-10](26-10_FastAPI_전환_및_Node이전_26-9진행.md).

---

## 1. FastAPI 백엔드 실검증 (uvicorn + 실 MySQL project2db) — 재확인 ✅
서버 `uvicorn app.main:app --port 8090`, `db_mode=mysql`.

| 검증 | 결과 |
| --- | --- |
| `GET /health` | `{ok, eval:true, db_mode:"mysql", server:"fastapi"}` |
| 인증 가드 | 키 없이 `/models` → **401** |
| `scripts/submit_models.py` | **7/7 제출**(registry+evaluation, mysql), active=CatBoost |
| `GET /dashboard/summary` | best=CatBoost 0.791, 7모델 |
| 차트 `roc`·`shap`(→feature_importance 폴백)·**`value-at-risk`(viz14)**·**`revenue-recovery`(viz15)** | 정상 JSON |
| **`POST /predict/realtime`** | **모델 직접 로드·추론** — CatBoost 1.0, LogReg 0.8814(없는 유저는 404) |
| `POST /predict` ×3 | high/medium/low + 리텐션 + prediction_log |
| `top-risk`·`latest` | 정상 |

→ **FastAPI의 핵심 이점 실현**: prep_*_v2.joblib을 백엔드가 직접 로드·점수(Node-Python 다리 불필요).

## 2. P3-13 business_value/VaR — 이미 완비 확인 ✅
`src/build_eval_artifacts.py`가 per-model로 생성 → `data/processed/evaluation/churn/<model>/{value_at_risk.json, business_value.json}`.
- viz14(VaR treemap): high/medium/low 세그먼트별 위험매출(고위험 99,446명·₩190,839).
- viz15(revenue recovery): top 5/10/20%별 VaR vs 기대회수(save_rate 8%, 쿠폰 ₩3,000 가정).
- 백엔드 `/models/CatBoost/charts/{value-at-risk, revenue-recovery}`로 서빙 확인.

## 3. P2 실시간 이탈예측 루프 — **신규 구현·검증** ✅ (이번 핵심)
"행동 → 실시간 추론 → 위험등급 → 리텐션/추천 push"의 닫힌 루프.

### 3.1 구성
| 파일 | 역할 |
| --- | --- |
| `simulation_site/index.html` | 정적 이커머스 시뮬(REES46 카탈로그). 보기/담기/빼기/구매 → 백엔드 |
| `backend/.../files/catalog_store.py` | seed CSV(상품 54K·카테고리 526·유사도 5K) reader + 유사카테고리 추천 |
| `backend/.../application/sim_usecase.py` | 세션 메모리 + **세션→v2 피처 집계** + 활성모델 직접추론 + 위험/리텐션/추천 |
| `backend/.../interfaces/http/sim_router.py` | `/sim/products·categories·event·score·reset` (CORS 허용) |
| `backend/.../mysql/session.py` `SimEventRepository` | `sim_event_log` 영속 |
| `db/schema_mysql.sql` | `sim_event_log` 테이블(16번째) |

### 3.2 세션→피처(전처리 투명)
v2 10피처를 세션 행동에서 집계: `n_view/n_cart/n_remove_from_cart/n_purchase/n_events`, `avg_price/purch_amt`(가격), `recency/tenure/ndays`(프로필 베이스라인: 신규/재방문/충성/이탈징후). 응답에 `features`를 그대로 노출(블랙박스 금지 원칙).

### 3.3 검증(실DB)
| 시나리오 | 행동 | 결과 |
| --- | --- | --- |
| **바운스** | 이탈징후 프로필 + view 1회 | churn **0.7781 high** → **push=True** + 유사카테고리 추천 |
| **인게이지** | 충성 프로필 + view×2·담기·구매 | churn **0.12 low** → push 없음(점수 순차 하락) |
| 영속 | — | `sim_event_log` 5건(2세션)·`prediction_log`에 sim 유저 반영(→ top-risk/대시보드 루프 닫힘) |

→ **이탈 잡기**의 핵심 데모: 바운스 유저를 즉시 식별→개입, 충성 유저는 정상. Neon 미사용 로컬(MySQL)로 동작, `NEON_URL` 설정 시 `schema_neon.sql`로 드롭인 전환.

## 4. 진행 중/잔여 (26-9)
| # | 항목 | 상태 |
| --- | --- | --- |
| P2 #5–8 | 시뮬·이벤트수집·스코어링·리텐션 push | ✅ 완료(본 리포트 §3) |
| P3-11 | **진짜 SHAP** | ✅ **완료** — 격리 `.venv_shap`(numpy2+shap0.47+imbalanced-learn)에서 **TreeSHAP** 산출(`gen_real_shap.py`). CatBoost/LightGBM/XGBoost per-model `shap_summary.json`(`method:TreeSHAP`) 덮어씀, 백엔드 `/charts/shap`이 실 SHAP 서빙(CatBoost recency 0.4267·tenure 0.2978·ndays 0.2267). 26-8의 "불가" 결론을 격리 env로 해소. LogReg(선형)·Transformer(시퀀스)는 피처중요도 유지 |
| P3-12 | viz3 cohort retention·viz5 train/val loss | 잔여(ML은 train_history 빈값, DL만 의미) |
| P4 | 스냅샷 적재·얼굴 실검증·S1 | 잔여 |
| P2 #6 | Neon pull worker | 로컬 MySQL로 대체 동작, Neon은 옵션(드롭인) |

## 5. 비고(커밋)
가지마는 팀 repo — 본 변경은 **working tree에만**. 커밋 명령 시 `feature/backend`에만. 비밀 `.env`·`node_modules`·`.venv*`는 gitignore.

# BUG-009. 대시보드 ↔ 시뮬 Churn Rate 헤드라인 불일치 (정책 이중 계산)

> 연계: [BUG-008](BUG-008_차트스키마_대시보드리팩터_시뮬UX_배치_2026-06-23.md), 26-19(시뮬·정책)

작성일: 2026-06-23
대상: `SKN32-2nd_GAJIMA_Dev/` (backend + dashboard_streamlit + simulation_site)

---

## 0. 요약

같은 유저·같은 세션인데 **헤드라인 Churn Rate가 두 화면에서 다르게** 표시됐다.
- 대시보드: **30.6%** (정책 표기 "Bounce 재척도")
- 시뮬: **45.3%** (= bounce/max)
- 3종(7일 11.0% · 하자드 2.3% · 바운스 45.3%)은 **양쪽 동일**.

원인은 **헤드라인을 두 곳에서 각각 계산**하면서 적용 정책이 달랐던 것. 단일 서버 소스로 통일해 해소.

---

## 1. 근본 원인

| 요소 | 동작 |
|---|---|
| 시뮬 헤드라인 | `/api/churn/predict` → 서버 `apply_policy(3종, **서버 정책**)` → `churn_probability` 표시 |
| 대시보드 헤드라인 | `live_session_diag`가 `_apply_policy(3종, **st.session_state 정책**)` 로 **로컬 재계산** |
| `_LAST_SIM` 캐시 | `churn_probability`에 **하자드**만 저장(정책 적용값 아님) → 대시보드가 쓸 단일값 부재 |

→ 대시보드 session_state엔 직전에 적용한 `bounce_scaled`가 남아 있고(30.6%), 서버 정책은 `max`(45.3%)였다.
서버 `_CHURN_POLICY`는 **인메모리**라 백엔드 재기동 시 `max`로 리셋되는데, 대시보드 세션 상태는 유지돼 **정책이 갈라짐**.

핵심: **(a) 헤드라인 이중 계산**(서버 vs 대시보드 로컬), **(b) 정책이 재기동에 안 남음**.

---

## 2. 수정 — 단일 서버 소스

1. **서버가 정책 적용값을 한 번만 계산**(`sim_usecase.churn_three`): `churn_rate = apply_policy(7일, 하자드, bounce)` + `policy_mode`를 결과와 `_LAST_SIM`에 저장.
2. **시뮬 라우터**: `/api/churn/predict`가 `three["churn_rate"]`를 그대로 반환(재계산 제거).
3. **대시보드**: `live_session_diag`가 `simd["churn_rate"]`(서버값)를 그대로 표시 — **로컬 `_apply_policy` 제거**(구버전 서버용 폴백만 유지).
4. **정책 파일 영속**(`set_churn_policy` → `data/processed/realtime/churn_policy.json`, 기동 시 `_load_churn_policy()`): 백엔드 재기동에도 정책 유지 → 리셋으로 인한 갈라짐 방지.

→ 시뮬·대시보드 모두 **서버가 계산한 동일한 `churn_rate`**를 표시. 정책을 바꾸면(대시보드 설정 → POST `/churn-policy`) 양쪽이 동시에 같은 값으로 갱신.

---

## 3. 검증

| 정책 | 시뮬 churn_probability | 대시보드 `/sim/user-score` churn_rate | 일치 |
|---|---|---|---|
| max | 77.5% | — | (정책 적용 확인) |
| ensemble | 43.5% | **0.4354 (43.5%)** | ✅ 동일 |
| bounce_scaled | 95.1% | (동일 소스) | ✅ |

- 영속 파일 `churn_policy.json` 생성 확인(재기동 유지).
- 대시보드 py 문법 OK · 프론트 `pnpm check` 0에러.

---

## 4. 변경 파일
- 백엔드: `application/sim_usecase.py`(`churn_three` churn_rate 저장 · 정책 파일 영속), `interfaces/http/sim_external_router.py`(churn_rate 단일 사용)
- 대시보드: `pages/02_dashboard.py`(`live_session_diag`가 서버 churn_rate 표시)

## 5. 재발 방지
- **헤드라인은 서버에서 한 번만 계산** — 클라/대시보드는 표시만. 같은 값을 두 곳에서 재계산하지 않는다.
- 설정성 전역 상태(정책)는 **파일/DB 영속** — 인메모리는 재기동에 사라져 화면 간 갈라짐을 만든다.

미커밋(가지마 feature/backend, 명령 시).

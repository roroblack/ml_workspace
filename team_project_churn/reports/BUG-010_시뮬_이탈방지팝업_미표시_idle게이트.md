# BUG-010 — 시뮬 이탈방지 팝업 미표시 (대시보드는 '다음 액션' 뜨는데)

- 작성일: 2026-06-23
- 심각도: Medium-High (핵심 시연 루프 — 이탈예측→이탈방지 액션 — 이 시뮬에서 안 보임)
- 상태: **FIXED**
- 영역: `backend/app/application/sim_usecase.py` (`decide_action`)

## 증상
- 시뮬 위젯: Churn Rate **62.1%**(이탈 Bounce 61.1%) 표시.
- **대시보드** 개인진단 ⚡실시간 탭엔 "다음 액션"이 정상 표시.
- 그러나 **시뮬 사이트의 이탈방지 팝업(ChurnActionPanel)이 안 뜸**.

## 원인 (root cause)
시뮬 팝업은 백엔드 `/api/churn/predict` 의 `recommended_action.action_type` 가 `'none'` 이 아닐 때만 렌더된다(`ChurnActionPanel`: `if (!action) return null`).

`recommended_action` 은 `sim_usecase.decide_action(churn_rate, events)` 산출인데, **3개 시나리오 전부 "무활동(idle) 임계"를 요구**했다:
- ① 장바구니+미구매 → `idle_sec >= CART_IDLE_SEC(5s)`
- ② SNS → `n_events==0`(첫 접속) 또는 `idle_sec >= SNS_IDLE_SEC(30s)`
- ③ 조회만(view≥3) → `idle_sec >= CART_IDLE_SEC(5s)`
- 그 외 → `action_type:'none'`

시뮬 위젯은 **4초마다** 최신 이벤트로 채점 → **활발히 둘러보는 중**이면 마지막 이벤트가 방금이라 `idle_sec`가 작다(<5s). 따라서 churn이 높아도 어떤 시나리오도 매칭되지 않아 **'none' → 팝업 미표시**.

반면 **대시보드**의 "다음 액션"은 `churn_rate >= 0.5` 만으로 클라이언트가 자체 표시 → **시뮬↔대시보드 불일치**.

## 수정
`decide_action` 에 **고이탈 즉시 트리거** 도입(대시보드와 동일 기준):
- `ACTION_P = 0.5` 신설. `hot = churn_rate >= ACTION_P`.
- ① 장바구니+미구매: 게이트 `idle≥5s` **또는 hot**.
- ③ 조회만(view≥3): 게이트 `idle≥5s` **또는 hot**.
- ② SNS: 기존 유지(첫 접속/장시간 무활동 — 활동 중 SNS 강요 방지).
- ④ **폴백 신설**: 시나리오 미매칭이라도 `hot` 이면 등급별 쿠폰(`coupon_grade(p)`) + 추천 → 활동 적은데 bounce↑ 같은 케이스도 노출.
- 즉 **churn_rate ≥ 0.5 이면 idle 없이도 액션** → 대시보드 '다음 액션'과 1:1 정합. 0.5 미만이면 양쪽 모두 미표시(스팸 방지).

(프론트 `ChurnActionPanel` 은 변경 불필요 — `action_type≠'none'`이면 정상 렌더.)

## 검증 (`/api/churn/predict`)
| 세션 | churn% | bounce% | action_type | 결과 |
| --- | ---: | ---: | --- | --- |
| 장바구니+미구매(활발, idle<5s) | 59.8 | 59.9 | **discount_related** | 팝업 표시 ✅ (이전 none) |
| 구매 포함 고이탈 | 60.2 | 60.1 | **discount**(high_churn 폴백) | 표시 ✅ |
| 조회만 churn<0.5 | 47.4 | 53.7 | none | 미표시(대시보드도 동일 — 일관) |

## 파일
- `backend/app/application/sim_usecase.py` — `ACTION_P` + `decide_action` 고이탈 트리거/폴백

## 비고
- 시뮬 위젯은 4초 폴링이라 백엔드 재기동 후 다음 폴링부터 팝업이 뜬다(프론트 무수정).
- 액션 기준값(0.5)은 대시보드 churn-policy(max/ensemble/...)로 산정된 headline `churn_rate` 에 적용 → 정책 바꾸면 양쪽이 함께 반영된다.

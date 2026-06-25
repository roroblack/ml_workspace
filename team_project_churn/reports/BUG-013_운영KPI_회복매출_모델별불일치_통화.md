# BUG-013 — 운영 KPI '회복 예상 매출액' 모델별 들쭉날쭉(0·소액) + 통화(₩) 오표기

- 작성일: 2026-06-23
- 심각도: Medium (지표 신뢰성·비교 가능성 훼손, 발표 시 오해)
- 상태: **FIXED**
- 영역: `backend/app/application/dashboard_usecase.py` · `infrastructure/files/dataset_reader.py` · `dashboard_streamlit/pages/02_dashboard.py`

## 증상
- 운영 탭 '💰 회복 예상 매출액'이 모델마다 **스케일이 천차만별**: XGBoost ₩4,608,120 · LightGBM ₩19,805,000 · CatBoost ₩152 · **RandomForest ₩0** 등. "₩(원화)" 표기인데 값이 비현실적.

## 원인
모델별 `business_value.json` 이 **제각각**으로 생성됨:
| 모델 | 공식/필드 | avg_revenue 가정 | 결과 |
|---|---|---|---|
| catboost·logreg | expected_recovery(top%) | **51.02**($급) | 152·221 |
| decisiontree·xgboost | expected_recovery(top%) | **42000**(₩급) | 4.6M |
| lightgbm | estimated_total_value_KRW(confusion) | 별도 공식 | 19.8M |
| randomforest | expected_recovery | 51.02 | **0,0,0(계산 0/버그)** |

→ ① **avg_revenue 가정 불일치(51 vs 42000, 800배)**, ② **단위 혼용($·₩)**, ③ **RF 전부 0**, ④ lightgbm은 아예 다른 공식. `_expected_revenue_recovery`가 이 이질적 파일을 그대로 읽어 **비교 불가·신뢰 불가**. 게다가 데이터 실제 통화는 **원화가 아님**(eval revenue mean 4.32·max 1850 = 소액 달러급).

## 수정 — 단일 공식·실데이터로 일관 계산
`dataset_reader.revenue_recovery(model)` 신설: **eval_predictions(실데이터)** 에서
```
회복매출 = Σ(정탐 이탈자 TP[y_pred=1 & y_true=1]의 실제 revenue) × SAVE_RATE(0.08)
```
- 전 모델 **동일 공식·동일 가정(save_rate 8%)** + **실제 per-user revenue** 사용 → 비교 가능, 0/버그 제거.
- `dashboard_usecase._expected_revenue_recovery` 가 이 함수를 호출(business_value.json 의존 제거).
- 대시보드 라벨 **₩ → $**(데이터 통화 = 소액 달러급), help에 산식 명시.

## 검증 (`/dashboard/summary?model=`)
| 모델 | 회복매출 | 고위험 |
|---|---:|---:|
| DecisionTree | $20,363 | 98,894 |
| LogReg | $17,928 | 99,549 |
| CatBoost | $17,756 | 99,446 |
| XGBoost | $17,078 | 100,661 |
| RandomForest | $16,356 | 102,268 |
| LightGBM | $16,337 | 99,816 |
| Transformer | $0 | 18 |

→ 6개 부스트 모델 **$16.3k~$20.4k 로 일관**(이전 0·152·19.8M 혼재 해소).

## 잔여
- **Transformer = $0**: 통합 `eval_predictions.parquet` 에 Transformer 행이 없음(별도/소규모 eval). 모델팀이 Transformer 예측을 동일 포맷으로 합치면 자동 산정됨. (현재는 시퀀스 모델 eval 분리)
- 통화 단위: REES46 원천이 정확히 USD인지 미확정 → "데이터 통화(달러급)"로 표기. 명세 확정 시 라벨 조정.

## 파일
- `backend/app/infrastructure/files/dataset_reader.py`(`revenue_recovery`+`SAVE_RATE`)
- `backend/app/application/dashboard_usecase.py`(`_expected_revenue_recovery` 일관화)
- `dashboard_streamlit/pages/02_dashboard.py`(₩→$ + help)

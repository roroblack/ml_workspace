# 계획서 B — SHAP 설명가능성 + 임계값/비용 최적화

## 목표
1. **SHAP**으로 모델이 "왜" 특정 고객을 이탈로 예측하는지 설명 → 리텐션 액션과 연결 (발표 임팩트 ↑)
2. **임계값(threshold) 최적화**로 비즈니스 목표(이탈자 포착 = Recall)에 맞게 운영점 결정

## 배경 / 근거
- 실증 결과 AUC는 ~0.87 천장이지만, **임계값 0.5→0.33 조정만으로 F1 0.60→0.64, Recall 0.48→0.64** 로 개선됨 (`outputs/metrics_advanced.json`).
- 이탈 예측의 가치는 "예측"이 아니라 **"누구에게 무엇을 할지"** → SHAP로 개별 사유 제시가 핵심.

## 작업 항목 (체크리스트)
- [ ] **1. 라이브러리**: `pip install shap` (현재 미설치)
- [ ] **2. SHAP 전역 분석** (`src/explain_shap.py`):
  - 최적 모델(GradientBoosting/HistGB) + `shap.TreeExplainer`
  - `summary_plot`(피처 영향 전체) → `outputs/shap_summary.png`
  - 피처별 평균 |SHAP| 막대 → 중요 피처 순위 (피처 중요도와 교차 검증)
- [ ] **3. SHAP 개별 분석**:
  - 이탈 위험 상위 고객 몇 명의 `force_plot`/`waterfall` → "이 고객은 나이↑·비활성·상품 1개라 이탈 위험" 식 해석
- [ ] **4. 임계값 최적화** (`src/threshold_opt.py`):
  - Precision-Recall 곡선 + F1 최대 임계값 탐색
  - **비용 기반**: FN(이탈 놓침) 비용 vs FP(불필요 캠페인) 비용 가중 → 기대비용 최소 임계값
  - 임계값별 Recall/Precision/F1 표 + 곡선 → `outputs/threshold_curve.png`
- [ ] **5. 확률 보정(선택)**: `CalibratedClassifierCV`로 예측확률 신뢰도 향상
- [ ] **6. 결과서 반영**: 학습 결과서에 SHAP 그림 + 권장 운영 임계값 추가

## 산출물
- `src/explain_shap.py`, `src/threshold_opt.py`
- `outputs/shap_summary.png`, `shap_force_*.png`, `threshold_curve.png`
- 학습 결과서 보강(설명가능성 + 운영 임계값 섹션)

## 예상 시간 / 난도
- 약 1~2시간 / **중**. (TreeExplainer는 트리 모델에서 빠름)

## 리스크 / 주의
- shap 설치/버전 호환(matplotlib). 트리 모델엔 TreeExplainer, MLP엔 KernelExplainer(느림) 사용.
- 비용 가중치는 가정값 → 발표 시 "가정"임을 명시.

## 우선순위
**상** — 적은 노력으로 **차별화 효과가 가장 큼**(설명가능성 + 실질 Recall 향상). 먼저 진행 권장.

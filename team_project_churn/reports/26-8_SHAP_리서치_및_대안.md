# 26-8. SHAP — 무엇이고, 왜 안 됐고, 대안은? (리서치)

## 1. SHAP이 뭔가
- **SHAP(SHapley Additive exPlanations)**: 게임이론의 **Shapley value**로 "각 피처가 이 예측을 얼마나 밀었나"를 계산하는 설명가능성(XAI) 기법.
- 특징: **지역(local) 설명**(개별 고객 1명의 이탈 사유) + **전역(global)**(mean|shap|로 피처 랭킹). 트리계열은 **TreeSHAP**로 빠르게(정확) 계산.
- 발표 가치: "이 고객이 이탈 위험인 이유 = recency↑·n_purchase↓…" 같은 **사유 분해**.

## 2. 왜 우리 환경에서 안 됐나
- 우리 env: **numpy 1.26.4** (scipy·scikit-learn·pandas·catboost가 numpy 1.x ABI로 빌드됨).
- 리서치 결과(아래 출처):
  - **shap 0.46.0+ 는 numpy≥2 요구**(2024-06부터 numpy2 지원).
  - **shap 0.45 이하**는 **numpy≥1.24에서 깨짐**(`np.bool`/`np.obj2sctype` 제거됨).
  - 즉 **numpy 1.26.4와 맞는 shap 버전이 사실상 없음**.
- 실제 증상: `pip install shap`(0.52) → **numpy가 2.0으로 강제 업그레이드** → numpy 1.x로 빌드된 scipy/sklearn import가 `numpy.core.multiarray failed to import`로 **전부 깨짐**. → 즉시 `numpy<2` 복구 + shap 제거로 정상화.

## 3. 대안 (검토)
| 대안 | 설명 | 의존성 | 채택 |
| --- | --- | --- | --- |
| **네이티브 피처 중요도** | 트리=`feature_importances_`(gain), 선형=`|coef|` | 없음 | ✅ **채택**(viz13 대체) |
| sklearn `permutation_importance` | 모델 불문 전역 중요도(셔플) | 없음 | 보조 가능 |
| **별도 numpy2 env + shap 0.46+** | 격리된 환경에서 TreeSHAP → `shap_summary.json` export | 별도 venv | 🔶 진짜 SHAP 필요 시 권장 |
| 구버전 shap + numpy<1.24 | numpy를 더 내려야 함(우리 스택과 충돌) | — | ❌ 비현실적 |
| captum/LIME/eli5/treeinterpreter | 기타 XAI | 추가 | 옵션 |

## 4. 우리 결론·구현
- **데모: 네이티브 피처 중요도로 viz13 대체**(`pp_feature_importance.py` → `feature_importance.json`). 트리 gain/선형 coef, top10.
  - 결과(예): CatBoost ndays 0.22·tenure 0.22·recency 0.18 → **recency 지배 발견과 일치**(전역 설명으로 충분).
- **진짜 SHAP(개별 사유 분해)가 발표에 꼭 필요하면**: `python -m venv shap_env; pip install "numpy>=2" "shap>=0.47" catboost`로 **격리 env** 만들어 거기서만 TreeSHAP 계산 → `shap_summary.json` 산출 후 대시보드에 배치(대시보드는 있으면 SHAP, 없으면 피처중요도 자동 폴백).

## 출처
- [shap/shap #3707 Numpy2 Support](https://github.com/shap/shap/issues/3707)
- [shap/shap #3716 incompatibility between numpy and shap](https://github.com/shap/shap/issues/3716)
- [slundberg/shap #2911 Shap not compatible with numpy>=1.24.0](https://github.com/slundberg/shap/issues/2911)
- [SHAP Releases](https://github.com/shap/shap/releases) · [shap · PyPI](https://pypi.org/project/shap/)
- [NumPy 2.0.0 Release Notes](https://numpy.org/devdocs/release/2.0.0-notes.html)

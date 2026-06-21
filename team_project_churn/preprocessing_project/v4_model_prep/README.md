# v4_model_prep — 모델별 전처리 (하나씩, 실시간 안전)

[17-7-1] 계획서 §4.1을 **모델 하나씩** 실행하는 버전. 각 모델에 대해 **전처리+HP 베이지안 탐색**을 돌리고, 고른 기술·옵션별 결과·선택 이유·컬럼/사용법을 **인수인계 리포트**로 남겨 학습 담당이 바로 이어가게 한다.

## 설계 결정 (사용자 #4 우려 반영)
1. **실시간 안전 우선**: 시뮬 사이트 로그 → 실시간 이탈예측을 고려해, v4의 입력 피처는 **모두 최근 이벤트 스트림에서 재계산 가능**한 것만 정본으로 쓴다(현재 10피처). → `realtime_compat_test.py`로 **이벤트→피처→예측**이 됨을 검증(PASS).
2. **추가형·무해**: 신규 피처는 **추가 컬럼**(별도 파일)로만. 기존 모델은 **이름으로 10피처만 선택**해 추가 컬럼을 무시 → 안 깨짐(테스트 ②로 검증).
3. **무거운 DL 임베딩(seq_emb)은 선택적 오프라인**: SASRec/TiSASRec hidden state 스태킹은 성능 +α용 **오프라인 캐시 피처**로 분리. 실시간 경로의 **하드 의존성 아님**(연산비용↑). 즉 실시간은 Wide(통계) 피처로, 오프라인 배치는 +임베딩으로.

## 개념 설명 (질문 답)
- **HP 탐색공간 = "하이퍼파라미터 탐색공간"**: 모델이 학습 전 사람이 정하는 값들(HP)과 그 **후보 범위**의 집합. 예) DecisionTree → `max_depth∈[3,20]`, `min_samples_leaf∈[1,80]`, `ccp_alpha∈[0,0.02]`. optuna가 이 공간을 **베이지안(TPE)** 으로 뒤져 최적점을 찾는다. v4는 **전처리 선택 + HP를 한 study로 공동탐색**.
- **권고 아키텍처(스태킹) 쉬운 설명**: ①(L0) DL이 유저의 이벤트 시퀀스를 읽어 **요약 벡터(임베딩)** 를 뽑는다 → ②(L1) 그 벡터를 **통계피처 옆에 붙여 LightGBM/CatBoost로 최종 이탈 분류**. "DL=피처 생성기, GBM=최종 판정". recency 지배인 이탈에서 GBM이 가장 강하고, 임베딩이 시퀀스 정보를 +α로 보탠다. → **실시간엔 부담** → 오프라인 전용.
- **10개 추가 컬럼 호환?**: 추가 컬럼은 새 파일에만. 기존/실시간은 **컬럼 이름으로 10개만** 읽으므로 영향 0(테스트 ②). 새 컬럼은 그걸 학습한 모델만 사용.

## 파일 (네이밍 규칙: `pp_`=실제 전처리 / `models_`=모델용 / `test_`=테스트·데모 / `legacy/`=구버전)
| 파일 | 종류 | 역할 |
| --- | --- | --- |
| `src/pp_features.py` | 전처리 | raw→피처(category/brand/세션/remove/price 살림) → `processed_eventbase/{train,test}_tabular_v2.parquet` |
| `src/pp_catalog.py` | 전처리 | 추천 카탈로그 + 유사분류 사전 |
| `src/models_bayes.py` | 모델 | 7모델 전처리+HP 베이지안(**v2 피처**, Brier/ECE, 30행텍스트). 자체완결(구 의존 없음) |
| `src/models_transformer.py` | 모델 | 시퀀스(Transformer) 정규화 탐색+검증 |
| `src/test_realtime_compat.py <Model>` | 테스트 | 실시간 재현 + 10컬럼 호환 검증 |
| `src/test_session_preview.py` | 테스트/데모 | 이벤트·세션 미리보기(바운스·Δt) PNG/CSV |
| `src/legacy/` | 구버전 | `models_bayes_one.py`(base-10), `preview_eventlevel.py` |
| `output/<Model>/` | 산출 | `prep_*.joblib`·`*_bayes.json`·`*_train.parquet`·`*_전처리리포트.md`·**`*_first30.txt`(30행 텍스트)** |

## 진행 현황 (하나씩) — 모델별 폴더 `output/<Model>/`
| # | 모델 | 상태 | CV PR-AUC | **Feb AUC** | 임계값 | 최적 전처리 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | DecisionTree | ✅ | 0.9288 | 0.7773 | 0.42 | none · classweight |
| 2 | RandomForest | ✅ | 0.9365 | 0.7892 | 0.54 | robust · none |
| 3 | LogReg | ✅ | 0.9356 | 0.7860 | 0.59 | minmax · classweight |
| 4 | **XGBoost** ★ | ✅ | 0.9370 | **0.7904** | 0.53 | robust · classweight |
| 5 | LightGBM | ✅ | 0.9370 | 0.7902 | 0.55 | standard · none |
| 6 | CatBoost | ✅ | 0.9370 | 0.7902 | 0.52 | none · none |
| 7 | Transformer | ✅ | — | 0.7855(val) | — | 시퀀스 norm=log |

### v1(기본10피처) vs v2(category/brand/세션/remove/price 살린 22피처) — Feb OOT
| 모델 | v1 AUC | **v2 AUC** | Brier(v2) | ECE(v2) | Δ |
| --- | --- | --- | --- | --- | --- |
| DecisionTree | 0.7773 | 0.7754 | 0.116 | 0.035 | −0.002 |
| RandomForest | 0.7892 | 0.7900 | 0.112 | 0.030 | +0.001 |
| LogReg | 0.7860 | 0.7871 | 0.112 | 0.029 | +0.001 |
| XGBoost | 0.7904 | 0.7905 | 0.111 | 0.025 | +0.000 |
| LightGBM | 0.7902 | 0.7905 | 0.111 | 0.026 | +0.000 |
| **CatBoost** ★ | 0.7902 | **0.7910** | 0.111 | 0.026 | **+0.001(신규 최고)** |

- **해석**: 살린 피처(category/brand/세션/remove/price)로 **부스팅·RF·LogReg 소폭 일관 향상**, CatBoost 0.7910 신규 최고. 향상폭이 작은 건 **recency 지배 구조**(17-7-0) 때문 — 큰 도약은 시퀀스/스태킹(계획서23 C·D) 몫.
- 산출(`models_bayes.py`): `prep_<Model>_v2.joblib`·`<Model>_v2_bayes.json`·`<Model>_v2_train.parquet`·`<Model>_first30.txt`.

- **결론(7모델)**: 부스팅 3종 최강(**XGBoost 0.7904** ≈ LGBM·CatBoost 0.7902), Transformer는 내부 val 0.7855(시퀀스, log 정규화)로 부스팅급이나 train17주/test3주 길이상이로 Feb 직접평가는 정렬 후. DT 최약(0.7773). [17-6-2]의 "부스팅 일관 최강"과 정합.
- 각 모델 폴더 산출: `prep_{Model}.joblib`(전처리+모델+isotonic보정+임계값) · `{Model}_bayes.json`(옵션별 결과) · `{Model}_{train,test}.parquet`(전처리본 백업) · `{Model}_전처리리포트.md`(인수인계).
- 전 모델 **실시간/10컬럼 호환 테스트 PASS**(이벤트→10피처→예측, 추가컬럼 무해).
- 부스팅은 추후 `last_cat`/`brand` 네이티브 범주형 추가형 컬럼 탐색 여지(호환 확인 후).

## X/Y 요약
- X(현재): 정본 10피처(realtime-safe). (향후 추가형: `last_cat_id`, `mean_dt_sec`, `dwell_*`, `cat_entropy`, `brand_loyalty` 등 — 전부 실시간 계산 가능한 것만)
- Y: `churn`(주), `churn_no_purchase`(보조, 보유).

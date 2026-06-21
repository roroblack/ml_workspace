# 17-6-5. 계획서 — 5m → models7 차이·부족점·개선안 + 호환성 정합(재구성)

목적: **(1)** `processed_5m`(5개월 정본)에서 `models7`(7모델 데이터셋)으로 넘어오며 **무엇이 바뀌고 무엇이 부족한지**, **(2)** 지금 작업으로 **무엇이 개선되는지**, **(3)** `models7`가 `5m`·배포본과 **호환이 깨진 지점**과 **필요 시 models7부터 호환되게 재구성하는 계획**을 한눈에 정리한다.
관련: [17-5-0](17-5-0_5개월_이탈예측_전처리_계획서.md) · [17-6-1](17-6-1_백엔드_DB_변경사항.md) · [17-6-2](17-6-2_7개모델별_최적전처리_데이터셋.md) · [17-6-4](17-6-4_얼리스탑_적용_및_추가개선_점검.md)

---

## 0. 한눈에 — 전체 파이프라인과 단절 지점

```
 RAW 5개월 zip (2,069만 이벤트)
        │  prep_5month.py
        ▼
 ┌─────────────────────────────────────────────────────────┐
 │ [A] processed_5m  (정본·전체 1.27M 유저)                 │
 │   train/test_tabular.parquet  10피처 + churn + churn_np  │  ◀ 라벨 2종, raw
 │   train/test_seq.npz          주별 [N,17,3]/[N,3,3]      │
 └─────────────────────────────────────────────────────────┘
        │  bayes_5m.py / finalize_5m_opt.py  (코호트 recency≤7)
        ▼
 ┌─────────────────────────────────────────────────────────┐
 │ [B] cohort + ML/DL 최적본                                │
 │   train/test_cohort_tabular.parquet  raw 10피처 + churn  │  ◀ churn_np 드롭
 │   ML_opt: log+minmax  +  ML_opt_scaler.joblib  ✅저장     │
 │   DL_opt: raw 최근4주 [N,4,3]                            │
 └─────────────────────────────────────────────────────────┘
        │  models7_opt.py / models7_earlystop.py
        ▼
 ┌─────────────────────────────────────────────────────────┐
 │ [C] models7  (모델 7종 학습 데이터셋)                    │
 │   {Model}_train/test.parquet  ← 모델마다 다른 baked 값   │  ✗ 전처리기 유실
 │   Transformer_train(17주)/test(4주)_seq.npz              │  ✗ 라벨 churn만
 │   최종 = CatBoost ES Feb AUC 0.7902                      │
 └─────────────────────────────────────────────────────────┘

        ╳╳╳  단절  ╳╳╳

 ┌─────────────────────────────────────────────────────────┐
 │ [D] 배포본 sample_project (운영)                         │
 │   feature_schema v1 = 7피처(active_days)  ·  seq 14일(daily)│  ✗ 스키마 불일치
 │   predictor.py + seq_scaler.npz + LSTM  ·  churn만        │
 └─────────────────────────────────────────────────────────┘
```

> 핵심: **[C] models7는 [A/B]에서 "변환된 값으로 구워진(baked)" 데이터셋인데 그 변환기가 저장되지 않았고, 라벨도 1종으로 줄었으며, [D] 배포본은 아예 다른 7피처·일별 스키마**라서 서로 그대로 붙지 않는다.

---

## 1. 데이터 계보 (각 단계 산출물)

| 단계 | 생성 스크립트 | 산출물 | 모집단 | 피처 | 시퀀스 | 라벨 | 전처리기 저장 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **A 정본** | `prep_5month.py` | `processed_5m/{train,test}_tabular.parquet`, `_seq.npz`, `meta_5m.json` | 전체 1.27M | **10**(raw) | 주별 17/3 | **churn + churn_np** | — (raw) |
| **B 코호트·최적** | `bayes_5m.py`,`finalize_5m_opt.py` | `{train,test}_cohort_tabular.parquet`, `*_ML_opt.parquet`+`ML_opt_scaler.joblib`, `*_seq_DL_opt.npz` | recency≤7 코호트 | 10 | 4주 | churn | **ML_opt_scaler ✅** |
| **C 7모델** | `models7_opt.py`,`models7_earlystop.py` | `processed_5m/models7/{Model}_{train,test}.parquet`, `Transformer_*_seq.npz`, `models7_opt/earlystop.json` | 코호트 | 10(모델별 변환) | 17/4 | **churn만** | **❌ 모델별 스케일러 유실** |
| **D 배포본** | `sample_project/src/build_features.py` | `processed/features.csv`, `sequences.npz`, DB, `seq_scaler.npz`,`lstm.pth` | 샘플 | **7**(active_days) | **일별 14** | churn | seq_scaler ✅ |

10피처(A/B/C) = `recency_days, tenure_days, ndays, n_events, n_view, n_cart, n_remove_from_cart, n_purchase, avg_price, purch_amt`
7피처(D) = `recency_days, n_view, n_cart, n_purchase, n_events, active_days, avg_price`  ← `ndays`를 `active_days`로 부르고 4개(tenure/remove/purch_amt …) 없음

---

## 2. 5m → models7 「차이」와 「부족한 것」

### 2.1 무엇이 바뀌었나 (A/B → C)
```
 A/B 정본                         C models7
 ─────────                        ─────────
 raw 값 1벌(공통)        ──▶      모델마다 다른 baked 값 7벌
 churn + churn_np        ──▶      churn 1종만           (라벨 축소)
 전체/코호트 단일기준     ──▶      코호트만
 ML_opt_scaler 저장       ──▶      모델별 scaler 함수내 fit→버림 (유실)
 길이 17/3(주)·4(DL)      ──▶      Transformer 17/4 비대칭 유지
```

### 2.2 부족한 것 (models7 시점의 결손) ★
| # | 부족점 | 근거(코드) | 영향 |
| --- | --- | --- | --- |
| 1 | **전처리기 미저장** | `models7_opt.transform()`가 모델별 scaler를 내부 fit 후 폐기. 저장은 B의 `ML_opt_scaler`(1종)뿐 | **서빙 재현 불가** — 새 유저를 같은 변환공간으로 못 보냄 |
| 2 | **라벨 1종** | `models7_opt`가 `churn`만 보존, `churn_no_purchase` 드롭 | 재구매-이탈 분석/보조타깃 불가 |
| 3 | **정본 7벌 분산** | `{Model}_train.parquet` 각각 값 다름 | 단일 비교기준 없음 → 앙상블·스태킹·공정비교 곤란 |
| 4 | **모델 HP 미튜닝** | 17-6-4 P1: 전처리만 튜닝, depth/lr/reg 미탐색 | 추가 향상분 미회수(최대 누락) |
| 5 | **확률 보정/임계값 없음** | 17-6-4 P1: raw 확률·0.5 고정 | risk 임계값(0.35/0.65) 신뢰성 X |
| 6 | **시퀀스 길이 비대칭** | train 17주 / test 4주 | Feb 직접평가 불가, 서빙 길이와도 불일치 |

---

## 3. 호환성 단절 — models7 vs 5m vs 배포본 ★

> 사용자 직감 확인: **models7 산출물은 5m 정본과도, 운영 배포본과도 호환되지 않는다.**

| 축 | [D] 배포본 v1 | [A] 5m 정본 | [B] 코호트/ML_opt | [C] models7 | 호환? |
| --- | --- | --- | --- | --- | --- |
| 피처 수 | **7** | 10 | 10 | 10 | **D ≠ A/B/C** |
| 활동일수 이름 | `active_days` | `ndays` | `ndays` | `ndays` | 이름 불일치 |
| 라벨 | churn | churn+np | churn | **churn만** | C가 축소 |
| 시퀀스 | **일별 14** | 주별 17/3 | 4주 | 17/4 | **모두 상이** |
| 모집단 | 샘플 | 전체 1.27M | 코호트 | 코호트 | A↔B/C 상이 |
| 데이터 표현 | raw daily | raw | log+minmax(1) | **모델별 baked** | C 비정합 |
| 전처리기 | seq_scaler✅ | 없음 | ML_opt_scaler✅ | **유실❌** | C 서빙불가 |
| model_registry 확장 | 미반영 | — | — | 미반영 | 레지스트리 분기 불가 |

**결론**: ① models7→배포본은 **피처·시퀀스 계약 자체가 다름**(7/일별 ↔ 10/주별). ② models7 자체가 **전처리기 유실로 재현 불가** + 라벨 축소. ③ 즉 **models7부터 "정본·전처리기 분리 + 스키마 통일"로 다시 만들어야** 5m·배포본과 한 줄로 이어진다.

---

## 4. 개선 설계 — "데이터(raw 정본) ⟂ 전처리(저장된 변환기) ⟂ 모델"

baked parquet을 정본 취급하던 구조를 끊고, **변환기를 산출물로** 만든다.

```
 [현재 C]  cohort.parquet ──(내부 fit·폐기)──▶ Model별 baked.parquet  ✗재현불가
 ───────────────────────────────────────────────────────────────────
 [개선 C']  train_cohort_tabular.parquet (raw·10피처·두 라벨 = 단일 정본)
               │
               ├─ prep_{Model}.joblib      ← log+scaler+imbalance 파이프라인(train fit, 저장)
               ├─ models7_manifest.json    ← model→preprocessing_config→artifact→metric
               └─ feature_schema.yaml v2   ← 10피처(ndays) + 시퀀스 granularity 고정
                       │
                       ▼  서빙: raw 피처 로드 → 저장된 prep transform → 예측  (재현 ✅)
                  model_registry(+preprocessing_config,dataset_path,model_type)
```

핵심 4원칙:
1. **단일 raw 정본** = `train/test_cohort_tabular.parquet`(10피처 + `churn` + `churn_no_purchase` 모두 유지). 변환본은 정본이 아니라 캐시.
2. **전처리기 = 산출물**: 모델별 `prep_{Model}.joblib` + `models7_manifest.json`. 서빙은 transform만(누수 차단·재현).
3. **스키마 단일화** `feature_schema.yaml v2`: 10피처 canonical(`ndays`), 시퀀스 **주별 고정길이 + 패딩 마스크**로 train/test/serve 통일(17/4 비대칭 제거).
4. **추가형(회귀안전)**: 배포본 v1 산출물(7피처·일별·seq_scaler·lstm)은 **그대로 둠**. v2를 옆에 얹고 단계적 컷오버. DB는 [17-6-1]대로 **컬럼 추가**(기존 의미 불변, 기본값).

---

## 5. models7부터 호환되게 재구성 — 작업 계획(추가형)

| 단계 | 작업 | 입력 → 산출 | 회귀 안전장치 |
| --- | --- | --- | --- |
| **S1 스키마 확정** | `feature_schema.yaml v2`(10피처 canonical + seq granularity) + `active_days↔ndays` 매핑표 | 새 파일/섹션 | v1 섹션 유지 |
| **S2 정본 보강** | A/B가 **두 라벨 모두** 코호트 parquet에 유지되게 점검(이미 A엔 있음, B에 `churn_no_purchase` 복원) | `*_cohort_tabular.parquet`(+churn_np) | 기존 컬럼 불변·추가만 |
| **S3 models7 재작성** | `models7_opt.py` 개정: baked parquet 대신 **`prep_{Model}.joblib` + `models7_manifest.json`** 산출. 변환본은 `cache/`에만 | 정본 → 변환기+manifest | 기존 `models7/*.parquet`는 `legacy/`로 이동(삭제X) |
| **S4 시퀀스 통일** | train/test/serve 동일 granularity·길이(주별 고정 N + mask) | `*_seq_v2.npz` | 기존 npz 유지 |
| **S5 모델링 개선** | [17-6-4 P1] HP 베이지안 공동탐색 + 확률보정(isotonic) + 임계값 최적화 → manifest에 기록 | manifest 보강 | 별도 단계 |
| **S6 DB 반영** | [17-6-1] `model_registry(+preprocessing_config…)`, `feature_user_snapshot(+churn_np,cohort_flag…)`, `sequence_snapshot(+dataset_tag)` ALTER(기본값) | 마이그레이션 | 기존 행 유효 |
| **S7 서빙 어댑트** | predictor를 `model_registry+prep_joblib` 기반 분기로(7피처 v1 경로는 폴백 유지) | — | v1 폴백 |
| **S8 회귀 스모크** | 배포본 v1(7피처·일별) 파이프라인 그대로 동작 확인 | — | — |

소요 가늠: S1~S4(전처리/데이터 정합) 우선 → S5(모델 개선) → S6~S8(서빙). S1~S3가 호환 회복의 핵심.

---

## 6. 개선 전/후 — 한눈 비교

| 항목 | 현재 (models7) | 개선 후 (C' + v2) |
| --- | --- | --- |
| 서빙 재현성 | ❌ 전처리기 유실 | ✅ `prep_{Model}.joblib`로 transform |
| 데이터 정본 | 7벌(baked) 분산 | ✅ 1벌 raw + 변환기 분리 |
| 라벨 | churn만 | ✅ churn + churn_no_purchase |
| 피처 스키마 | 7(배포) vs 10(5m) 단절 | ✅ v2 10피처 단일 + v1 폴백 |
| 시퀀스 | 14일 vs 17/4주 혼재 | ✅ 주별 고정길이+mask 통일 |
| 모델 성능 | 전처리만 튜닝(Feb 0.7902) | ➕ HP튜닝·보정·임계값으로 추가 향상 |
| 운영 분기 | 코드 하드코딩 | ✅ `model_registry.preprocessing_config` 기반 |
| 배포본 영향 | — | ✅ v1 유지·추가형(무중단) |

---

## 7. 결론 / 권고
1. 사용자 판단대로 **models7는 5m·배포본과 호환되지 않으며, models7 단계부터 재구성이 맞다**. 단 **삭제가 아니라 추가형 재작성**(legacy 보존)으로 진행.
2. 가장 먼저 할 일은 **S1~S3**(스키마 v2 확정 + 두 라벨 정본 + 전처리기 산출). 이것만으로 "재현 가능 + 5m 정합"이 회복된다.
3. 그 위에 **S5 모델 개선**(HP·보정·임계값, [17-6-4 P1])을 얹어 성능을, **S6~S8**로 운영 연결을 마무리.

원하시면 **S1(`feature_schema.yaml v2` + active_days↔ndays 매핑)부터 추가형으로 착수**하고, 기존 산출물은 건드리지 않은 채 회귀 스모크까지 확인해 드리겠습니다.

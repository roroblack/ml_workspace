# preprocessing_project — 전처리 전용 프로젝트 (버전 관리)

`sample_project`(서빙)와 분리된 **전처리기 전용 프로젝트**. 전처리기를 **버전별 폴더**로 관리하며, 각 버전은 ① 재현 소스(`src/`) ② 결과물(`output/`) ③ 구성·컬럼·X/Y 명세(`README.md`)를 모두 보유한다.

## 버전 개요
| 버전 | 폴더 | 정의 | 라벨 | 데이터 백업 | 상태 |
| --- | --- | --- | --- | --- | --- |
| **v1_daily** | `v1_daily/` | 배포본(일별 14일) 7피처 | churn | output/processed/ | 운영 |
| **v2_5m** | `v2_5m/` | 5개월 풀+코호트+7모델 10피처·주별 | churn+no_purchase | **output/processed_5m/(96M)** | 완료 |
| **v3_event_additive** | `v3_event_additive/` | 이벤트 시퀀스(실시간) 추가형 | churn | output/processed_eventseq/ | 완료 |
| **v4_model_prep** | `v4_model_prep/` | **모델별 전처리(하나씩) — 모델별 폴더** | churn | output/<Model>/ | 진행 중 |
| legacy_* / aux_* | 8폴더 | 과거 작업(온라인리테일·REES46 단/다월·라벨·게임·세션·다음카테고리·추천) | 각종 | output/(물리 백업) | 보존 |

> **데이터셋 백업 원칙(유실 복원)**: 각 버전 `output/`에 전처리 **결과 데이터셋을 물리 복사**해 둔다(`backup_datasets.py`로 일괄). 총 ~386M. 마스터 목록 `DATASET_INDEX.md`.
> 호환성 분석 [reports/17-6-5]. v3/v4는 v1/v2 정본을 건드리지 않는 **추가형**.

## 원칙
1. **데이터(raw 정본) ⟂ 전처리(저장된 변환기) ⟂ 모델** 분리. baked parquet은 캐시일 뿐, 변환기(joblib)가 산출물.
2. **재현성**: 각 버전 `src/`의 스크립트만으로 결과물 재생성 가능. 스크립트의 `HERE`는 repo 루트(`team_project_churn`)로 해석되어, 정본 데이터(`sample_project/data/...`)를 그대로 읽고 쓴다.
3. **대용량 산출물**(수십 MB parquet/npz)은 중복 저장하지 않고 **정본 위치(`sample_project/data/processed_5m/`)** 에 두며, 각 버전 `output/`에는 **소형 정의 산출물**(메타 JSON·스케일러·결과 JSON)과 `OUTPUT_INDEX.md`(대용량 목록·경로·크기)를 둔다.

## 데이터 출처
REES46 Cosmetics (Kaggle: `ecommerce-events-history-in-cosmetics-shop`), 2019-10 ~ 2020-02, 약 2,069만 이벤트. 원본 zip: `src/2019-{Oct..Feb}.csv.zip`.

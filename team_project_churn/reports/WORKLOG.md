# 작업 일지 (Work Log)

> 작업 과정·의사결정·실험 기록을 한 곳에 남깁니다. 발표·결과서의 "근거"가 되고,
> 평가자에게 **재현 가능하고 체계적인 프로세스**를 보여줍니다. CRISP-DM 단계로 구성.

---

## 0. 프로젝트 메타
- 주제: 가입 고객 이탈 예측
- 데이터: Bank Customer Churn (10,000행) / (변경 시 기재)
- 기간: 2026-06-22 ~ 06-23
- 팀원/역할: (이름 - 역할)

## 1. 작업 타임라인 (날짜순)
| 날짜/시간 | 담당 | 한 일 | 결과/산출물 | 다음 할 일 |
| --- | --- | --- | --- | --- |
| 06-22 09:00 | 전체 | 비즈니스 이해·데이터 확정·역할 분담 | 본 문서 초안 | EDA |
| 06-22 ... | (이름) | EDA + 전처리 | `processed/*.csv`, 전처리결과서 | 모델링 |
| ... | | | | |

## 2. 의사결정 기록 (Decision Log)
> "왜 그렇게 했는가"를 짧게. 평가/Q&A 대비 핵심.

| # | 결정 | 이유 | 대안(기각 사유) |
| --- | --- | --- | --- |
| 1 | 평가지표를 ROC-AUC·Recall 중심으로 | 이탈 불균형(20%)이라 Accuracy는 오해 소지 | Accuracy(불충분) |
| 2 | 이상치 삭제 안 함 | 비율 낮고 의미 있는 극단값 | IQR 제거(정보 손실) |
| 3 | 불균형은 학습단계 처리(class_weight/SMOTE) | 데이터 누수 방지·간단 | 데이터단 언더샘플(정보손실) |
| ... | | | |

## 3. 실험 기록 (Experiment Log)
> 모델/하이퍼파라미터/전처리 바꿀 때마다 한 줄. 재현성의 핵심.

| 실험ID | 모델 | 주요 설정 | 전처리 | Acc | Recall | F1 | AUC | 비고 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E01 | LogReg | class_weight=balanced | std scale | 0.714 | 0.700 | 0.499 | 0.777 | 베이스라인 |
| E02 | RandomForest | n=300 | std scale | 0.863 | 0.447 | 0.571 | 0.855 | |
| E03 | GradientBoosting | 기본 | std scale | 0.869 | 0.494 | 0.605 | 0.870 | ⭐ AUC 최고 |
| E04 | MLP | 64-32, pos_weight | std scale | 0.783 | 0.764 | 0.590 | 0.862 | Recall 최고 |
| E05 | HistGB(튜닝) | RandomizedSearch | std scale | 0.871 | 0.482 | 0.602 | 0.871 | |
| E06 | HistGB+SMOTE | SMOTE | std scale | 0.840 | 0.663 | 0.628 | 0.867 | Recall↑ |
| E07 | HistGB, thr=0.33 | 임계값 조정 | - | - | 0.639 | 0.644 | - | F1 최고 |

## 4. 막힌 점 / 해결 (Troubleshooting)
| 이슈 | 원인 | 해결 |
| --- | --- | --- |
| (예) 그래프 한글 깨짐 | 폰트 없음 | 라벨 영문화 |
| ... | | |

## 5. 재현 방법 (Reproducibility)
- 환경: `pip install -r requirements.txt`
- 실행 순서: `preprocess.py → train_ml.py → train_dl.py → compare.py → predict.py`
- 시드 고정: SEED=42(전처리/ML), 1234/42(DL)
- 산출물: `models/`, `outputs/`

## 6. 회고 (마무리 시 작성)
- 잘된 점 / 아쉬운 점 / 다음에 시도할 것

# 가입 고객 이탈 예측 (Customer Churn Prediction) — 팀 프로젝트

> **현재 최종 기준**: 최종 모델·실시간 예측·대시보드·추천 기능은 **REES46 데이터만 사용**합니다. Bank Churn(`Churn_Modelling.csv`)은 초기 베이스라인/과거 실험으로만 남기고, 최종 모델에는 사용하지 않습니다. 자세한 규칙은 [`PROJECT_RULES.md`](PROJECT_RULES.md)를 우선합니다.

데이터 분석 / 머신러닝 / 딥러닝 팀 프로젝트. 가입 고객의 **이탈(Churn) 여부를 예측**하고,
성능이 가장 좋은 모델을 선정·배포하는 것을 목표로 합니다.

## 목표 (비즈니스 이해)
이탈 가능성이 높은 고객을 **미리 식별**하여 리텐션(유지) 캠페인 대상으로 활용합니다.
신규 고객 유치보다 기존 고객 유지가 비용 효율적이므로, **이탈자를 놓치지 않는 것(Recall)** 이 핵심입니다.

## 데이터
- Bank Customer Churn (`Churn_Modelling.csv`, 공개 데이터) — 고객 10,000명, 13개 변수
- 타깃 `Churn`: 이탈(1) 20.4% / 유지(0) 79.6% (불균형)
- ※ 강사님 제공 데이터가 있으면 `data/raw/`에 넣고 `src/preprocess.py`의 컬럼명만 맞추면 됩니다.

## 폴더 구조
```
team_project_churn/
├─ data/
│  ├─ raw/Churn_Modelling.csv      # 원본
│  └─ processed/                   # 전처리 결과(train.csv, test.csv)
├─ src/
│  ├─ preprocess.py                # ① 전처리
│  ├─ train_ml.py                  # ② 머신러닝 학습/평가
│  ├─ train_dl.py                  # ③ 딥러닝(MLP) 학습/평가
│  ├─ compare.py                   # ④ 모델 비교·최적 선정
│  └─ predict.py                   # ⑤ 추론(배포) 예시
├─ models/                         # 학습된 모델 + 전처리기
├─ outputs/                        # 지표(json)·그래프(png)
└─ reports/                        # 산출물 문서
   ├─ 01_데이터_전처리_결과서.md
   ├─ 02_인공지능_학습_결과서.md
   └─ 03_발표_및_역할_체크리스트.md
```

## 실행 방법
```powershell
# (권장) 의존성 설치
pip install -r requirements.txt

# 파이프라인 순서대로 실행
python src/preprocess.py     # 전처리 → data/processed, models/preprocessor.joblib
python src/train_ml.py       # ML 3종 학습/평가 → outputs, models/*.joblib
python src/train_dl.py       # MLP 학습/평가 → outputs, models/mlp.pth
python src/compare.py        # 비교표·최적 모델 선정 → outputs/comparison*.md
python src/predict.py        # 새 고객 이탈 확률 예측(배포 예시)
```

## 핵심 결과 (베이스라인)
| 모델 | Accuracy | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- |
| **GradientBoosting** ⭐ | 0.869 | 0.494 | 0.605 | **0.870** |
| MLP (딥러닝) | 0.783 | 0.764 | 0.590 | 0.862 |
| RandomForest | 0.863 | 0.447 | 0.571 | 0.855 |
| LogisticRegression | 0.714 | 0.700 | 0.499 | 0.777 |

→ 종합 분별력 최적은 **GradientBoosting(AUC 0.870)**. 이탈자 포착(Recall) 극대화가 목표라면 MLP 또는 임계값 하향 운영. 자세한 내용은 `reports/` 참고.

## 필수 산출물 매핑
| 요구 산출물 | 위치 |
| --- | --- |
| 인공지능 데이터 전처리 결과서 | `reports/01_데이터_전처리_결과서.md` |
| 인공지능 학습 결과서 | `reports/02_인공지능_학습_결과서.md` |
| 학습된 인공지능 모델 | `models/GradientBoosting.joblib` (+ `preprocessor.joblib`, `mlp.pth`) |

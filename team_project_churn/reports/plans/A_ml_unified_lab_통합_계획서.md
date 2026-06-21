# 계획서 A — ml_unified_lab에 churn 데이터셋/모델 통합

## 목표
이탈(churn) 데이터를 `ml_unified_lab`에 등록하여, **하나의 설정(YAML)으로 ML/DL 모델과 하이퍼파라미터 탐색을 전환**하며 실험하고 비교표·그래프를 자동 생성한다. (학습 결과서 작성 효율화)

## 배경 (현재 구조 확인 완료)
- 데이터셋 로더: `src/datasets/tabular.py` 의 **`load_csv_generic(path, target, name, task, drop=...)`** 가 CSV → 원-핫 인코딩 + 결측 보간 + StandardScaler + 층화분할까지 처리.
- 데이터 등록: `configs/datasets.yaml` (예: `heart`, `concrete`).
- 모델: `configs/models.yaml` 에 **gradient_boosting / random_forest / logistic_regression / mlp / dnn / lstm** 이미 존재.
- 실험: `configs/experiments/*.yaml` 프리셋 → `run_experiment.py` 또는 `streamlit run app.py` 로 실행.

## 작업 항목 (체크리스트)
- [ ] **1. 데이터 배치**: `team_project_churn/data/raw/Churn_Modelling.csv` 를 lab이 접근할 경로(예: `ml_unified_lab/archive/raw_workspace/churn/`)로 복사하거나 절대경로 참조.
- [ ] **2. 로더 추가** (`src/datasets/tabular.py`):
  ```python
  CHURN_CSV = "archive/raw_workspace/churn/Churn_Modelling.csv"
  def load_churn(*, test_size=0.2, random_state=42):
      return load_csv_generic(
          CHURN_CSV, target="Churn", name="churn", task="classification",
          drop=["CustomerId", "Surname"],
          test_size=test_size, random_state=random_state)
  ```
- [ ] **3. 레지스트리 연결**: `src/datasets/__init__.py`(또는 로더 매핑)에서 `"churn" -> load_churn` 등록.
- [ ] **4. datasets.yaml 등록**:
  ```yaml
  churn:
    task: classification
    source: archive/raw_workspace/churn/Churn_Modelling.csv
  ```
- [ ] **5. 실험 프리셋 작성** (`configs/experiments/`):
  - `churn_gbm_optuna.yaml` (주력)
  - `churn_rf_grid.yaml`
  - `churn_mlp_random.yaml`
  예시:
  ```yaml
  experiment_name: churn_gbm_optuna
  dataset: churn
  model: gradient_boosting
  framework: sklearn
  task: classification
  search:
    method: optuna
    n_trials: 30
    params:
      n_estimators: [100, 500]
      learning_rate: [0.02, 0.2]
      max_depth: [2, 5]
  train: {test_size: 0.2, random_state: 42}
  evaluation:
    primary_metric: roc_auc
    plots: [confusion_matrix, roc_curve]
  ```
- [ ] **6. 실행/검증**: `python run_experiment.py --config configs/experiments/churn_gbm_optuna.yaml` 또는 Streamlit 앱에서 선택 실행 → `runs/` 결과 확인.
- [ ] **7. (선택) 평가지표 보강**: 불균형 대응을 위해 `primary_metric: roc_auc`(또는 f1) 사용, 가능하면 class_weight 옵션 노출.

## 산출물
- `ml_unified_lab` 내 churn 데이터셋 + 3개 실험 프리셋
- `runs/` 자동 생성 지표/그래프 (학습 결과서에 인용)

## 예상 시간 / 난도
- 약 1~2시간 / **중하**. (로더 패턴이 이미 있어 추가가 간단)

## 리스크 / 주의
- `load_csv_generic`은 자체적으로 스케일링/원핫을 하므로, 본 프로젝트 `preprocess.py`와 **이중 전처리 주의**(lab 경로는 원본 CSV를 넣을 것).
- 레지스트리 매핑 위치가 다를 수 있어 `src/datasets/__init__.py` 확인 필요.
- class_weight/불균형 옵션이 프레임워크에 노출 안 돼 있으면 코드 소폭 수정 필요.

## 우선순위
**중** — 결과서 자동화·복습용으로 유용하나, 본 프로젝트는 `team_project_churn` 단독 스크립트로도 완결됨. 시간 여유 시 진행.

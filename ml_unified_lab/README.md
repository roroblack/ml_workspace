# ML Unified Lab

`ml_unified_lab`는 지금까지 `ml_workspace`에서 진행한 머신러닝/딥러닝 실습을 한 곳에 모으고, 하나의 설정 파일로 데이터셋, 모델, 하이퍼파라미터, 탐색 방법, 평가 그래프를 전환해 실험할 수 있게 만든 **설정 기반 통합 실험 프레임워크**입니다.

## 빠른 시작

### 가상환경(.venv) 권장 — Python 3.12

```powershell
# 프로젝트 폴더에서 (Windows PowerShell)
py -3.12 -m venv .venv            # 또는: <python3.12 경로> -m venv .venv
.\.venv\Scripts\Activate.ps1      # 활성화
python -m pip install -U pip
pip install -r requirements.txt   # torch 포함 전체 설치
```

> ⚠️ torch는 Python 3.13/3.14용 휠이 아직 없을 수 있어 **3.12 권장**.
> 활성화 없이 바로 쓰려면 `.\.venv\Scripts\python.exe run_experiment.py ...` 처럼 호출.

활성화하지 않을 경우(시스템 파이썬 사용):

```bash
pip install -r requirements.txt
```

### GUI (권장)

```bash
streamlit run app.py
```

브라우저(http://localhost:8501)에서 데이터셋·모델·탐색을 드롭다운으로 골라 실행하고,
지표/그래프/탐색 trial 표와 과거 실행 결과를 바로 확인할 수 있습니다. 4개 탭으로 구성:

- **▶ 프리셋 실험** — `configs/experiments/*.yaml` 을 골라 그대로 실행
- **🧩 직접 구성** — 데이터셋·모델·탐색을 조합하고, 모델별 하이퍼파라미터를 위젯으로 설정해 실행
- **📝 코드 보기/편집** — `src/`·`configs/` 소스를 코드 블럭으로 보고 수정·저장(.bak 백업)하며 복습
- **📁 지난 결과** — `runs/` 의 과거 실행 지표/그래프 열람

> 설정 가능한 모든 YAML 옵션/하이퍼파라미터는 [reports/yaml_options_guide.md](reports/yaml_options_guide.md) 참고.
> Colab에서 바로 띄우는 방법은 [reports/colab_setup_guide.md](reports/colab_setup_guide.md) 참고.
> 프로젝트 규칙은 [RULES.md](RULES.md), 작업 이력은 [HISTORY.md](HISTORY.md).

### CLI

```bash
python run_experiment.py --list                                          # 실험 목록
python run_experiment.py -c configs/experiments/iris_logistic_grid.yaml  # 단일 실험
python run_experiment.py --all                                           # 전체 실행
```

> Windows PowerShell에서 한글 로그가 깨지면 `$env:PYTHONUTF8=1` 설정 후 실행하세요.

### 실행 검증된 실험 (모두 `--all`로 통과)

| 실험 | 구성 | 결과 |
| --- | --- | --- |
| `iris_logistic_grid` | Iris · LogReg · Grid | accuracy 1.00 |
| `breast_cancer_svm_random` | Breast Cancer · SVM · Random | f1 0.965 |
| `concrete_gbr_grid` | Concrete · GradientBoosting · Grid | RMSE 4.95, R² 0.914 |
| `housing_dnn_random` | Housing · DNN(torch) · Random | RMSE 3.06, R² 0.872 |
| `fashion_mnist_cnn_optuna` | FashionMNIST · CNN(torch) · Optuna | accuracy 0.86 |
| `cifar10_cnn_optuna` | CIFAR10 · CNN(torch) · Optuna | accuracy 0.54 |

> 딥러닝 실험은 CPU에서 빠르게 검증되도록 `max_train_samples`/`epochs`를 줄여 두었습니다.
> GPU나 전체 학습 시 해당 값을 키우면(또는 제거하면) 정확도가 올라갑니다.

## 지원 항목

| 구분 | 항목 |
| --- | --- |
| **데이터셋 (tabular)** | iris, wine, breast_cancer, titanic, subway, concrete, heart, housing |
| **데이터셋 (vision)** | mnist, fashion_mnist, cifar10 *(torchvision 필요)* |
| **데이터셋 (연관규칙)** | groceries, online_retail |
| **모델 (sklearn)** | logistic/linear regression, knn, svm(분류·회귀), naive_bayes, decision_tree, random_forest, gradient_boosting |
| **모델 (torch)** | mlp, dnn, cnn_basic, lstm(분류·회귀·시계열) *(torch 필요)* |
| **모델 (special)** | apriori (연관규칙) |
| **탐색** | none(단일), grid, random, bayesian(Optuna, 없으면 random 폴백), apriori |

## 실험 설정 작성법

`configs/experiments/` 아래에 YAML 한 개를 추가하면 새 실험이 됩니다.

```yaml
experiment_name: iris_logistic_grid
dataset: iris                 # configs/datasets.yaml 의 키
model: logistic_regression    # configs/models.yaml 의 키
framework: sklearn            # sklearn | torch
task: classification          # classification | regression | image_classification
search:
  method: grid                # none | grid | random | bayesian
  params:                     # 생략 시 모델별 기본 탐색 그리드 사용
    C: [0.1, 1.0, 10.0]
    solver: [lbfgs]
train:
  test_size: 0.2
  random_state: 42
evaluation:
  primary_metric: accuracy
  plots: [confusion_matrix]   # confusion_matrix | residual_plot | prediction_scatter
                              # | loss_curve | accuracy_curve
```

## 결과물

각 실행은 `runs/{timestamp}_{experiment_name}/` 에 저장됩니다.

- `config.yaml` — 실행 설정 스냅샷
- `metrics.json` — best 파라미터 + 평가 지표 + 산출물 경로
- `model.joblib` / `model.pt` — 학습된 모델
- `search_trials.csv` — 탐색 trial별 점수
- `plots/*.png` — confusion matrix, residual/scatter, loss/accuracy curve, search history

## 현재 정리 상태

- 원본 실습 파일 모음: `archive/raw_workspace/`
- 전체 파일 인벤토리: `reports/workspace_inventory.csv`
- 파일 타입 요약: `reports/file_type_summary.csv`
- 주제별 분류: `reports/topic_inventory.csv`, `reports/topic_summary.csv`
- 완전 중복 파일: `reports/duplicate_files.csv`
- 같은 이름 중복 파일: `reports/name_duplicates.csv`
- 구현 계획서: `PROJECT_PLAN.md`

## 통합 대상

- `from_colab`
- `from_colab_deep_learning`
- `ml_data_preprocessing`

`.venv`, `.git`, 캐시, IDE 설정 폴더는 통합 대상에서 제외했습니다.

## 구현 현황 (PROJECT_PLAN.md 기준)

- [x] **Phase 2. 데이터셋 레지스트리** — `src/datasets/` (tabular 8종 동작, vision 3종 torch 연동)
- [x] **Phase 3. 모델 레지스트리** — `src/models/` (sklearn 8종 동작, torch 4종 연동)
- [x] **Phase 4. 실험 실행기** — `run_experiment.py` + `src/training/`
- [x] **Phase 5. 하이퍼파라미터 탐색** — `src/search/` (grid / random / bayesian)
- [x] **Phase 6. 결과 확인** — `src/reporting/` (지표 + 플롯 자동 저장)

`src/` 패키지 구성:

```text
src/
  config.py        # 설정 로딩/병합
  datasets/        # 데이터 로더 (tabular + vision)
  models/          # sklearn / torch 모델 팩토리
  search/          # grid / random / bayesian 탐색
  training/        # 실험 오케스트레이터 + 학습 루프
  reporting/       # 지표 계산 + 플롯 저장
```


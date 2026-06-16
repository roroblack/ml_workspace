# 실험 설정(YAML) 옵션 설명서

> `configs/experiments/*.yaml` 한 개가 "실험 1개"를 정의한다. 이 문서는 넣을 수 있는
> 모든 키와 값, 그리고 모델별로 조정 가능한 하이퍼파라미터를 정리한다.
> (단일 출처: 하이퍼파라미터 사양은 [`src/param_specs.py`](../src/param_specs.py)와 일치한다.)

작성: 2026-06-14 · 규칙: [RULES.md](../RULES.md)

---

## 1. 기본 골격

```yaml
experiment_name: iris_logistic_grid   # 결과 폴더 이름에 사용
dataset: iris                         # configs/datasets.yaml 의 키
model: logistic_regression            # configs/models.yaml 의 키
framework: sklearn                    # sklearn | torch | special  (생략 시 모델에서 추론)
task: classification                  # 생략 시 dataset 에서 추론
search:
  method: grid                        # none | grid | random | bayesian | apriori
  trials: 20                          # random/bayesian 일 때 시도 횟수
  params:                             # 탐색할 하이퍼파라미터 (아래 §4)
    C: [0.1, 1.0, 10.0]
    solver: [lbfgs]
train:                                # 학습/전처리 설정 (아래 §5)
  test_size: 0.2
  random_state: 42
evaluation:
  primary_metric: accuracy
  plots: [confusion_matrix]           # 아래 §6
```

## 2. 데이터셋 (`dataset`)

| 키 | task | 종류 | 비고 |
| --- | --- | --- | --- |
| iris, wine, breast_cancer | classification | tabular | sklearn/CSV |
| titanic, heart | classification | tabular | archive CSV |
| subway, concrete, housing | regression | tabular | archive CSV |
| mnist, fashion_mnist, cifar10 | image_classification | vision | torchvision (torch 필요) |
| groceries, online_retail | association_rules | transactions | 장바구니(연관규칙) |

## 3. 모델 (`model`) — task 호환표

| 모델 | framework | 지원 task |
| --- | --- | --- |
| logistic_regression | sklearn | classification |
| linear_regression | sklearn | regression |
| knn | sklearn | classification, regression |
| svm | sklearn | classification, regression |
| naive_bayes | sklearn | classification |
| decision_tree | sklearn | classification, regression |
| random_forest | sklearn | classification, regression |
| gradient_boosting | sklearn | classification, regression |
| mlp, dnn | torch | classification, regression |
| cnn_basic | torch | image_classification |
| lstm | torch | classification, regression, time_series |
| apriori | special | association_rules |

> 모델과 데이터셋의 task가 맞아야 한다. (예: `linear_regression`은 regression 데이터셋에만)

## 4. 탐색 (`search`)

| method | 설명 | 필요 키 |
| --- | --- | --- |
| `none` | 단일 학습(탐색 없음). `params`의 첫 값으로 1회 학습 | - |
| `grid` | 모든 조합 전수 탐색(GridSearchCV) | `params` |
| `random` | 무작위 샘플링 | `params`, `trials` |
| `bayesian` | Optuna(TPE). 미설치 시 random 폴백(로그 표시) | `params`, `trials` |
| `apriori` | 연관규칙 전용 | `params`(min_support 등) |

### 4-1. 모델별 `search.params` (실습 과제에서 다룬 항목 포함)

값은 **리스트**로 적으면 탐색 후보가 된다. (단일 실행이면 첫 값 사용)

**sklearn**

| 모델 | 파라미터 | 예시 값 | 의미 |
| --- | --- | --- | --- |
| logistic_regression | `C` | [0.1, 1.0, 10.0] | 규제 역수 |
| | `solver` | [lbfgs, liblinear, saga] | 최적화기 |
| knn | `n_neighbors` | [3,5,7,9] | 이웃 수 K |
| | `weights` | [uniform, distance] | 가중 방식 |
| | `p` | [1,2] | 1=맨해튼,2=유클리드 |
| svm | `C` | [0.1,1.0,10.0] | 마진 trade-off |
| | `kernel` | [rbf, linear, poly, sigmoid] | 커널 |
| | `gamma` | [scale, auto] | 커널 계수 |
| naive_bayes | `var_smoothing` | [1e-9,1e-8,1e-7] | 분산 안정화 |
| decision_tree | `max_depth` | [3,5,10] | 최대 깊이 |
| | `min_samples_split` | [2,5,10] | 분할 최소 샘플 |
| random_forest | `n_estimators` | [100,200] | 트리 수 |
| | `max_depth` | [5,10] | 최대 깊이 |
| gradient_boosting | `n_estimators` | [100,200] | 단계 수 |
| | `learning_rate` | [0.05,0.1] | 학습률 |
| | `max_depth` | [3,5] | 약학습기 깊이 |

**torch** (딥러닝 — 실습 과제의 핵심 조정 대상)

| 모델 | 파라미터 | 예시 값 | 의미 |
| --- | --- | --- | --- |
| mlp/dnn | `lr` | [0.001,0.01] | 학습률 |
| | `hidden_units` | [64,128,256] | 은닉 노드 수 |
| | `depth` (dnn) | [2,3] | 은닉층 수 |
| | `dropout` | [0.0,0.2] | 드롭아웃 |
| | `optimizer` | [adam,adamw,sgd,rmsprop] | 옵티마이저 |
| | `batch_size` | [32,64] | 배치 크기 |
| cnn_basic | `lr` | [0.0005,0.001,0.003] | 학습률 |
| | `optimizer` | [adam,adamw] | 옵티마이저 |
| | `batch_size` | [64,128] | 배치 크기 |
| lstm | `lr`,`hidden_units`,`num_layers`,`optimizer`,`batch_size` | - | 시퀀스 모델 |

**apriori** (연관규칙)

| 파라미터 | 예시 값 | 의미 |
| --- | --- | --- |
| `min_support` | 0.005, 0.01, 0.02, 0.05 | 최소 지지도(0605 실습 비교 대상) |
| `min_confidence` | 0.2, 0.3, 0.5 | 최소 신뢰도 |
| `max_len` | 2, 3 | 항목집합 최대 크기 |

## 5. 학습 설정 (`train`)

| 키 | 적용 | 기본 | 의미 |
| --- | --- | --- | --- |
| `test_size` | sklearn/torch tabular | 0.2 | 검증셋 비율 |
| `random_state` | 전체 | 42 | 랜덤 시드(SEED). 제거 실험 시 매 실행 결과 변동 |
| `epochs` | torch | 5 | 에폭 수 |
| `batch_size` | torch | 64 | 배치 크기(탐색 미사용 시) |
| `lr` | torch | 0.001 | 학습률(탐색 미사용 시) |
| `optimizer` | torch | adam | 옵티마이저(탐색 미사용 시) |
| `max_train_samples` | torch vision | 8000 | CPU용 학습 서브샘플(전체=제거 또는 0) |
| `max_test_samples` | torch vision | 2000 | 평가 서브샘플(전체=제거 또는 0) |
| `min_support`/`min_confidence`/`max_len` | apriori | 0.01/0.3/2 | 연관규칙 임계값 |

## 6. 평가/플롯 (`evaluation.plots`)

| 플롯 | 적용 task |
| --- | --- |
| `confusion_matrix` | classification, image_classification |
| `residual_plot`, `prediction_scatter` | regression |
| `loss_curve`, `accuracy_curve` | torch (학습 곡선) |
| (연관규칙은 `rules_scatter` 자동 생성) | association_rules |

## 7. GUI에서 설정하기

위 모든 옵션은 GUI **🧩 직접 구성** 탭에서 드롭다운/슬라이더로 설정할 수 있다.
GUI가 만든 설정은 `runs/_gui_configs/`에 YAML로 저장되므로, 거기서 실제 생성된 설정을
확인하고 `configs/experiments/`로 복사해 프리셋으로 재사용할 수 있다.

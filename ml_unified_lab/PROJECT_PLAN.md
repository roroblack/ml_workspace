# ML Unified Lab 구현 계획서

## 1. 목표

지금까지 진행한 모든 ML/DL 실습을 정리하고, 하나의 프로젝트에서 설정만 바꿔 다양한 데이터셋, 모델, 하이퍼파라미터 탐색, 평가 방식을 실행할 수 있게 만든다.

## 2. 1차 정리 결과

- 통합 파일 수: 171개
- 주요 파일 타입: `.ipynb` 93개, `.py` 21개, `.csv` 15개, `.txt` 12개, `.pdf` 10개
- 완전 중복 파일: 8개 행
- 같은 파일명 중복 파일: 20개 행
- 주요 주제: 전처리, 회귀, 분류, KNN, SVM, Naive Bayes, Decision Tree, Ensemble, PCA, Clustering, Association Rules, ANN/DNN, CNN, LSTM, GAN, Autograd

## 3. 프로젝트 구조

```text
ml_unified_lab/
  archive/raw_workspace/    # 기존 실습/데이터/결과 파일 보존본
  configs/                  # 데이터셋, 모델, 탐색, 실험 설정
  reports/                  # 전체 작업 정리 및 중복 분석 리포트
  runs/                     # 향후 실험 실행 결과 저장
  src/                      # 통합 실행 코드
```

## 4. 구현 단계

### Phase 1. 인벤토리 보강

- notebook 코드 셀 분석
- import, model, dataset, metric, optimizer, lr, batch size, epoch 추출
- 원본 실습과 통합 실험 설정 간 매핑표 작성

### Phase 2. 데이터셋 레지스트리

- tabular: Iris, Wine, Breast Cancer, Titanic, Subway, Concrete, Heart, Housing
- vision: MNIST, FashionMNIST, CIFAR10, Fruit image
- text: SMS Spam
- time series: Airline
- unsupervised/association: Online Retail, Groceries

### Phase 3. 모델 레지스트리

- sklearn: Linear/Logistic Regression, KNN, SVM, Naive Bayes, Decision Tree, Random Forest, AdaBoost, Gradient Boosting
- torch: MLP, DNN, CNN, LSTM
- special task: PCA, KMeans, Association Rules, GAN

### Phase 4. 실험 실행기

- `run_experiment.py --config configs/experiments/*.yaml`
- 설정 기반 데이터 로딩, 모델 생성, 학습, 평가, 결과 저장
- 실행 결과는 `runs/{timestamp}_{experiment_name}/`에 저장

### Phase 5. 하이퍼파라미터 탐색

- Grid Search
- Random Search
- Bayesian Search / Optuna
- trial별 metric, best config, 그래프 자동 저장

### Phase 6. 결과 확인

- classification: accuracy, precision, recall, f1, confusion matrix
- regression: MAE, RMSE, R2, residual plot
- deep learning: train/valid loss curve, accuracy curve
- search: trial history, best parameter chart

## 5. MVP 우선순위

1. Iris + Logistic Regression + Grid Search
2. Breast Cancer + KNN/SVM + Random Search
3. Housing/Concrete + DNN + Random/Bayesian Search
4. FashionMNIST/MNIST + CNN + Optuna
5. CIFAR10 + CNN + lr/batch size/optimizer switch


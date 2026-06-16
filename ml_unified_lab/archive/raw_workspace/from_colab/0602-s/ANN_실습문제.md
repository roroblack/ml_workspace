# ANN(Artificial Neural Network) 실습문제

# 심화 실습문제

## 실습문제 1. 상관계수(Correlation Coefficient) 분석

1. 상관계수(Correlation Coefficient)의 의미를 설명하시오.

두 변수가 선형적으로 밀접한가를 확인
때문에 곡선과 같은 경우 유사도가 0에 가깝게 될 수 있음
해당 특성을 고려하면 회귀 모델에 적합

2. 상관계수 값의 범위를 작성하시오.

0 에서 1 사이의 값이면 좋음
-1 에 가까울 수록 나쁨
범위는 -1 < 0 < 1

3. 상관계수가 0.95인 경우 모델 성능을 평가하시오.

기울기와 수치를 고려하지 않음

- 결국 값이 오를지 내릴지는 맞춰도 
정확히 얼마로 될 지는 알 수 없음

- Data Leakage 고려 한 건지 알 수 없음
    학습할 데이터 변수가 적은 상황에서 0.95 면 정답을 그대로 배낀 것이 
    아닌가 고려가 필요
    => 잔차분석 (Residual Plot) : 실제값 - 예측값 을 한 값을 그래프로 그려봄
                                    이 때 오차가 특정 구간에서만 크거나
                                    패턴을 보인다면 모델에 문제가 있음

- 과적합 Overfitting 인지 알 수 없음
    학습량만 늘려서 과적합이 생겨서 0.95 인 지 또한 알 수 없음
    => 학습과 검증 셋 사이의 비교가 필요

이를 보정하기 위해서는 RMSE, MSE, MAE 와 같은 수치를 학습, 검증셋에 따라 비교하는
걸 함께 시행해 관측할 필요가 있음.


4. 상관계수와 MSE의 차이점을 설명하시오.

상관계수는 -1 에서 1 사이의 값이 나오게 스케일링 한 결과
    => 값이 올라갈 경우 예측도 같이 올라가는지 확인
MSE 는 정답과 예측값의 차를 단순히 제곱한 형태의 평균 
    => test 값에서 예측이 얼마나 벗어난지 확인

즉, 둘은 사용하는 목적이 다름. 전자는 진행 방향성이 맞는지 보고
    후자는 예측이 얼마나 다른지를 봄

---

## 실습문제 2. 과적합(Overfitting) 분석

1. 과적합(Overfitting)의 의미를 설명하시오.

노이즈에 대한 편향이 발생함을 의미
학습이 과도하게 되서 테스트의 결과는 잘 맞추지 못하면 과적합이 발생한 것

2. 은닉노드를 50개 이상으로 증가시켰을 때 발생할 수 있는 문제를 설명하시오.

    1) 모델의 복잡도 증가 -> 과적합 발생
    2) 연산속도 저하
    3) 기울기 소실 -> 앞쪽 노드 가중치가 학습되지 않는 문제
    4) 메모리 사용량 증가
    5) 데이터 요구량 증가


3. 과적합을 방지하는 방법을 3가지 이상 작성하시오.

    데이터 중심 기법
    - 데이터 증강
    - 데이터 추가 도입


    모델 구조 및 학습 제어
    - 은닉 레이어 & 노드 수 조절
    - Dropout 도입 및 수치 올리기
    - 배치 정규화 Batch Normalization
    - Early Stopping 도입
    - Epoch 수치 조절

    가중치 규제
    - L1 규제
    - L2 규제 (Ridge, Weight Decay)
    - 옵티마이저 적용
    -> Adam 적용 & - lr, weight_decay 옵션 조정
        스텝과 가속도를 조절

    - 앙상블 도입


---

## 실습문제 3. 하이퍼파라미터 튜닝

다음 항목을 변경하여 실험을 수행하시오.

* Epoch
* Learning Rate
* Hidden Node 수
* Hidden Layer 수
* Activation Function

| 순위 | Epoch | LR | Node | Layer | Activation | MSE | Corr |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 5000 | 0.01 | 64 | 2 | tanh | 0.003237 | 0.9601 |
| 2 | 2000 | 0.005 | 64 | 3 | tanh | 0.003655 | 0.9546 |
| 3 | 5000 | 0.005 | 64 | 2 | tanh | 0.003774 | 0.9534 |
| 4 | 5000 | 0.005 | 32 | 2 | tanh | 0.003845 | 0.9523 |
| 5 | 5000 | 0.001 | 64 | 3 | tanh | 0.003985 | 0.9505 |
| 6 | 2000 | 0.01 | 64 | 2 | tanh | 0.003990 | 0.9503 |
| 7 | 5000 | 0.005 | 64 | 3 | tanh | 0.004045 | 0.9523 |
| 8 | 2000 | 0.01 | 64 | 3 | tanh | 0.004051 | 0.9499 |
| 9 | 5000 | 0.001 | 64 | 2 | relu | 0.004085 | 0.9497 |
| 10 | 5000 | 0.005 | 64 | 1 | relu | 0.004097 | 0.9490 |
| 11 | 5000 | 0.01 | 32 | 2 | tanh | 0.004213 | 0.9486 |
| 12 | 5000 | 0.01 | 64 | 3 | softplus | 0.004229 | 0.9479 |
| 13 | 5000 | 0.001 | 64 | 1 | relu | 0.004274 | 0.9472 |
| 14 | 2000 | 0.001 | 64 | 2 | relu | 0.004339 | 0.9462 |
| 15 | 5000 | 0.01 | 64 | 1 | relu | 0.004390 | 0.9461 |

### 작성 내용

| 실험번호 | Epoch | Learning Rate | Hidden Node | Activation | MSE |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 실험1 | 5000 | 0.01 | 64 | tanh | 0.003237 |
| 실험2 | 2000 | 0.005 | 64 | tanh | 0.003655 |
| 실험3 | 5000 | 0.005 | 64 | tanh | 0.003774 |

실험 결과를 비교하고 가장 성능이 좋은 모델을 선택하시오.

---

## 실습문제 4. 활성화 함수 성능 비교

다음 활성화 함수를 적용하여 성능을 비교하시오.

* Sigmoid
* Tanh
* ReLU
* Softplus

### 작성 내용

| Activation Function | MSE | Correlation |

| Sigmoid                  |    0.005413    |   0.9322  |
| Tanh                     |    0.004814    |   0.9397  |
| ReLU                     |    0.005048    |   0.9370  | 
| Softplus                 |    0.004755    |   0.9404  |

가장 좋은 성능을 보인 활성화 함수를 선택하고 이유를 설명하시오.

Softplus - 가장 균일한 y=1x 에 가까운 기울기 그래프를 지님
            상관계수 또한 가장 높고 MSE 도 가장 낮음

---

## 실습문제 5. 최종 모델 평가 보고서 작성

다음 항목을 포함하여 최종 보고서를 작성하시오.

1. 데이터셋 설명

- **데이터셋**: 콘크리트 압축강도(Concrete Compressive Strength) 데이터 (`concrete_stg.csv`)
- **목적**: 콘크리트 배합 성분과 양생 일수로 **압축강도(strength)** 를 예측하는 회귀 문제
- **샘플 수**: 총 1,030개 (학습 780개 / 테스트 250개)
- **결측치**: 없음, 모든 변수 수치형(연속형)

**입력 변수 (8개)** — 단위는 1m³ 배합 기준 kg, age는 일(day)

| 변수 | 의미 |
|------|------|
| cement | 시멘트 양 |
| slag | 고로슬래그(고로 부산물) 양 |
| ash | 플라이애시(석탄재) 양 |
| water | 물 양 |
| superplastic | 고성능 감수제(유동화제) 양 |
| coarseagg | 굵은 골재(자갈) 양 |
| fineagg | 잔골재(모래) 양 |
| age | 양생(굳히기) 일수 |

**목표 변수 (1개)**

| 변수 | 의미 |
|------|------|
| strength | 콘크리트 압축강도 (MPa) — 평균 35.8, 표준편차 16.7, 범위 2.33 ~ 82.6 |

- **분할 방식**: 정규화 후 앞 780행을 학습, 뒤 250행을 테스트로 사용


2. 정규화 방법 설명

방법 1 : Min-Max 정규화
            변환을 원하는 값에서 최소값을 범위의 최소로 빼고 최대에서 최소를 뺀 값으로 나눔
방법 2 : 표준화 (Standardization / Z-Score Scaling)
            변환을 원하는 값에서 평균을 뺀 값을 표준편차로 나눔

3. Model1 구조 설명

Model1 - 은닉노드 1개, sigmoid
    입력 8 -> Linear(8,1) -> Sigmoid -> Linear(1,1) -> 출력 1
    가장 단순한 구조 -> 표현력 부족으로 과소적합
    => corr 0.7995, MSE 0.0149 로 최하위

4. Model2 구조 설명

Model2 - 은닉노드 5개, sigmoid
    입력 8 -> Linear(8,5) -> Sigmoid -> Linear(5,1) -> 출력 1
    노드를 5개로 늘려 비선형 표현력 상승
    => corr 0.9286, MSE 0.0057 로 크게 개선

5. Model3 구조 설명

Model3 - 은닉층 2개 [5,5], softplus
    입력 8 -> Linear(8,5) -> Softplus -> Linear(5,5) -> Softplus -> Linear(5,1) -> 출력 1
    층을 깊게 + softplus로 부드러운 비선형 적용
    => corr 0.9256, MSE 0.0059 로 Model2와 비슷한 수준

6. 손실 함수 설명

이상치에 대해 강한 패널티를 주고 싶을 때 사용.
MSE, RMSE, MAE, 크로스 엔트로피 등이 있으며 선형모델의 릿지 모델 등에서 사용 됨
실습의 코드에선 결과와 예측의 오차를 보고 싶어서 사용

7. Optimizer 설명

최적화 알고리즘. 경사하강법을 사용. 예측치에 미분을 적용해 기울기를 구하고
이를 통해 가장 기울기가 가파른 방향으로 이동하며 최적화를 진행하는 방식
이 때 경사가 가파르면 보폭(step)을 줄이고 반대의 경우 늘리고
학습률이 적은 구간에 들어가면 가속도를 적용하는 등으로 알고리즘을 개선함
=> Adam
그리고 Adam LR 패널티에 들어가는 문제를 해결한 게 AdamW
이는 과적합 방지에 도움이 되며 실제 코드에 사용

8. 실험 결과 비교

Model1 < Model2 ≈ Model3
    노드 1개(Model1)는 과소적합으로 최하위
    Model2(노드5) 와 Model3(2층 softplus)은 비슷한 성능
    => 그리드서치로 더 넓게 탐색하니 tanh + 노드64 + 2층 조합이 최고
    MSE 0.0149 -> 0.0057 -> 0.0032 순으로 개선됨

9. 최종 선택 모델

그리드서치 최적 조합 선택
    Epoch 5000 / LR 0.01 / Hidden Node 64 / Hidden Layer 2 / tanh
    => MSE 0.003237, 상관계수 0.9601 로 전 216개 조합 중 최고 성능

10. 모델 성능 향상 방안

모델 성능 향상을 위한 일부 방안을 실제 시행해봄.
해당 사항은 아래와 같음

추가 성능향상 진행한 사항 :

학습 제어
    얼리스탑 - L2, 스케쥴링 적용
    Best-weight 복원
    검증셋 모니터링 (X_val/y_val)

옵티마이저, 가중치 규제
    weight_decay 적용
    adamW 적용

활성화 함수 확장
    ReLU, Tanh 추가

탐색 & 실험 방법론
    그리드서치 진행해 최적해 찾기
    결과 자료구조에 기록 - 정렬 출력 - 최저 MSE 조합 자동 선택

결과적으로 기존 Model2 와 비교해 최적해에서 MSE 약 43% 감소 확인. (상관계수는 0.929 -> 0.960)

---

### 추가 도전 과제 : 
하단에 코드셀 추가해서 코드로 작성하고, 코드를 이곳에 복사해서 제출합니다. ==============

#### 도전 과제 1






은닉층을 다음과 같이 변경하여 성능을 비교하시오.

```
hidden_layers=[10,10]
hidden_layers=[20,20]
hidden_layers=[50,50]
```

import itertools, time

# ---------- 조절한 그리드 (실제 실행한 216개 조합) ----------
# 참고: 처음 시도한 7680개(EPOCHS4 x LR5 x WD6 x NODES4 x LAYERS4 x ACT4)는
#       학습 시간이 너무 길어, 핵심 차원을 커버하는 216개로 축소해 실제 실행했습니다.
# EPOCHS_GRID = [2000, 5000]
EPOCHS_GRID = [5000]
# LR_GRID     = [0.001, 0.005, 0.01]
LR_GRID     = [0.01]
# NODES_GRID  = [5, 32, 64]
NODES_GRID  = [10, 20, 50]
# LAYERS_GRID = [1, 2, 3]
LAYERS_GRID = [2]
# ACT_GRID    = ['sigmoid', 'softplus', 'relu', 'tanh']
ACT_GRID    = ['sigmoid',]
# -----------------------------------------------------------

def train_quiet(model, X, y, epochs, lr, wd=0.0):       # 출력 없는 학습
    model = model.to(device)
    criterion, optimizer = nn.MSELoss(), optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        model.train(); optimizer.zero_grad()
        loss = criterion(model(X), y); loss.backward(); optimizer.step()
    return model

# (1) 결과를 담을 자료구조
results = []

# (2) 학습 루프: 매 조합 결과를 메모리에 기록
combos = list(itertools.product(EPOCHS_GRID, LR_GRID, NODES_GRID, LAYERS_GRID, ACT_GRID))
print(f'총 조합: {len(combos)}개')

t0 = time.time()
for i, (ep, lr, nodes, n_layers, act) in enumerate(combos, 1):
    torch.manual_seed(SEED); np.random.seed(SEED)        # 공정 비교용 동일 초기화
    model = ConcreteANN(input_dim=input_dim, hidden_layers=[nodes]*n_layers, activation=act)
    model = train_quiet(model, X_train, y_train, ep, lr)
    pred_np, corr, mse = evaluate_model(model, X_test, y_test)
    results.append({'ep': ep, 'lr': lr, 'nodes': nodes, 'layers': n_layers,
                    'act': act, 'corr': corr, 'mse': mse})
    if i % 30 == 0:
        print(f'  {i}/{len(combos)} ({time.time()-t0:.0f}s)')

# (3) mse 낮은 순 정렬 후 상위 20개 출력
results.sort(key=lambda r: r['mse'])
print(f"{'ep':>6}{'lr':>8}{'node':>6}{'layer':>6}{'act':>10}{'corr':>8}{'mse':>10}")

print('-' * 54)
for r in results[:20]:
    print(f"{r['ep']:>6}{r['lr']:>8}{r['nodes']:>6}{r['layers']:>6}{r['act']:>10}{r['corr']:>8.4f}{r['mse']:>10.6f}")
print('최고 조합:', results[0])

# 전체 결과 저장
import pandas as pd
pd.DataFrame(results).to_csv('grid_results.csv', index=False)



총 조합: 3개
    ep      lr  node layer       act    corr       mse
------------------------------------------------------
  5000    0.01    20     2   sigmoid  0.9417  0.004691
  5000    0.01    10     2   sigmoid  0.9390  0.004892
  5000    0.01    50     2   sigmoid  0.9354  0.005164
최고 조합: {'ep': 5000, 'lr': 0.01, 'nodes': 20, 'layers': 2, 'act': 'sigmoid', 'corr': np.float64(0.9416827842034351), 'mse': 0.004690581001341343}







---

#### 도전 과제 2

Dropout Layer를 추가하여 과적합을 감소시키시오.




class ConcreteANN(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation='sigmoid', drpt=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'softplus':
                layers.append(nn.Softplus())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError('지원하지 않는 activation')

            layers.append(nn.Dropout(drpt))   # ← 활성화 뒤에 추가, layers에 append
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))  # 출력층 (뒤에 드롭아웃 없음)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)   # 드롭아웃은 이미 network 안에 포함됨


결과 :


### dropout 적용 전

총 조합: 3개
    ep      lr  node layer       act    corr       mse
------------------------------------------------------
  5000    0.01    20     2   sigmoid  0.9417  0.004691
  5000    0.01    10     2   sigmoid  0.9390  0.004892
  5000    0.01    50     2   sigmoid  0.9354  0.005164
최고 조합: {'ep': 5000, 'lr': 0.01, 'nodes': 20, 'layers': 2, 'act': 'sigmoid', 'corr': np.float64(0.9416827842034351), 'mse': 0.004690581001341343}

### dropout 적용 후

총 조합: 3개
    ep      lr  node layer       act    corr       mse
------------------------------------------------------
  5000    0.01    50     2   sigmoid  0.9379  0.005016
  5000    0.01    20     2   sigmoid  0.9246  0.005973
  5000    0.01    10     2   sigmoid  0.9003  0.007860
최고 조합: {'ep': 5000, 'lr': 0.01, 'nodes': 50, 'layers': 2, 'act': 'sigmoid', 'corr': np.float64(0.9378804474293136), 'mse': 0.0050158146768808365}


결과에선 과적합이 오히려 없어서 적용 후 값이 안좋아 짐






---

#### 도전 과제 3

Batch Normalization을 추가하여 성능 변화를 확인하시오.


class ConcreteANN(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation='sigmoid', drpt=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            layers.append(nn.BatchNorm1d(hidden_dim)) # batch normalization

            if activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'softplus':
                layers.append(nn.Softplus())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError('지원하지 않는 activation')

            layers.append(nn.Dropout(drpt))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)   # 드롭아웃은 이미 network 안에 포함됨



총 조합: 3개
    ep      lr  node layer       act    corr       mse
------------------------------------------------------
  5000    0.01    50     2   sigmoid  0.9454  0.004910
  5000    0.01    20     2   sigmoid  0.9351  0.005259
  5000    0.01    10     2   sigmoid  0.9259  0.005905
최고 조합: {'ep': 5000, 'lr': 0.01, 'nodes': 50, 'layers': 2, 'act': 'sigmoid', 'corr': np.float64(0.9454131880746726), 'mse': 0.004909626208245754}


결과 값은 이전보다 좋은 mse 와 corr 을 얻음




---

#### 도전 과제 4

학습 과정에서 Epoch별 Loss 그래프를 출력하고 결과를 분석하시오.

def plot_loss(train_losses, val_losses=None):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    if val_losses is not None and len(val_losses) > 0:   # 빈 리스트도 안전 처리
        plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()

# 사용 예시 — train_model이 (model, loss_history) 반환
model = ConcreteANN(input_dim, [20, 20], 'sigmoid')
model, loss_history = train_model(model, X_train, y_train, epochs=5000, lr=0.01)
plot_loss(loss_history)          # ← history['train_loss'] 대신 loss_history


Epoch     1 | Loss: 0.141635 | LR: 0.010000
Epoch   500 | Loss: 0.009334 | LR: 0.010000
Epoch  1000 | Loss: 0.008249 | LR: 0.000625
Epoch  1500 | Loss: 0.008964 | LR: 0.000039
얼리스탑: epoch 1644에서 중단 (best loss=0.007111)



Epoch별 Loss 분석

- 초기 급감 : Loss 0.1417(ep1) -> 0.0093(ep500)
    초반 500 epoch에서 대부분 학습 완료 -> 이후 완만

- LR 스케줄러 작동 : ep500까지 LR 0.01 유지 -> 이후 손실 정체로 자동 감소
    0.01 -> 0.000625(ep1000) -> 0.000039(ep1500)
    => 학습이 정체될 때마다 ReduceLROnPlateau가 LR을 절반씩 줄여 보폭을 좁힘(미세조정)

- 수렴/정체 : ep1000 이후 Loss 0.0082~0.0090 에서 더 안 내려감
    ep1500은 오히려 약간 상승 -> LR이 너무 작아져(0.000039) 추가 진전 없음

- 얼리스탑 작동 : ep1644에서 개선 없어 중단
    best loss 0.007111 시점의 가중치로 복원
    => 불필요한 학습(과적합·시간 낭비) 방지

결론 : 초반 빠른 학습 -> 후반 LR 축소로 미세조정 -> 정체 시 조기 종료까지
       의도대로 정상 학습됨. best=0.0071로 잘 수렴.



---

#### 도전 과제 5

실제값(Actual)과 예측값(Predicted)을 Scatter Plot으로 시각화하고 모델 성능을 분석하시오.

# 테스트 예측값·실제값
pred_np, corr, mse = evaluate_model(model, X_test, y_test)
actual = y_test.cpu().numpy().reshape(-1)

plt.figure(figsize=(6, 6))
plt.scatter(actual, pred_np, alpha=0.5, label='예측 결과')

# 완벽예측선 y=x (점이 이 선에 붙을수록 정확)
lo, hi = actual.min(), actual.max()
plt.plot([lo, hi], [lo, hi], 'r--', label='Perfect (y=x)')

plt.xlabel('Actual (정규화 strength)')
plt.ylabel('Predicted (정규화 strength)')
plt.title(f'Actual vs Predicted  (corr={corr:.4f}, MSE={mse:.5f})')
plt.legend()
plt.grid(True)
plt.show()




실제값 vs 예측값 Scatter Plot 분석

- 대부분의 점이 y=x(빨간 점선) 주변에 모여 있음
    => 예측값이 실제값을 전반적으로 잘 따라감 (corr 약 0.94)

- 중간 강도 구간(점 밀집 구역)은 대각선에 바짝 붙음 -> 예측 정확

- 양 끝(아주 낮거나 높은 강도)에서 점이 선에서 더 벌어짐
    => 극단값은 학습 데이터가 적어 예측이 상대적으로 부정확
    => 모델이 평균 쪽으로 당겨 예측하는 경향(회귀의 일반적 특성)

- 선 위/아래로 흩어진 정도 = 오차 -> MSE 값으로 정량화됨

결론 : 점들이 대각선에 밀집 -> 모델이 콘크리트 강도를 잘 예측.
       다만 극단 강도 구간의 예측 정확도는 다소 낮음.

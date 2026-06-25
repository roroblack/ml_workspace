# 과적합 방지 최신 기법 리포트

작성 기준: 2026-06-17, 웹서치 및 로컬 `from_colab_deep_learning` 실습 비교 기준.

## 1. 결론

과적합 방지는 한 가지 기술로 끝나는 문제가 아니라, `데이터 다양화 + 모델 용량 제어 + 손실/라벨 정규화 + 최적화 정규화 + 검증 기반 학습 제어`를 함께 쓰는 쪽이 현재 실무 기준에 가깝다.

`from_colab_deep_learning/0617-s` 실습은 이미 기초 조합을 잘 다룬다.

- PyTorch: 데이터 증강, Dropout, BatchNorm, AdamW weight decay, EarlyStopping, label smoothing, 모델 단순화
- TensorFlow/Keras: 데이터 증강, Dropout, BatchNorm, L2, AdamW, EarlyStopping, 모델 단순화

다만 최신 실무형으로 올리려면 아래가 빠져 있다.

- EarlyStopping에 `min_delta`, `start_from_epoch`, 명시적 `mode`, checkpoint 저장
- `ReduceLROnPlateau` 또는 cosine 계열 스케줄러
- EMA/SWA 같은 weight averaging
- 이미지 분류에서 MixUp/CutMix 또는 auto augmentation
- TensorFlow 실습의 실제 label smoothing 적용 누락 수정

## 2. 분류별 최신 과적합 방지 기법

### 데이터 레벨

| 분류 | 대표 기법 | 언제 쓰나 | 주의점 |
| --- | --- | --- | --- |
| 기본 증강 | crop, flip, rotate, affine, color jitter, random erasing | 이미지 데이터가 적거나 위치/회전 변화가 자연스러울 때 | 검증/테스트에는 적용하지 않는다 |
| 자동 증강 | AutoAugment, RandAugment, TrivialAugmentWide, AugMix | 수동 증강 정책을 튜닝하기 어렵거나 CV 성능을 끌어올릴 때 | 작은 이미지나 흑백 데이터에는 강도가 과하면 성능 하락 |
| 샘플 혼합 | MixUp, CutMix | 분류 모델이 특정 샘플을 외우는 경향이 강할 때 | 라벨이 soft target이 되므로 loss/accuracy 계산을 맞춰야 한다 |
| 데이터 분리 | train/validation/test 엄격 분리, stratified split, k-fold | 작은 데이터셋, 클래스 불균형, 튜닝 반복이 많을 때 | 검증셋에 튜닝 정보가 과도하게 새면 test 성능이 떨어진다 |
| 노이즈 대응 | label cleaning, robust loss, noisy-label aware training | 라벨 오류가 많을 때 | 무조건 어려운 샘플만 강조하면 노이즈를 더 외울 수 있다 |

최신 PyTorch/Torchvision 쪽은 v2 transforms 사용을 권한다. 공식 문서는 v2가 CutMix/MixUp 같은 추가 변환을 지원하고, 향후 개선도 v2에 들어간다고 설명한다. 또한 CutMix/MixUp은 배치 단위로 적용되며 분류 정확도 개선에 쓰이는 인기 증강이다.

### 모델/아키텍처 레벨

| 분류 | 대표 기법 | 언제 쓰나 | 주의점 |
| --- | --- | --- | --- |
| 모델 단순화 | layer/width/depth 축소, pruning | train 성능만 높고 val 성능이 나쁠 때 | 너무 줄이면 underfitting |
| Dropout 계열 | Dropout, SpatialDropout, DropBlock | Dense/Transformer/작은 CNN에서 외움이 강할 때 | BatchNorm과 함께 쓸 때 위치/비율 조절 필요 |
| Stochastic depth | residual branch/drop path | ResNet, ConvNeXt, ViT 계열 | MLP 단독 모델에는 직접 효과가 작다 |
| Normalization | BatchNorm, LayerNorm, GroupNorm | 학습 안정화와 약한 정규화 | BatchNorm은 작은 batch에서 불안정할 수 있다 |
| Pretrained backbone | transfer learning, freezing, fine-tuning | 데이터가 적고 유사 도메인의 사전학습 모델이 있을 때 | 전체 fine-tuning은 작은 데이터에서 다시 과적합 가능 |

Keras 공식 문서는 Dropout이 학습 중 입력 일부를 0으로 만들어 과적합을 줄인다고 설명한다. Torchvision은 stochastic depth를 residual branch를 무작위로 drop하는 기법으로 제공한다.

### 손실/라벨 레벨

| 분류 | 대표 기법 | 언제 쓰나 | 주의점 |
| --- | --- | --- | --- |
| Label smoothing | CrossEntropy label smoothing | 모델이 정답 확률을 과하게 확신할 때 | 너무 크게 주면 분류 경계가 흐려진다 |
| Focal loss | Binary/Categorical focal crossentropy | 클래스 불균형, 쉬운 샘플이 많은 문제 | 과적합 방지보다는 hard example 집중 성격 |
| Soft labels | MixUp/CutMix, distillation | 샘플 사이 경계를 부드럽게 만들 때 | metric 계산 시 hard label 변환 필요 |
| Confidence penalty | entropy regularization | 과신 예측을 줄이고 calibration 개선 | 모든 문제에 항상 이득은 아님 |

PyTorch `CrossEntropyLoss`는 `label_smoothing` 인자를 제공한다. Keras의 `CategoricalCrossentropy`는 `label_smoothing`을 지원하지만, `SparseCategoricalCrossentropy`는 정수 라벨용이라 label smoothing 인자가 없다.

### 최적화/학습 레벨

| 분류 | 대표 기법 | 언제 쓰나 | 주의점 |
| --- | --- | --- | --- |
| Decoupled weight decay | AdamW | Adam을 쓰면서 L2/weight decay를 안정적으로 적용할 때 | bias/Norm에는 decay 제외하는 경우도 많다 |
| LR scheduler | ReduceLROnPlateau, cosine, OneCycle | validation 정체나 후반 수렴 개선 | scheduler와 early stopping patience를 충돌시키지 않는다 |
| SAM/ASAM | sharpness-aware minimization | 큰 모델의 일반화 개선 | 한 step에 forward/backward가 2번 필요해 느리다 |
| EMA/SWA | exponential moving average, stochastic weight averaging | 마지막 가중치보다 안정적인 평균 가중치가 필요할 때 | BatchNorm 통계 처리 필요 |
| Gradient clipping | clip norm/value | RNN/Transformer/불안정 학습 | 직접적인 과적합 방지라기보다 안정화 |

PyTorch 공식 문서는 AdamW가 weight decay를 momentum/variance 누적과 분리한다고 설명한다. `ReduceLROnPlateau`는 metric이 정체될 때 학습률을 낮춘다. SAM 논문은 train loss 값만 최적화하면 일반화 보장이 약하므로 loss sharpness도 함께 낮추는 방식이 일반화에 도움을 준다고 제안한다. PyTorch `AveragedModel`은 SWA와 EMA를 지원한다.

### 학습 제어/검증 레벨

| 분류 | 대표 기법 | 핵심 |
| --- | --- | --- |
| EarlyStopping | 검증 성능이 더 이상 개선되지 않으면 중단 |
| ModelCheckpoint | best epoch의 모델을 파일로 보존 |
| Validation protocol | stratified split, k-fold, leakage 방지 |
| 실험 추적 | seed, config, metric, checkpoint 기록 |

EarlyStopping은 단순히 "epoch 줄이기"가 아니라, 검증 metric을 기준으로 `best epoch`를 선택하는 모델 선택 절차다. Keras 공식 API는 `monitor`, `min_delta`, `patience`, `mode`, `restore_best_weights`, `start_from_epoch`를 제공한다.

## 3. EarlyStopping에서 실제로 하는 일

문맥상 사용자가 말한 "얼리엑세스"는 실습의 `Early Stopping`으로 해석했다.

EarlyStopping의 작업 흐름은 다음과 같다.

1. 매 epoch 종료 후 validation metric을 계산한다.
2. `monitor` 대상이 이전 best보다 충분히 좋아졌는지 본다.
3. 좋아졌다면 best metric과 best weights를 저장하고 counter를 0으로 돌린다.
4. 좋아지지 않았다면 counter를 1 올린다.
5. counter가 `patience` 이상이면 학습을 중단한다.
6. `restore_best_weights=True`이면 마지막 epoch 가중치가 아니라 best epoch 가중치로 되돌린다.

현대식으로는 여기에 다음을 붙인다.

- `min_delta`: 0.000001 같은 잡음성 개선은 개선으로 보지 않는다.
- `start_from_epoch`: 초반 warm-up 기간에는 중단 판단을 하지 않는다.
- `mode='min'/'max'`: `val_loss`는 min, `val_accuracy`는 max를 명시한다.
- `ModelCheckpoint`: 메모리뿐 아니라 파일에도 best model을 남긴다.
- `ReduceLROnPlateau`: 바로 중단하기 전에 학습률을 낮춰 한 번 더 기회를 준다.
- EMA/SWA 평가: 마지막 raw weights보다 평균 weights가 더 안정적인 경우가 많다.

## 4. 로컬 실습과 비교

대상 파일:

- `from_colab_deep_learning/0617-s/deep_learning_overfitting_prevention_torch.ipynb`
- `from_colab_deep_learning/0617-s/deep_learning_overfitting_prevention_tensorflow.ipynb`

### PyTorch 실습

현재 적용:

- `RegularizedMLP`
- `Dropout(0.5, 0.3)`
- `BatchNorm1d`
- `AdamW(weight_decay=0.01)`
- `CrossEntropyLoss(label_smoothing=0.1)`
- `patience=5` 기반 EarlyStopping
- 기본 증강 `RandomRotation`, `RandomAffine`

개선점:

- EarlyStopping에 `min_delta`와 `start_from_epoch` 추가
- `val_loss < best_val_loss` 대신 noise threshold 사용
- `ReduceLROnPlateau` 추가
- best state를 메모리뿐 아니라 checkpoint 파일로 저장
- MLP 대신 작은 CNN으로 이미지 구조 활용
- Torchvision v2 + MixUp/CutMix 도입
- EMA weights를 검증/최종 평가에 활용

### TensorFlow/Keras 실습

현재 적용:

- `data_augmentation`
- Dense 기반 정규화 모델
- `regularizers.l2(0.001)`
- `Dropout(0.5, 0.3)`
- `AdamW(weight_decay=0.01)`
- `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`

확인된 문제:

- 설명에는 `SparseCategoricalCrossentropy(label_smoothing=0.1)`라고 되어 있지만 실제 코드는 `SparseCategoricalCrossentropy()`라 label smoothing이 적용되지 않는다.
- Keras에서 label smoothing을 쓰려면 one-hot label과 `CategoricalCrossentropy(label_smoothing=...)` 조합을 쓰는 편이 명확하다.

개선점:

- `EarlyStopping(min_delta=1e-4, start_from_epoch=5, mode='min')`
- `ModelCheckpoint(save_best_only=True)`
- `ReduceLROnPlateau`
- `AdamW(use_ema=True)` + `SwapEMAWeights`
- MLP 대신 작은 CNN
- L2와 AdamW를 동시에 강하게 걸지 말고 둘 중 하나를 약하게 조정

## 5. 권장 조합

Fashion-MNIST 같은 소형 이미지 분류 실습 기준:

1. 모델: MLP보다 작은 CNN
2. 데이터: 기본 affine 증강 + 필요 시 MixUp/CutMix
3. 손실: label smoothing 0.05~0.1
4. 최적화: AdamW, weight_decay 1e-4~1e-3부터 탐색
5. 스케줄러: ReduceLROnPlateau 또는 cosine
6. 종료: EarlyStopping with `min_delta`, `start_from_epoch`, checkpoint
7. 추가 안정화: EMA weights

실습용으로는 `AdamW + label smoothing + robust EarlyStopping + ReduceLROnPlateau + checkpoint`가 1차 최적 조합이다. 성능을 더 올리고 싶으면 `CNN + MixUp/CutMix + EMA`를 붙인다.

## 6. 제공 코드

아래 파일에 실제 적용 가능한 코드를 분리해 두었다.

- `modern_overfitting_pytorch.py`: PyTorch 실습 개선판
- `modern_overfitting_tensorflow.py`: TensorFlow/Keras 실습 개선판

## 7. 참고 출처

- Keras EarlyStopping API: https://keras.io/api/callbacks/early_stopping/
- Keras ModelCheckpoint API: https://keras.io/api/callbacks/model_checkpoint/
- Keras Dropout API: https://keras.io/api/layers/regularization_layers/dropout/
- Keras Regularizers API: https://keras.io/api/layers/regularizers/
- Keras Probabilistic Losses, CategoricalCrossentropy, Focal loss: https://keras.io/api/losses/probabilistic_losses/
- Keras SwapEMAWeights API: https://keras.io/api/callbacks/swap_ema_weights/
- PyTorch AdamW API: https://docs.pytorch.org/docs/2.12/generated/torch.optim.AdamW.html
- PyTorch CrossEntropyLoss label_smoothing: https://docs.pytorch.org/docs/2.12/generated/torch.nn.CrossEntropyLoss.html
- PyTorch ReduceLROnPlateau API: https://docs.pytorch.org/docs/2.12/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
- PyTorch AveragedModel SWA/EMA API: https://docs.pytorch.org/docs/2.12/generated/torch.optim.swa_utils.AveragedModel.html
- Torchvision v2 transforms: https://docs.pytorch.org/vision/main/transforms.html
- Torchvision CutMix/MixUp tutorial: https://docs.pytorch.org/vision/main/auto_examples/transforms/plot_cutmix_mixup.html
- Torchvision stochastic_depth API: https://docs.pytorch.org/vision/main/generated/torchvision.ops.stochastic_depth.html
- SAM paper: https://arxiv.org/abs/2010.01412
- EMA of Weights paper: https://arxiv.org/html/2411.18704v1

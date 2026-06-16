# ============================================================
# 1. 기본 라이브러리 불러오기
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch 핵심 라이브러리
import torch

# 신경망 모델을 만들기 위한 모듈
import torch.nn as nn

# 최적화 알고리즘을 사용하기 위한 모듈
import torch.optim as optim

# PyTorch Dataset, DataLoader 사용
from torch.utils.data import TensorDataset, DataLoader

# Iris 데이터셋
from sklearn.datasets import load_iris

# 학습용 / 테스트용 데이터 분리
from sklearn.model_selection import train_test_split

# 데이터 표준화
from sklearn.preprocessing import StandardScaler

# 평가 지표
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# 2. 실행 장치 설정 - GPU 사용 가능 여부 확인
# ============================================================

# Colab에서 GPU를 켜면 cuda 사용 가능
# GPU가 없으면 CPU 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("사용 장치:", device)

# ============================================================
# 3. 데이터 불러오기
# ============================================================

iris = load_iris()

# 입력 데이터
X = iris.data

# 정답 데이터
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df["target"] = y
df["target_name"] = df["target"].apply(lambda x: target_names[x])

df.head()

# ============================================================
# 4. 데이터 기본 정보 확인
# ============================================================

print("데이터 크기:", df.shape)

print("\n기초 통계량:")
print(df.describe())

print("\n품종별 개수:")
print(df["target_name"].value_counts())

# ============================================================
# 5. 데이터 시각화
# ============================================================

sns.countplot(data=df, x="target_name")
plt.title("Iris 품종별 데이터 개수")
plt.xlabel("품종")
plt.ylabel("개수")
plt.show()

sns.pairplot(df, hue="target_name")
plt.show()

# ============================================================
# 6. 입력 데이터와 정답 데이터 분리
# ============================================================

X = df[feature_names].values
y = df["target"].values

print("입력 데이터 크기:", X.shape)
print("정답 데이터 크기:", y.shape)

# ============================================================
# 7. 학습용 / 테스트용 데이터 분리
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("학습용 입력 데이터:", X_train.shape)
print("테스트용 입력 데이터:", X_test.shape)

# ============================================================
# 8. 데이터 전처리 - 표준화
# ============================================================

scaler = StandardScaler()

# 학습 데이터 기준으로 평균과 표준편차 계산 후 변환
X_train_scaled = scaler.fit_transform(X_train)

# 테스트 데이터는 학습 데이터 기준으로만 변환
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 9. NumPy 데이터를 PyTorch Tensor로 변환
# ============================================================

# 입력 데이터는 float32 타입으로 변환
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# 정답 데이터는 CrossEntropyLoss 사용을 위해 long 타입으로 변환
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

print(X_train_tensor.shape)
print(y_train_tensor.shape)

# ============================================================
# 10. Dataset과 DataLoader 생성
# ============================================================

# TensorDataset은 입력 데이터와 정답 데이터를 하나로 묶는다
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

# DataLoader는 데이터를 batch 단위로 나누어 학습에 사용한다
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

# ============================================================
# 11. PyTorch 신경망 모델 정의
# ============================================================

class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()

        # 입력 특성은 4개
        self.fc1 = nn.Linear(4, 16)

        # 은닉층 # 중간단계의 모든 층 (모들들)
        self.fc2 = nn.Linear(16, 16)

        # 출력 클래스는 3개
        self.fc3 = nn.Linear(16, 3)

        # 활성화 함수
        self.relu = nn.ReLU()

    def forward(self, x):
        # 첫 번째 완전연결층 (fully connected == fc)
        x = self.fc1(x)
        x = self.relu(x)

        # 두 번째 완전연결층
        x = self.fc2(x)
        x = self.relu(x)

        # 출력층
        # CrossEntropyLoss는 내부적으로 softmax를 처리하므로 여기서 softmax를 쓰지 않는다
        x = self.fc3(x)

        return x

# ============================================================
# 12. 모델 생성
# ============================================================

model = IrisNet().to(device)

print(model)

# ============================================================
# 13. 손실 함수와 최적화 함수 정의
# ============================================================

# 다중 분류 문제이므로 CrossEntropyLoss 사용
criterion = nn.CrossEntropyLoss()

# Adam 옵티마이저 사용
optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)

# ============================================================
# 14. 모델 학습
# ============================================================

epochs = 100

loss_history = []

for epoch in range(epochs):
    model.train()

    total_loss = 0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # 이전 계산된 기울기 초기화
        optimizer.zero_grad()

        # 모델 예측
        outputs = model(batch_X)

        # 손실 계산
        loss = criterion(outputs, batch_y)

        # 역전파
        loss.backward()

        # 가중치 업데이트
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# ============================================================
# 15. 학습 손실 그래프 확인
# ============================================================

plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# ============================================================
# 16. 테스트 데이터 예측
# ============================================================

model.eval()

with torch.no_grad():
    X_test_device = X_test_tensor.to(device)

    outputs = model(X_test_device)

    # 가장 큰 점수를 가진 클래스 선택
    _, predicted = torch.max(outputs, 1)

y_pred = predicted.cpu().numpy()

print("예측 결과:", y_pred)
print("실제 정답:", y_test)

# ============================================================
# 17. 모델 평가
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("모델 정확도:", accuracy)
print("모델 정확도(%):", accuracy * 100)

print(classification_report(
    y_test,
    y_pred,
    target_names=target_names
))

# ============================================================
# 18. 혼동 행렬 확인
# ============================================================

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names
)

plt.xlabel("예측값")
plt.ylabel("실제값")
plt.title("Confusion Matrix")
plt.show()

# ============================================================
# 19. 새로운 데이터 예측
# ============================================================

new_data = [[5.1, 3.5, 1.4, 0.2]]

new_data_scaled = scaler.transform(new_data)

new_data_tensor = torch.tensor(
    new_data_scaled,
    dtype=torch.float32
).to(device)

model.eval()

with torch.no_grad():
    output = model(new_data_tensor)
    _, prediction = torch.max(output, 1)

predicted_class = prediction.item()
predicted_name = target_names[predicted_class]

print("새 데이터 예측 번호:", predicted_class)
print("새 데이터 예측 품종:", predicted_name)

# ============================================================
# 20. PyTorch 모델 저장
# ============================================================

torch.save(model.state_dict(), "iris_pytorch_model.pth")

print("PyTorch 모델 저장 완료")

# ============================================================
# 21. 저장된 PyTorch 모델 불러오기
# ============================================================

loaded_model = IrisNet().to(device)

loaded_model.load_state_dict(torch.load("iris_pytorch_model.pth", map_location=device))

loaded_model.eval()

print("저장된 PyTorch 모델 불러오기 완료")

# ============================================================
# 22. 불러온 모델로 다시 예측
# ============================================================

sample = [[6.2, 3.4, 5.4, 2.3]]

sample_scaled = scaler.transform(sample)

sample_tensor = torch.tensor(
    sample_scaled,
    dtype=torch.float32
).to(device)

with torch.no_grad():
    sample_output = loaded_model(sample_tensor)
    _, sample_pred = torch.max(sample_output, 1)

print("예측 번호:", sample_pred.item())
print("예측 품종:", target_names[sample_pred.item()])

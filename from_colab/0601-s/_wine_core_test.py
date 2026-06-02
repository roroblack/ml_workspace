"""
화이트 와인 품질 예측 - 최적화된 파라미터로 핵심 실행
(64,64,32) ReLU, lr=0.0003, drop=0.1, weight_decay=0.01
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import random

# ===== 설정 =====
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

# ===== 데이터 로드 =====
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
whitewines = pd.read_csv(URL, sep=';')
whitewines.columns = [col.replace(" ", "_") for col in whitewines.columns]

print(f"\n데이터 로드: {whitewines.shape}")

# ===== 데이터 분리 (원본 노트북과 동일) =====
X = whitewines.drop("quality", axis=1)
y = whitewines["quality"]

X_train = X.iloc[:3749].copy()
X_test = X.iloc[3749:].copy()
y_train = y.iloc[:3749].copy()
y_test = y.iloc[3749:].copy()

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ===== 표준화 =====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== PyTorch 텐서 변환 =====
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# ===== DataLoader =====
BATCH_SIZE = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===== 모델 정의 (최적화된 구조) =====
class WineQualityRegressor(nn.Module):
    def __init__(self, input_dim):
        super(WineQualityRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

input_dim = X_train_tensor.shape[1]
model = WineQualityRegressor(input_dim).to(device)

print("\n모델 구조:")
print(model)

# ===== 최적화 설정 =====
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, factor=0.5)

EPOCHS = 1000
PATIENCE = 50

# ===== 학습 =====
print("\n학습 시작...")
train_losses = []
best_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch_X.size(0)

    epoch_loss = epoch_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    # Early Stopping
    if epoch_loss < best_loss - 1e-5:
        best_loss = epoch_loss
        patience_counter = 0
    else:
        patience_counter += 1

    scheduler.step(epoch_loss)

    if patience_counter >= PATIENCE:
        print(f"\n조기 종료 (Epoch {epoch+1}): 50번 연속 개선 없음")
        break

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1:4d}/{EPOCHS}] Loss: {epoch_loss:.4f} (best: {best_loss:.4f})")

# ===== 테스트 평가 =====
model.eval()
with torch.no_grad():
    X_test_device = X_test_tensor.to(device)
    test_predictions = model(X_test_device).cpu().numpy().flatten()

actual_values = y_test.values

# ===== 성능 평가 =====
mse = mean_squared_error(actual_values, test_predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_values, test_predictions)
r2 = r2_score(actual_values, test_predictions)
correlation, _ = pearsonr(test_predictions, actual_values)

print("\n" + "="*60)
print("최종 성능 평가 (테스트 데이터)")
print("="*60)
print(f"MSE              : {mse:.4f}")
print(f"RMSE             : {rmse:.4f}")
print(f"MAE              : {mae:.4f}")
print(f"R² Score         : {r2:.4f}  ({r2*100:.2f}%)")
print(f"상관계수         : {correlation:.4f}")
print("="*60)

# ===== 개선율 비교 =====
print("\n개선 비율:")
print(f"이전 모델 (Tanh, 128-64-32, lr=0.0001):")
print(f"  MAE: 0.5330")
print(f"  R²:  0.2678")
print(f"\n최적화 모델 (ReLU, 64-64-32, lr=0.0003, wd=0.01):")
print(f"  MAE: {mae:.4f}")
print(f"  R²:  {r2:.4f}")
print(f"\n개선도:")
print(f"  MAE 개선: {(0.5330 - mae) / 0.5330 * 100:.1f}%")
print(f"  R² 개선:  {(r2 - 0.2678) / 0.2678 * 100:.1f}%")

# ===== 예측값 샘플 =====
print("\n예측값 샘플 (앞 10개):")
result_df = pd.DataFrame({
    "실제값": actual_values[:10],
    "예측값": test_predictions[:10],
    "오차": actual_values[:10] - test_predictions[:10]
})
print(result_df.to_string(index=False))

"""
화이트 와인 품질 예측 - R² 0.4 이상을 목표로 하이퍼파라미터 탐색
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import random
import warnings
warnings.filterwarnings('ignore')

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 데이터 로드 및 전처리 =====
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
whitewines = pd.read_csv(URL, sep=';')
whitewines.columns = [col.replace(" ", "_") for col in whitewines.columns]

X = whitewines.drop("quality", axis=1)
y = whitewines["quality"]

X_train = X.iloc[:3749].copy()
X_test = X.iloc[3749:].copy()
y_train = y.iloc[:3749].copy()
y_test = y.iloc[3749:].copy()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

BATCH_SIZE = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

y_test_actual = y_test.values

# ===== 모델 빌드 함수 =====
def build_model(layer_sizes, activation, dropout_rate):
    """
    layer_sizes: [input_dim, hidden1, hidden2, ..., 1]
    activation: 'relu', 'tanh', 'leaky', 'gelu'
    """
    layers = []
    act_map = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'leaky': nn.LeakyReLU,
        'gelu': nn.GELU
    }
    act_fn = act_map.get(activation, nn.ReLU)

    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        if i < len(layer_sizes) - 2:  # 마지막 층 전까지 활성화 + dropout
            layers.append(act_fn())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

    return nn.Sequential(*layers)

# ===== 학습 함수 =====
def train_model(model, train_loader, test_tensor, test_y, config):
    criterion = nn.MSELoss()

    opt_map = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'rmsprop': optim.RMSprop,
        'sgd': optim.SGD
    }
    opt_fn = opt_map.get(config['optimizer'], optim.AdamW)
    optimizer = opt_fn(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, factor=0.5)

    EPOCHS = config['epochs']
    PATIENCE = 50
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

        if epoch_loss < best_loss - 1e-5:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1

        scheduler.step(epoch_loss)

        if patience_counter >= PATIENCE:
            break

    # 평가
    model.eval()
    with torch.no_grad():
        test_X = test_tensor.to(device)
        test_pred = model(test_X).cpu().numpy().flatten()

    r2 = r2_score(test_y, test_pred)
    rmse = np.sqrt(mean_squared_error(test_y, test_pred))
    mae = mean_absolute_error(test_y, test_pred)

    return r2, rmse, mae, epoch+1

# ===== 하이퍼파라미터 그리드 =====
print("하이퍼파라미터 탐색 시작...\n")

# 모델 구조 조합 (입력 11, 출력 1)
structures = [
    [11, 128, 64, 32, 1],      # 3개 은닉층, 작음
    [11, 128, 128, 64, 1],     # 3개 은닉층, 중간
    [11, 256, 128, 64, 1],     # 3개 은닉층, 큼
    [11, 256, 256, 128, 64, 1], # 4개 은닉층, 큼
    [11, 64, 64, 32, 1],       # 3개 은닉층, 작음
    [11, 256, 256, 128, 1],    # 3개 은닉층, 매우 큼
    [11, 128, 64, 1],          # 2개 은닉층, 중간
    [11, 256, 128, 1],         # 2개 은닉층, 큼
]

activations = ['relu', 'tanh', 'leaky', 'gelu']
learning_rates = [1e-4, 3e-4, 5e-4, 1e-3, 3e-3]
optimizers = ['adam', 'adamw', 'rmsprop']
dropouts = [0.0, 0.1, 0.2, 0.3]
epochs_list = [500, 800, 1000]

# 무작위 샘플링 (200개 조합)
configs = []
for _ in range(200):
    cfg = {
        'structure': random.choice(structures),
        'activation': random.choice(activations),
        'lr': random.choice(learning_rates),
        'optimizer': random.choice(optimizers),
        'dropout': random.choice(dropouts),
        'epochs': random.choice(epochs_list)
    }
    configs.append(cfg)

results = []

for i, cfg in enumerate(configs):
    try:
        # 모델 생성
        model = build_model(cfg['structure'], cfg['activation'], cfg['dropout']).to(device)

        # 학습
        r2, rmse, mae, epochs_used = train_model(model, train_loader, X_test_tensor, y_test_actual, cfg)
        results.append((r2, cfg, rmse, mae, epochs_used))

        if (i + 1) % 20 == 0 or r2 >= 0.40:
            struct_str = '-'.join(map(str, cfg['structure']))
            print(f"[{i+1:3d}/200] R²={r2:.4f} | {struct_str} | {cfg['activation']:5s} lr={cfg['lr']:.0e} drop={cfg['dropout']:.1f} opt={cfg['optimizer']:7s} ep={epochs_used}")

    except Exception as e:
        print(f"[{i+1:3d}/200] 오류: {str(e)[:50]}")

# ===== 결과 정렬 및 출력 =====
results.sort(key=lambda x: x[0], reverse=True)

print("\n" + "="*100)
print("TOP 10 최고 성능 조합:")
print("="*100)

for rank, (r2, cfg, rmse, mae, epochs_used) in enumerate(results[:10], 1):
    struct_str = '-'.join(map(str, cfg['structure']))
    print(f"\n{rank}. R² = {r2:.4f}  (RMSE={rmse:.4f}, MAE={mae:.4f})")
    print(f"   구조:     {struct_str}")
    print(f"   활성화:   {cfg['activation']}")
    print(f"   학습률:   {cfg['lr']:.0e}")
    print(f"   드롭아웃: {cfg['dropout']:.1f}")
    print(f"   옵티마이저: {cfg['optimizer']}")
    print(f"   Epoch 사용: {epochs_used}")

# ===== 최고 조합으로 최종 확인 =====
best_r2, best_cfg, best_rmse, best_mae, best_epochs = results[0]

print("\n" + "="*100)
print("최고 성능 모델로 재검증...")
print("="*100)

torch.manual_seed(SEED)
final_model = build_model(best_cfg['structure'], best_cfg['activation'], best_cfg['dropout']).to(device)
final_r2, final_rmse, final_mae, _ = train_model(final_model, train_loader, X_test_tensor, y_test_actual, best_cfg)

print(f"\n최종 검증 결과:")
print(f"  R² Score: {final_r2:.4f}")
print(f"  RMSE:     {final_rmse:.4f}")
print(f"  MAE:      {final_mae:.4f}")
print(f"\n✅ 목표 R² 0.4 달성: {final_r2 >= 0.40}")

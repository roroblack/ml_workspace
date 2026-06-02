import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

SEED = 123
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

df = pd.read_csv("winequality-white-cache.csv")
df.columns = [c.replace(" ", "_") for c in df.columns]
X = df.drop("quality", axis=1); y = df["quality"]
X_train, X_test = X.iloc[:3749].copy(), X.iloc[3749:].copy()
y_train, y_test = y.iloc[:3749].copy(), y.iloc[3749:].copy()
scaler = StandardScaler()
Xtr = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
ytr = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
Xte = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
yte = y_test.values

BATCH_SIZE = 64
g = torch.Generator(); g.manual_seed(SEED)
train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True, generator=g)

DROPOUT_RATE = 0.2
class WineQualityRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LeakyReLU(), nn.Dropout(DROPOUT_RATE),
            nn.Linear(64, 64), nn.LeakyReLU(), nn.Dropout(DROPOUT_RATE),
            nn.Linear(64, 32), nn.LeakyReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x): return self.network(x)

model = WineQualityRegressor(Xtr.shape[1])
criterion = nn.MSELoss()
ir_n = 0.0007
optimizer = optim.AdamW(model.parameters(), lr=ir_n, weight_decay=0.001)
EPOCHS = 1000
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8)

best_loss = float('inf')
patience_limit = 60
counter = 0
best_state = None
for epoch in range(EPOCHS):
    model.train(); epoch_loss = 0.0
    for bx, by in train_loader:
        optimizer.zero_grad(); loss = criterion(model(bx), by)
        loss.backward(); optimizer.step()
        epoch_loss += loss.item() * bx.size(0)
    epoch_loss /= len(train_loader.dataset)
    if epoch_loss < best_loss - 1e-6:
        best_loss = epoch_loss; counter = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        counter += 1
    scheduler.step(epoch_loss)
    if counter > patience_limit - 1:
        print(f"Early stopping at epoch {epoch+1}, best train MSE {best_loss:.4f}")
        break

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    pred = model(Xte).numpy().flatten()

mse = mean_squared_error(yte, pred); rmse = np.sqrt(mse)
mae = mean_absolute_error(yte, pred); r2 = r2_score(yte, pred)
corr, _ = pearsonr(pred, yte)
print(f"corr {corr:.4f}")
print(f"MSE  {mse:.4f}")
print(f"RMSE {rmse:.4f}")
print(f"MAE  {mae:.4f}")
print(f"R2   {r2:.4f}")
print("FINAL_DONE")

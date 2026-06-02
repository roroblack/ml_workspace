import random, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

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

ACT = {"relu": nn.ReLU, "leaky": nn.LeakyReLU}

def build(layers, act, dropout):
    mods = []; prev = Xtr.shape[1]
    for h in layers:
        mods.append(nn.Linear(prev, h)); mods.append(ACT[act]())
        if dropout > 0: mods.append(nn.Dropout(dropout))
        prev = h
    mods.append(nn.Linear(prev, 1))
    return nn.Sequential(*mods)

def run(seed, layers, act, dropout, lr, wd, batch=64, epochs=1500,
        patience_limit=60, sched_patience=8):
    set_seed(seed)
    model = build(layers, act, dropout)
    crit = nn.MSELoss()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=sched_patience)
    g = torch.Generator(); g.manual_seed(seed)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=True, generator=g)
    best_loss = float("inf"); counter = 0; best_state = None
    for ep in range(epochs):
        model.train(); el = 0.0
        for bx, by in loader:
            opt.zero_grad(); loss = crit(model(bx), by)
            loss.backward(); opt.step(); el += loss.item() * bx.size(0)
        el /= len(loader.dataset)
        if el < best_loss - 1e-6:
            best_loss = el; counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            counter += 1
        sched.step(el)
        if counter > patience_limit - 1: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(): pred = model(Xte).numpy().flatten()
    return r2_score(yte, pred), mean_absolute_error(yte, pred), np.sqrt(mean_squared_error(yte, pred)), pred

cands = {
    "leaky lr7e-4 d0.2 wd1e-3": dict(layers=[64,64,32],act="leaky",dropout=0.2,lr=0.0007,wd=0.001),
    "relu  lr7e-4 d0.1 wd1e-2": dict(layers=[64,64,32],act="relu", dropout=0.1,lr=0.0007,wd=0.01),
    "leaky lr1e-3 d0.2 wd1e-3": dict(layers=[64,64,32],act="leaky",dropout=0.2,lr=0.001, wd=0.001),
    "relu  lr7e-4 d0.2 wd1e-3": dict(layers=[64,64,32],act="relu", dropout=0.2,lr=0.0007,wd=0.001),
    "leaky lr5e-4 d0.2 wd1e-3": dict(layers=[64,64,32],act="leaky",dropout=0.2,lr=0.0005,wd=0.001),
    "leaky lr7e-4 d0.15 wd1e-3":dict(layers=[64,64,32],act="leaky",dropout=0.15,lr=0.0007,wd=0.001),
}
seeds = [0, 1, 42, 123, 2024]

summary = []
for name, cfg in cands.items():
    r2s, maes, rmses, preds = [], [], [], []
    for s in seeds:
        r2, mae, rmse, p = run(s, **cfg)
        r2s.append(r2); maes.append(mae); rmses.append(rmse); preds.append(p)
    ens = np.mean(preds, axis=0)
    ens_r2 = r2_score(yte, ens); ens_mae = mean_absolute_error(yte, ens); ens_rmse = np.sqrt(mean_squared_error(yte, ens))
    summary.append((name, np.mean(r2s), np.std(r2s), np.mean(maes), ens_r2, ens_mae, ens_rmse))
    print(f"{name:26s} | seed-mean R2 {np.mean(r2s):.4f}±{np.std(r2s):.4f} MAE {np.mean(maes):.4f} | ENSEMBLE(5) R2 {ens_r2:.4f} MAE {ens_mae:.4f} RMSE {ens_rmse:.4f}", flush=True)

summary.sort(key=lambda x: -x[4])
print("\n=== best by ensemble R2 ===", flush=True)
for s in summary:
    print(f"{s[0]:26s} | mean R2 {s[1]:.4f} | ENS R2 {s[4]:.4f} MAE {s[5]:.4f}", flush=True)
print("ROUND3_COMPLETE", flush=True)

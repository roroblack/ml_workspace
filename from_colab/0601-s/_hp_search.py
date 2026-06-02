import random, itertools, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SEED = 123

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

# ---- data (same pipeline as the notebook) ----
df = pd.read_csv("winequality-white-cache.csv")
df.columns = [c.replace(" ", "_") for c in df.columns]
X = df.drop("quality", axis=1)
y = df["quality"]
X_train, X_test = X.iloc[:3749].copy(), X.iloc[3749:].copy()
y_train, y_test = y.iloc[:3749].copy(), y.iloc[3749:].copy()

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

Xtr = torch.tensor(X_train_s, dtype=torch.float32)
ytr = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
Xte = torch.tensor(X_test_s, dtype=torch.float32)
yte = y_test.values

ACT = {"relu": nn.ReLU, "tanh": nn.Tanh, "leaky": nn.LeakyReLU, "gelu": nn.GELU}

def build(input_dim, layers, act, dropout):
    mods = []
    prev = input_dim
    for h in layers:
        mods.append(nn.Linear(prev, h))
        mods.append(ACT[act]())
        if dropout > 0:
            mods.append(nn.Dropout(dropout))
        prev = h
    mods.append(nn.Linear(prev, 1))
    return nn.Sequential(*mods)

def run(layers, act, dropout, lr, wd, opt_name, batch, epochs=1000,
        patience_limit=50, sched_patience=5):
    set_seed()
    model = build(Xtr.shape[1], layers, act, dropout)
    crit = nn.MSELoss()
    if opt_name == "adamw":
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adam":
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "rmsprop":
        opt = optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        opt = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=sched_patience)

    g = torch.Generator(); g.manual_seed(SEED)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=True, generator=g)

    best_loss = float("inf"); counter = 0
    best_state = None
    for ep in range(epochs):
        model.train(); el = 0.0
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward(); opt.step()
            el += loss.item() * bx.size(0)
        el /= len(loader.dataset)
        if el < best_loss - 1e-6:
            best_loss = el; counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            counter += 1
        sched.step(el)
        if counter > patience_limit - 1:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte).numpy().flatten()
    return {
        "mse": float(mean_squared_error(yte, pred)),
        "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
        "mae": float(mean_absolute_error(yte, pred)),
        "r2": float(r2_score(yte, pred)),
        "epoch": ep + 1,
    }

# baseline (current notebook config)
baseline = run([64, 64, 32], "relu", 0.1, 0.0003, 0.01, "adamw", 64)
print("BASELINE (64,64,32 relu lr3e-4 drop0.1 wd0.01 adamw b64):", baseline)

configs = []
# architectures x activations
archs = [[64,32],[128,64],[128,64,32],[256,128,64],[64,64,32],[128,128,64],[256,128],[64,32,16]]
for layers in archs:
    for act in ["relu","gelu","leaky","tanh"]:
        configs.append(dict(layers=layers, act=act, dropout=0.1, lr=0.001, wd=0.01, opt_name="adamw", batch=64))

results = []
for c in configs:
    r = run(**c)
    results.append((c, r))
    print(f"{c['layers']} {c['act']:5s} lr{c['lr']} drop{c['dropout']} wd{c['wd']} {c['opt_name']} b{c['batch']} -> R2 {r['r2']:.4f} MAE {r['mae']:.4f} RMSE {r['rmse']:.4f} ep{r['epoch']}")

results.sort(key=lambda x: -x[1]["r2"])
print("\n=== TOP 8 (round 1) ===")
for c, r in results[:8]:
    print(f"{c['layers']} {c['act']} lr{c['lr']} drop{c['dropout']} wd{c['wd']} {c['opt_name']} b{c['batch']} -> R2 {r['r2']:.4f} MAE {r['mae']:.4f}")

json.dump([{"cfg":c,"res":r} for c,r in results], open("_hp_round1.json","w"), indent=2)
print("\nBASELINE R2:", baseline["r2"])

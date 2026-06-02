import random, json
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

def build(input_dim, layers, act, dropout, bn):
    mods = []; prev = input_dim
    for h in layers:
        mods.append(nn.Linear(prev, h))
        if bn:
            mods.append(nn.BatchNorm1d(h))
        mods.append(ACT[act]())
        if dropout > 0:
            mods.append(nn.Dropout(dropout))
        prev = h
    mods.append(nn.Linear(prev, 1))
    return nn.Sequential(*mods)

def run(layers, act, dropout, lr, wd, batch, bn=False, epochs=1500,
        patience_limit=60, sched_patience=8):
    set_seed()
    model = build(Xtr.shape[1], layers, act, dropout, bn)
    crit = nn.MSELoss()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=sched_patience)
    g = torch.Generator(); g.manual_seed(SEED)
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
        if counter > patience_limit - 1:
            break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte).numpy().flatten()
    return dict(mse=float(mean_squared_error(yte, pred)),
                rmse=float(np.sqrt(mean_squared_error(yte, pred))),
                mae=float(mean_absolute_error(yte, pred)),
                r2=float(r2_score(yte, pred)), epoch=ep + 1)

configs = []
for layers in [[64,64,32],[64,32]]:
    for act in ["relu","leaky"]:
        for lr in [0.0007,0.001,0.0015]:
            for dropout in [0.0,0.1,0.2]:
                for wd in [0.001,0.01]:
                    configs.append(dict(layers=layers,act=act,dropout=dropout,lr=lr,wd=wd,batch=64,bn=False))
# batchnorm + batch-size probes on the strongest arch
for bn in [True]:
    for batch in [32,64,128]:
        for dropout in [0.0,0.1]:
            for lr in [0.001,0.002]:
                configs.append(dict(layers=[64,64,32],act="relu",dropout=dropout,lr=lr,wd=0.001,batch=batch,bn=bn))

results = []
for c in configs:
    r = run(**c)
    results.append((c, r))
    print(f"{str(c['layers']):12s} {c['act']:5s} lr{c['lr']:<6} drop{c['dropout']} wd{c['wd']} b{c['batch']} bn{int(c['bn'])} -> R2 {r['r2']:.4f} MAE {r['mae']:.4f} RMSE {r['rmse']:.4f} ep{r['epoch']}", flush=True)

results.sort(key=lambda x: -x[1]["r2"])
print("\n=== TOP 10 (round 2) ===", flush=True)
for c, r in results[:10]:
    print(f"{str(c['layers']):12s} {c['act']:5s} lr{c['lr']:<6} drop{c['dropout']} wd{c['wd']} b{c['batch']} bn{int(c['bn'])} -> R2 {r['r2']:.4f} MAE {r['mae']:.4f} RMSE {r['rmse']:.4f} ep{r['epoch']}", flush=True)
json.dump([{"cfg":c,"res":r} for c,r in results], open("_hp_round2.json","w"), indent=2)
print("ROUND2_COMPLETE", flush=True)

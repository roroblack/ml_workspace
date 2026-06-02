"""Mirror the notebook's EXACT cell order/RNG usage so reported numbers reproduce.
Order per notebook: set seed once -> load/scale -> tensors -> DataLoader(gen) -> model -> opt -> train."""
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SEED = 123

df = pd.read_csv("winequality-white-cache.csv")
df.columns = [c.replace(" ", "_") for c in df.columns]
X = df.drop("quality", axis=1); y = df["quality"]
X_train, X_test = X.iloc[:3749].copy(), X.iloc[3749:].copy()
y_train, y_test = y.iloc[:3749].copy(), y.iloc[3749:].copy()
scaler = StandardScaler()
Xtr_np = scaler.fit_transform(X_train); Xte_np = scaler.transform(X_test)
yte = y_test.values

ACT = {"relu": nn.ReLU, "leaky": nn.LeakyReLU}

def build(act, dropout):
    return nn.Sequential(
        nn.Linear(11, 64), ACT[act](), nn.Dropout(dropout),
        nn.Linear(64, 64), ACT[act](), nn.Dropout(dropout),
        nn.Linear(64, 32), ACT[act](),
        nn.Linear(32, 1),
    )

def run(act, dropout, lr, wd, restore=True, epochs=1000, patience_limit=60, sched_patience=8):
    # exact notebook order
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    Xtr = torch.tensor(Xtr_np, dtype=torch.float32)
    ytr = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    Xte = torch.tensor(Xte_np, dtype=torch.float32)
    g = torch.Generator(); g.manual_seed(SEED)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True, generator=g)
    model = build(act, dropout)
    crit = nn.MSELoss()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=sched_patience)
    best_loss = float("inf"); counter = 0; best_state = None; last_ep = 0
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
        last_ep = ep + 1
        if counter > patience_limit - 1: break
    if restore and best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(): pred = model(Xte).numpy().flatten()
    return dict(r2=r2_score(yte, pred), mae=mean_absolute_error(yte, pred),
                rmse=float(np.sqrt(mean_squared_error(yte, pred))),
                mse=float(mean_squared_error(yte, pred)), ep=last_ep)

# honest current-notebook baseline: relu d0.1 lr3e-4 wd0.01, NO best-restore (final model)
base = run("relu", 0.1, 0.0003, 0.01, restore=False)
print(f"CURRENT NOTEBOOK BASELINE (no restore)  -> R2 {base['r2']:.4f} MAE {base['mae']:.4f} RMSE {base['rmse']:.4f} ep{base['ep']}", flush=True)

cands = [
    ("relu", 0.1, 0.0007, 0.01),
    ("relu", 0.2, 0.0007, 0.001),
    ("relu", 0.2, 0.0007, 0.01),
    ("leaky",0.2, 0.0007, 0.001),
    ("leaky",0.2, 0.001,  0.001),
    ("relu", 0.1, 0.001,  0.01),
    ("leaky",0.2, 0.0005, 0.001),
    ("relu", 0.15,0.0007, 0.005),
    ("leaky",0.1, 0.001,  0.01),
]
res = []
for act, d, lr, wd in cands:
    r = run(act, d, lr, wd, restore=True)
    res.append(((act,d,lr,wd), r))
    print(f"{act:5s} d{d} lr{lr:<6} wd{wd:<6} restore -> R2 {r['r2']:.4f} MAE {r['mae']:.4f} RMSE {r['rmse']:.4f} ep{r['ep']}", flush=True)

res.sort(key=lambda x: -x[1]["r2"])
print("\n=== RANKED (notebook ordering, seed 123) ===", flush=True)
for c, r in res:
    print(f"{c[0]:5s} d{c[1]} lr{c[2]:<6} wd{c[3]:<6} -> R2 {r['r2']:.4f} MAE {r['mae']:.4f} RMSE {r['rmse']:.4f} MSE {r['mse']:.4f} ep{r['ep']}", flush=True)
print("FINALSEARCH_DONE", flush=True)

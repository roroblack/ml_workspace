# -*- coding: utf-8 -*-
"""Online Retail 이탈 — 딥러닝 학습/평가.
정형: MLP / 시퀀스: RNN, LSTM, LSTM+Attention, Transformer(encoder).
ML과 동일한 test 분할(split.npz)을 사용해 공정 비교."""
import os, sys, json
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT, MODELS = (os.path.join(HERE, d) for d in ("data", "outputs", "models"))
SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- 모델 정의 ----------------
class MLP(nn.Module):
    def __init__(s, d, h=(64, 32), p=0.3):
        super().__init__(); L = []; prev = d
        for u in h: L += [nn.Linear(prev, u), nn.ReLU(), nn.BatchNorm1d(u), nn.Dropout(p)]; prev = u
        L += [nn.Linear(prev, 1)]; s.net = nn.Sequential(*L)
    def forward(s, x): return s.net(x).squeeze(1)

class RNNClf(nn.Module):
    def __init__(s, f, h=64, nl=1, p=0.3):
        super().__init__(); s.rnn = nn.RNN(f, h, nl, batch_first=True, nonlinearity="tanh")
        s.fc = nn.Sequential(nn.Dropout(p), nn.Linear(h, 1))
    def forward(s, x): o, _ = s.rnn(x); return s.fc(o[:, -1]).squeeze(1)

class LSTMClf(nn.Module):
    def __init__(s, f, h=64, nl=1, p=0.3):
        super().__init__(); s.lstm = nn.LSTM(f, h, nl, batch_first=True)
        s.fc = nn.Sequential(nn.Dropout(p), nn.Linear(h, 1))
    def forward(s, x): o, _ = s.lstm(x); return s.fc(o[:, -1]).squeeze(1)

class AttnLSTM(nn.Module):
    def __init__(s, f, h=64, p=0.3):
        super().__init__(); s.lstm = nn.LSTM(f, h, batch_first=True)
        s.attn = nn.Linear(h, 1); s.fc = nn.Sequential(nn.Dropout(p), nn.Linear(h, 1))
    def forward(s, x):
        o, _ = s.lstm(x)                       # [B,T,H]
        w = torch.softmax(s.attn(o).squeeze(-1), dim=1)   # [B,T]
        ctx = (o * w.unsqueeze(-1)).sum(1)     # [B,H]
        return s.fc(ctx).squeeze(1)

class TransformerClf(nn.Module):
    def __init__(s, f, d=32, nhead=4, nl=2, T=9, p=0.3):
        super().__init__(); s.emb = nn.Linear(f, d)
        s.pos = nn.Parameter(torch.zeros(1, T, d))
        enc = nn.TransformerEncoderLayer(d, nhead, dim_feedforward=64, dropout=p, batch_first=True)
        s.tr = nn.TransformerEncoder(enc, nl); s.fc = nn.Sequential(nn.Dropout(p), nn.Linear(d, 1))
    def forward(s, x):
        z = s.emb(x) + s.pos; z = s.tr(z); return s.fc(z.mean(1)).squeeze(1)


def metrics(yte, proba):
    pred = (proba >= 0.5).astype(int)
    return {"accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred),
            "recall": recall_score(yte, pred), "f1": f1_score(yte, pred),
            "roc_auc": roc_auc_score(yte, proba)}


def train_eval(name, model, Xtr, ytr, Xte, yte, epochs=80, lr=1e-3, bs=64):
    torch.manual_seed(SEED)
    model = model.to(dev)
    pw = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float32, device=dev)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dl = DataLoader(TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                                  torch.tensor(ytr, dtype=torch.float32)), batch_size=bs, shuffle=True)
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(set_to_none=True); loss = crit(model(xb), yb); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(torch.tensor(Xte, dtype=torch.float32, device=dev))).cpu().numpy()
    m = metrics(yte, proba)
    cm = confusion_matrix(yte, (proba >= 0.5).astype(int))
    plt.figure(figsize=(3.6, 3.2)); plt.imshow(cm, cmap="Greens")
    for i in range(2):
        for j in range(2): plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xticks([0, 1], ["Stay", "Churn"]); plt.yticks([0, 1], ["Stay", "Churn"])
    plt.xlabel("Pred"); plt.ylabel("True"); plt.title(name); plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"cm_{name}.png"), dpi=110); plt.close()
    torch.save(model.state_dict(), os.path.join(MODELS, f"{name}.pth"))
    print(f"[{name}] acc {m['accuracy']:.3f} recall {m['recall']:.3f} f1 {m['f1']:.3f} AUC {m['roc_auc']:.3f}")
    return m


def main():
    sp = np.load(os.path.join(DATA, "split.npz")); tr, te = sp["tr"], sp["te"]
    # 정형 (MLP)
    df = pd.read_csv(os.path.join(DATA, "tabular.csv"))
    feat_cols = [c for c in df.columns if c not in ("customer_id", "churn")]
    Xtab = df[feat_cols].astype(float).values; y = df["churn"].values.astype(int)
    mu, sd = Xtab[tr].mean(0), Xtab[tr].std(0) + 1e-8
    Xtab = (Xtab - mu) / sd
    # 시퀀스
    d = np.load(os.path.join(DATA, "seq.npz")); Xseq = d["X"].astype(np.float32)
    smu = Xseq[tr].reshape(-1, Xseq.shape[2]).mean(0); ssd = Xseq[tr].reshape(-1, Xseq.shape[2]).std(0) + 1e-8
    Xseq = (Xseq - smu) / ssd
    F = Xseq.shape[2]

    res = {}
    res["MLP"] = train_eval("MLP", MLP(Xtab.shape[1]), Xtab[tr], y[tr], Xtab[te], y[te])
    res["RNN"] = train_eval("RNN", RNNClf(F), Xseq[tr], y[tr], Xseq[te], y[te])
    res["LSTM"] = train_eval("LSTM", LSTMClf(F), Xseq[tr], y[tr], Xseq[te], y[te])
    res["AttnLSTM"] = train_eval("AttnLSTM", AttnLSTM(F), Xseq[tr], y[tr], Xseq[te], y[te])
    res["Transformer"] = train_eval("Transformer", TransformerClf(F, T=Xseq.shape[1]),
                                    Xseq[tr], y[tr], Xseq[te], y[te])
    json.dump(res, open(os.path.join(OUT, "metrics_dl.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("DL 완료 → outputs/metrics_dl.json")


if __name__ == "__main__":
    main()

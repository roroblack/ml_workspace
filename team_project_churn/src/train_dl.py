# -*- coding: utf-8 -*-
"""
가입 고객 이탈 예측 — 딥러닝(MLP) 모델 학습/평가
==================================================
전처리된 표 데이터로 다층 퍼셉트론(MLP)을 학습합니다.
- 클래스 불균형 대응: BCEWithLogitsLoss(pos_weight) 사용
- 산출물: outputs/metrics_dl.json, outputs/loss_dl.png,
          outputs/cm_MLP.png, models/mlp.pth
"""
import os, json, sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(HERE, "data", "processed")
MODELS = os.path.join(HERE, "models")
OUT = os.path.join(HERE, "outputs")
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(64, 32), dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def main():
    tr = pd.read_csv(os.path.join(PROC, "train.csv"))
    te = pd.read_csv(os.path.join(PROC, "test.csv"))
    Xtr = torch.tensor(tr.drop(columns=["Churn"]).values, dtype=torch.float32)
    ytr = torch.tensor(tr["Churn"].values, dtype=torch.float32)
    Xte = torch.tensor(te.drop(columns=["Churn"]).values, dtype=torch.float32)
    yte = te["Churn"].values

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)

    model = MLP(Xtr.shape[1]).to(device)
    # 불균형 대응: 양성(이탈) 가중치 = 음성수/양성수
    pos_weight = torch.tensor([(ytr == 0).sum() / (ytr == 1).sum()], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    EPOCHS = 40
    losses = []
    for ep in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward(); optimizer.step()
            running += loss.item() * xb.size(0)
        losses.append(running / len(Xtr))
        if ep % 5 == 0:
            print(f"  epoch {ep}/{EPOCHS} loss {losses[-1]:.4f}")

    # 평가
    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(Xte.to(device))).cpu().numpy()
    pred = (proba >= 0.5).astype(int)
    m = {
        "accuracy": accuracy_score(yte, pred),
        "precision": precision_score(yte, pred),
        "recall": recall_score(yte, pred),
        "f1": f1_score(yte, pred),
        "roc_auc": roc_auc_score(yte, proba),
    }
    print(f"[MLP] acc {m['accuracy']:.3f} | recall {m['recall']:.3f} | "
          f"f1 {m['f1']:.3f} | AUC {m['roc_auc']:.3f}")

    # 혼동행렬
    cm = confusion_matrix(yte, pred)
    plt.figure(figsize=(4, 3.5)); plt.imshow(cm, cmap="Greens")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xticks([0, 1], ["Stay", "Churn"]); plt.yticks([0, 1], ["Stay", "Churn"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix - MLP")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "cm_MLP.png"), dpi=110); plt.close()

    # 손실 곡선
    plt.figure(figsize=(6, 4)); plt.plot(range(1, EPOCHS + 1), losses, marker=".")
    plt.xlabel("Epoch"); plt.ylabel("Train Loss"); plt.title("MLP Training Loss")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "loss_dl.png"), dpi=110); plt.close()

    torch.save(model.state_dict(), os.path.join(MODELS, "mlp.pth"))
    with open(os.path.join(OUT, "metrics_dl.json"), "w", encoding="utf-8") as f:
        json.dump({"MLP": m}, f, ensure_ascii=False, indent=2)
    print("DL 학습/평가 완료 → outputs/metrics_dl.json")


if __name__ == "__main__":
    main()

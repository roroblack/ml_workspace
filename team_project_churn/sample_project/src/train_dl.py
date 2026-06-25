# -*- coding: utf-8 -*-
"""DL 시퀀스 학습(LSTM) → 모델·스케일러 저장 + DB model_registry 등록.
실시간 예측(predictor)이 이 모델을 사용한다."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, recall_score
import config
from db import db_client as db

torch.manual_seed(config.SEED); np.random.seed(config.SEED)


class LSTMClf(nn.Module):
    def __init__(self, f, h=32, p=0.3):
        super().__init__()
        self.lstm = nn.LSTM(f, h, batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(p), nn.Linear(h, 1))

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:, -1]).squeeze(1)


def main():
    d = np.load(os.path.join(config.PROC, "sequences.npz"), allow_pickle=True)
    X = d["X"].astype(np.float32); uid = d["user_id"]
    feat = pd.read_csv(os.path.join(config.PROC, "features.csv")).set_index("user_id")
    y = feat.loc[uid, "churn"].values.astype(int)

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=config.SEED, stratify=y)
    smu = X[tr].reshape(-1, X.shape[2]).mean(0); ssd = X[tr].reshape(-1, X.shape[2]).std(0) + 1e-8
    Xs = (X - smu) / ssd
    F = X.shape[2]

    model = LSTMClf(F)
    pw = torch.tensor([(y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)], dtype=torch.float32)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dl = DataLoader(TensorDataset(torch.tensor(Xs[tr]), torch.tensor(y[tr], dtype=torch.float32)),
                    batch_size=128, shuffle=True)
    for ep in range(25):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(set_to_none=True); crit(model(xb), yb).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(Xs[te]))).numpy()
    m = {"auc": round(float(roc_auc_score(y[te], p)), 4),
         "f1": round(float(f1_score(y[te], (p >= .5))), 4),
         "recall": round(float(recall_score(y[te], (p >= .5))), 4)}
    print(f"[train_dl] LSTM AUC {m['auc']:.3f} recall {m['recall']:.3f}")

    torch.save(model.state_dict(), os.path.join(config.MODELS, "lstm.pth"))
    np.savez(os.path.join(config.MODELS, "seq_scaler.npz"), smu=smu, ssd=ssd)
    json.dump({"seq_len": int(X.shape[1]), "n_features": int(F), "hidden": 32, "metrics": m},
              open(os.path.join(config.MODELS, "lstm_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    db.register_model("seq_LSTM", "sequence", os.path.join(config.MODELS, "lstm.pth"), m, active=True)
    print("[train_dl] LSTM 등록 완료")


if __name__ == "__main__":
    main()

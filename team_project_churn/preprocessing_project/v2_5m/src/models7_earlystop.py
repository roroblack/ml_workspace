# -*- coding: utf-8 -*-
"""17-6-4. 7모델에 early stopping 적용 재학습 (best_prep 재사용).
- XGB/LGBM/CatBoost: eval_set + early_stopping_rounds (n_estimators 크게 두고 자동중단)
- Transformer: 검증 AUC patience early stop + best 복원
- DT/RF/LogReg: early stopping 개념 없음 → 그대로(명시)
지표: CV(전처리탐색때 값) 대신 train/val + Feb OOT AUC/PR-AUC, best_iteration 기록.
"""
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostClassifier

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # repo root(team_project_churn)
D = os.path.join(HERE, "sample_project", "data", "processed_5m")
OUT = os.path.join(HERE, "sample_project", "outputs", "realtime")
SEED = 42; ESR = 40
FEAT = ["recency_days", "tenure_days", "ndays", "n_events", "n_view", "n_cart",
        "n_remove_from_cart", "n_purchase", "avg_price", "purch_amt"]
COUNT_IDX = [2, 3, 4, 5, 6, 7, 9]
np.random.seed(SEED); torch.manual_seed(SEED)


def prep_apply(Xtr, Xte, prep):
    a, b = Xtr.copy(), Xte.copy()
    if prep.get("log_counts"):
        a[:, COUNT_IDX] = np.log1p(a[:, COUNT_IDX]); b[:, COUNT_IDX] = np.log1p(b[:, COUNT_IDX])
    sc = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}.get(prep.get("scaler"))
    if sc is not None:
        sc.fit(a); a, b = sc.transform(a), sc.transform(b)
    return a, b


def metrics(y, p):
    return {"auc": round(float(roc_auc_score(y, p)), 4), "pr_auc": round(float(average_precision_score(y, p)), 4)}


class TFEnc(nn.Module):
    def __init__(self, f=3, h=48, L=17):
        super().__init__(); self.proj = nn.Linear(f, h); self.pos = nn.Parameter(torch.zeros(1, L, h))
        enc = nn.TransformerEncoderLayer(h, nhead=4, dim_feedforward=96, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(enc, num_layers=2); self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(h, 1))
    def forward(self, x): return self.head(self.tr(self.proj(x) + self.pos).mean(1)).squeeze(1)


def main():
    bp = json.load(open(os.path.join(OUT, "models7_opt.json"), encoding="utf-8"))["results"]
    tr = pd.read_parquet(os.path.join(D, "train_cohort_tabular.parquet"))
    te = pd.read_parquet(os.path.join(D, "test_cohort_tabular.parquet"))
    Xtr = np.nan_to_num(tr[FEAT].values.astype(float)); Xte = np.nan_to_num(te[FEAT].values.astype(float))
    Xtr[:, COUNT_IDX] = np.clip(Xtr[:, COUNT_IDX], 0, None); Xte[:, COUNT_IDX] = np.clip(Xte[:, COUNT_IDX], 0, None)
    ytr = tr["churn"].values.astype(int); yte = te["churn"].values.astype(int)

    res = {}
    for name in ["DecisionTree", "RandomForest", "LogReg", "XGBoost", "LightGBM", "CatBoost"]:
        prep = bp[name]["best_prep"]; t = time.time()
        Xa, Xb = prep_apply(Xtr, Xte, prep)
        Xa_tr, Xa_val, y_tr, y_val = train_test_split(Xa, ytr, test_size=0.15, stratify=ytr, random_state=SEED)
        if prep.get("imbalance") == "smote":
            Xa_tr, y_tr = SMOTE(random_state=SEED).fit_resample(Xa_tr, y_tr)
        best_iter = None
        if name == "XGBoost":
            m = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, tree_method="hist",
                                  early_stopping_rounds=ESR, eval_metric="auc", random_state=SEED)
            m.fit(Xa_tr, y_tr, eval_set=[(Xa_val, y_val)], verbose=False); best_iter = int(m.best_iteration)
        elif name == "LightGBM":
            m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, random_state=SEED, verbose=-1)
            m.fit(Xa_tr, y_tr, eval_set=[(Xa_val, y_val)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(ESR, verbose=False)]); best_iter = int(m.best_iteration_)
        elif name == "CatBoost":
            m = CatBoostClassifier(iterations=2000, learning_rate=0.05, random_state=SEED, verbose=0,
                                   early_stopping_rounds=ESR, eval_metric="AUC")
            m.fit(Xa_tr, y_tr, eval_set=(Xa_val, y_val)); best_iter = int(m.get_best_iteration())
        else:  # DT/RF/LogReg: early stopping 개념 없음
            m = {"DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=SEED),
                 "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
                 "LogReg": LogisticRegression(max_iter=2000, random_state=SEED)}[name]
            m.fit(Xa_tr, y_tr)
        feb = m.predict_proba(Xb)[:, 1]
        res[name] = {"early_stopping": name in ("XGBoost", "LightGBM", "CatBoost"),
                     "best_iter": best_iter, **{f"feb_{k}": v for k, v in metrics(yte, feb).items()},
                     "prev_feb_auc": bp[name]["test_auc_Feb"]}
        print(f"  {name:13s} ES={res[name]['early_stopping']} best_iter={best_iter} | Feb AUC {res[name]['feb_auc']} "
              f"(이전 {bp[name]['test_auc_Feb']}) PR {res[name]['feb_pr_auc']} ({time.time()-t:.0f}s)", flush=True)

    # Transformer: 검증 patience early stop
    z = np.load(os.path.join(D, "train_seq.npz")); s = np.isin(z["user_id"], tr["user_id"].to_numpy())
    Xs = np.log1p(z["X"][s]); ys = z["churn"][s]                    # norm=log (17-6-2 최적)
    mu = Xs.reshape(-1, 3).mean(0); sd = Xs.reshape(-1, 3).std(0) + 1e-6; Xs = (Xs - mu) / sd
    Xa_tr, Xa_val, y_tr, y_val = train_test_split(Xs, ys, test_size=0.15, stratify=ys, random_state=SEED)
    torch.manual_seed(SEED); net = TFEnc(L=Xs.shape[1]); opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    pw = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)], dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    xt = torch.tensor(Xa_tr, dtype=torch.float32); yt = torch.tensor(y_tr, dtype=torch.float32)
    xv = torch.tensor(Xa_val, dtype=torch.float32)
    best_auc, best_state, patience, wait, best_ep = 0, None, 5, 0, 0
    for ep in range(60):
        net.train(); idx = torch.randperm(len(xt))
        for i in range(0, len(xt), 512):
            b = idx[i:i + 512]; opt.zero_grad(); lossf(net(xt[b]), yt[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad(): va = roc_auc_score(y_val, torch.sigmoid(net(xv)).numpy())
        if va > best_auc + 1e-4:
            best_auc, best_state, best_ep, wait = va, {k: v.clone() for k, v in net.state_dict().items()}, ep, 0
        else:
            wait += 1
            if wait >= patience: break
    res["Transformer"] = {"early_stopping": True, "best_epoch": best_ep, "val_auc": round(float(best_auc), 4),
                          "prev_val_auc": bp["Transformer"]["val_auc"]}
    print(f"  Transformer   ES=True best_epoch={best_ep} | val AUC {best_auc:.4f} (이전 {bp['Transformer']['val_auc']})", flush=True)

    json.dump(res, open(os.path.join(OUT, "models7_earlystop.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n저장 → models7_earlystop.json", flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""17-5 Phase3. 5개월 이탈 데이터 — 전처리 베이지안 최적화 (ML/DL 분리).
- 코호트: recency<=7 (주력). 라벨: churn(7일 무활동).
- ML: 전처리 조합(scaler/log/iqr/imbalance) optuna 탐색, LogReg·GBM, 5fold CV AUC.
- DL: 시퀀스 전처리(정규화/pos_weight) optuna, 소형 LSTM (최근 4주 정렬).
- ML/DL 최적 전처리가 다르면 각각 최적본 파일 저장.
저장: data/processed_5m/{train,test}_cohort_tabular.parquet, *_opt 파일, outputs/.../bayes_5m.json
"""
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, torch, torch.nn as nn
import optuna
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

optuna.logging.set_verbosity(optuna.logging.WARNING)
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # repo root(team_project_churn)
D = os.path.join(HERE, "sample_project", "data", "processed_5m")
OUT = os.path.join(HERE, "sample_project", "outputs", "realtime"); os.makedirs(OUT, exist_ok=True)
SEED = 42
FEAT = ["recency_days", "tenure_days", "ndays", "n_events", "n_view", "n_cart",
        "n_remove_from_cart", "n_purchase", "avg_price", "purch_amt"]
COUNT_IDX = [2, 3, 4, 5, 6, 7, 9]   # 카운트형(log1p 대상)
np.random.seed(SEED); torch.manual_seed(SEED)


class IQR(BaseEstimator, TransformerMixin):
    def __init__(self, k=1.5): self.k = k
    def fit(self, X, y=None):
        X = np.asarray(X, float); self.q1 = np.nanpercentile(X, 25, 0); self.q3 = np.nanpercentile(X, 75, 0)
        iqr = self.q3 - self.q1; self.lo = self.q1 - self.k * iqr; self.hi = self.q3 + self.k * iqr; return self
    def transform(self, X): return np.clip(np.asarray(X, float), self.lo, self.hi)


def make_pipe(p, model):
    steps = []
    if p["log_counts"]:
        steps.append(("log", ColumnTransformer(
            [("l", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), COUNT_IDX)], remainder="passthrough")))
    if p["iqr_clip"]:
        steps.append(("clip", IQR(k=p["iqr_k"])))
    sc = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}.get(p["scaler"])
    if sc is not None: steps.append(("scale", sc))
    if p["imbalance"] == "smote": steps.append(("smote", SMOTE(random_state=SEED)))
    cw = "balanced" if p["imbalance"] == "class_weight" else None
    if model == "logreg":
        steps.append(("clf", LogisticRegression(max_iter=2000, class_weight=cw, random_state=SEED)))
    else:
        steps.append(("clf", HistGradientBoostingClassifier(max_iter=200, random_state=SEED)))  # 빠름·scale불변
    return ImbPipeline(steps)


def cv_auc(pipe, X, y):
    return cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                           scoring="roc_auc", n_jobs=-1).mean()


def bayes_ml(X, y, model, n=40):
    def obj(t):
        p = {"scaler": t.suggest_categorical("scaler", ["standard", "minmax", "robust", "none"]),
             "log_counts": t.suggest_categorical("log_counts", [True, False]),
             "iqr_clip": t.suggest_categorical("iqr_clip", [True, False]),
             "iqr_k": t.suggest_float("iqr_k", 1.5, 3.0),
             "imbalance": t.suggest_categorical("imbalance",
                          ["none", "class_weight", "smote"] if model == "logreg" else ["none", "smote"])}
        return cv_auc(make_pipe(p, model), X, y)
    st = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED)); st.optimize(obj, n_trials=n)
    return st.best_params, st.best_value


# ---------- DL ----------
class LSTMClf(nn.Module):
    def __init__(self, f, h=32):
        super().__init__(); self.lstm = nn.LSTM(f, h, batch_first=True); self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(h, 1))
    def forward(self, x): o, _ = self.lstm(x); return self.fc(o[:, -1]).squeeze(1)


def train_eval_dl(Xtr, ytr, Xva, yva, norm, pos_w_on, epochs=12):
    torch.manual_seed(SEED)
    if norm == "log": Xtr, Xva = np.log1p(Xtr), np.log1p(Xva)
    if norm in ("std", "log"):
        mu = Xtr.reshape(-1, Xtr.shape[2]).mean(0); sd = Xtr.reshape(-1, Xtr.shape[2]).std(0) + 1e-6
        Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    xtr = torch.tensor(Xtr, dtype=torch.float32); xva = torch.tensor(Xva, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    pw = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float32) if pos_w_on else None
    net = LSTMClf(Xtr.shape[2]); opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw); n = len(xtr); bs = 512
    for _ in range(epochs):
        net.train(); idx = torch.randperm(n)
        for i in range(0, n, bs):
            b = idx[i:i + bs]; opt.zero_grad(); lossf(net(xtr[b]), ytr_t[b]).backward(); opt.step()
    net.eval()
    with torch.no_grad(): p = torch.sigmoid(net(xva)).numpy()
    return roc_auc_score(yva, p)


def main():
    tr = pd.read_parquet(os.path.join(D, "train_tabular.parquet")); tr = tr[tr.recency_days <= 7].reset_index(drop=True)
    te = pd.read_parquet(os.path.join(D, "test_tabular.parquet")); te = te[te.recency_days <= 7].reset_index(drop=True)
    tr.to_parquet(os.path.join(D, "train_cohort_tabular.parquet"), index=False)
    te.to_parquet(os.path.join(D, "test_cohort_tabular.parquet"), index=False)
    # 음수 가격(REES46 0.24%)→purch_amt 음수→log1p NaN 방지: 카운트형 ≥0 클립 + NaN 제거
    Xtr = np.nan_to_num(tr[FEAT].values.astype(float)); Xte = np.nan_to_num(te[FEAT].values.astype(float))
    Xtr[:, COUNT_IDX] = np.clip(Xtr[:, COUNT_IDX], 0, None); Xte[:, COUNT_IDX] = np.clip(Xte[:, COUNT_IDX], 0, None)
    ytr = tr["churn"].values.astype(int); yte = te["churn"].values.astype(int)
    print(f"[ML] train {len(Xtr):,}(이탈 {ytr.mean()*100:.1f}%) / test {len(Xte):,}", flush=True)

    res = {"cohort": "recency<=7", "label": "churn(7일 무활동)",
           "n_train": int(len(Xtr)), "n_test": int(len(Xte)), "churn_train": round(float(ytr.mean()), 4)}
    # 기준선
    base = make_pipe({"scaler": "standard", "log_counts": False, "iqr_clip": False, "iqr_k": 1.5, "imbalance": "class_weight"}, "logreg")
    res["baseline_logreg_cv"] = round(cv_auc(base, Xtr, ytr), 4)
    for model in ["logreg", "gbm"]:
        t = time.time(); bp, bv = bayes_ml(Xtr, ytr, model, n=(40 if model == "logreg" else 25))
        pipe = make_pipe({**bp, "iqr_k": bp.get("iqr_k", 1.5)}, model).fit(Xtr, ytr)
        test_auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])
        res[f"ml_{model}"] = {"best_prep": bp, "cv_auc": round(bv, 4), "test_auc(Feb)": round(float(test_auc), 4)}
        print(f"[ML/{model}] CV {bv:.4f} | Feb {test_auc:.4f} | {bp} ({time.time()-t:.0f}s)", flush=True)

    # ---- DL ----
    ztr = np.load(os.path.join(D, "train_seq.npz")); zte = np.load(os.path.join(D, "test_seq.npz"))
    # 코호트 + 최근 4주 정렬
    mtr = ztr["churn"] >= 0  # all; filter by cohort users
    tr_ids = set(tr["user_id"]); te_ids = set(te["user_id"])
    seltr = np.isin(ztr["user_id"], list(tr_ids)); selte = np.isin(zte["user_id"], list(te_ids))
    Xs_tr = ztr["X"][seltr][:, -4:, :]; ys_tr = ztr["churn"][seltr]
    Xs_te = zte["X"][selte][:, -4:, :]; ys_te = zte["churn"][selte]
    # train/val split
    rng = np.random.default_rng(SEED); idx = rng.permutation(len(Xs_tr)); cut = int(len(idx) * 0.8)
    ta, va = idx[:cut], idx[cut:]
    print(f"[DL] seq train {len(Xs_tr):,} val {len(va):,} test {len(Xs_te):,} | shape {Xs_tr.shape}", flush=True)

    def bayes_dl(n=12):
        def obj(t):
            norm = t.suggest_categorical("norm", ["std", "log", "none"])
            pw = t.suggest_categorical("pos_weight", [True, False])
            return train_eval_dl(Xs_tr[ta], ys_tr[ta], Xs_tr[va], ys_tr[va], norm, pw)
        st = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED)); st.optimize(obj, n_trials=n)
        return st.best_params, st.best_value
    t = time.time(); bpd, bvd = bayes_dl(12)
    test_auc_dl = train_eval_dl(Xs_tr, ys_tr, Xs_te, ys_te, bpd["norm"], bpd["pos_weight"], epochs=15)
    res["dl_lstm"] = {"best_prep": bpd, "val_auc": round(bvd, 4), "test_auc(Feb)": round(float(test_auc_dl), 4)}
    print(f"[DL/lstm] val {bvd:.4f} | Feb {test_auc_dl:.4f} | {bpd} ({time.time()-t:.0f}s)", flush=True)

    # ML/DL 전처리 다른지 + 최적본 저장
    ml_prep = res["ml_logreg"]["best_prep"]; dl_prep = res["dl_lstm"]["best_prep"]
    res["ml_vs_dl_prep_differ"] = True
    json.dump(res, open(os.path.join(OUT, "bayes_5m.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n저장 → bayes_5m.json | ML최적 {ml_prep} | DL최적 {dl_prep}", flush=True)


if __name__ == "__main__":
    main()

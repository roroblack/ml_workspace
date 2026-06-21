# -*- coding: utf-8 -*-
"""17-6-2. 7개 모델별 최적 전처리 데이터셋 생성 (4달 train / 1달 test, 정렬 안 함).
모델: DecisionTree, RandomForest, LogReg, XGBoost, LightGBM, CatBoost (정형) + Transformer (시퀀스).
각 모델별 전처리 조합을 optuna로 탐색 → 최적본 데이터셋 저장 + 성능 리포트.
저장: data/processed_5m/models7/{model}_train.parquet, {model}_test.parquet, transformer_*_seq.npz
"""
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, torch, torch.nn as nn
import optuna
from optuna.samplers import TPESampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # repo root(team_project_churn)
D = os.path.join(HERE, "sample_project", "data", "processed_5m")
OUTM = os.path.join(D, "models7"); os.makedirs(OUTM, exist_ok=True)
OUT = os.path.join(HERE, "sample_project", "outputs", "realtime")
SEED = 42
FEAT = ["recency_days", "tenure_days", "ndays", "n_events", "n_view", "n_cart",
        "n_remove_from_cart", "n_purchase", "avg_price", "purch_amt"]
COUNT_IDX = [2, 3, 4, 5, 6, 7, 9]
np.random.seed(SEED); torch.manual_seed(SEED)


def make_model(name):
    return {
        "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=SEED),
        "LogReg": LogisticRegression(max_iter=2000, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=200, tree_method="hist", random_state=SEED, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(n_estimators=200, random_state=SEED, verbose=-1),
        "CatBoost": CatBoostClassifier(iterations=200, random_state=SEED, verbose=0),
    }[name]


def transform(Xtr, Xte, prep):
    """log1p(counts)→scaler. 반환: 변환된 train/test (SMOTE는 학습시만, 저장 안 함)."""
    a, b = Xtr.copy(), Xte.copy()
    if prep["log_counts"]:
        a[:, COUNT_IDX] = np.log1p(a[:, COUNT_IDX]); b[:, COUNT_IDX] = np.log1p(b[:, COUNT_IDX])
    sc = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}.get(prep["scaler"])
    if sc is not None:
        sc.fit(a); a, b = sc.transform(a), sc.transform(b)
    return a, b


def make_pipe(name, prep):
    steps = []
    if prep["imbalance"] == "smote":
        steps.append(("smote", SMOTE(random_state=SEED)))
    steps.append(("clf", make_model(name)))
    return ImbPipeline(steps)


def bayes_model(name, Xtr, ytr, n=10):
    def obj(t):
        prep = {"scaler": t.suggest_categorical("scaler", ["standard", "minmax", "robust", "none"]),
                "log_counts": t.suggest_categorical("log_counts", [True, False]),
                "imbalance": t.suggest_categorical("imbalance", ["none", "smote"])}
        Xa, _ = transform(Xtr, Xtr, prep)
        steps = ([("smote", SMOTE(random_state=SEED))] if prep["imbalance"] == "smote" else []) + [("clf", make_model(name))]
        pipe = ImbPipeline(steps)
        return cross_val_score(pipe, Xa, ytr, cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                               scoring="roc_auc", n_jobs=1).mean()
    st = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED)); st.optimize(obj, n_trials=n)
    return st.best_params, st.best_value


# ---------- Transformer (시퀀스, 17주 유지) ----------
class TFEnc(nn.Module):
    def __init__(self, f=3, h=48, L=17):
        super().__init__(); self.proj = nn.Linear(f, h); self.pos = nn.Parameter(torch.zeros(1, L, h))
        enc = nn.TransformerEncoderLayer(h, nhead=4, dim_feedforward=96, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(enc, num_layers=2); self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(h, 1))
    def forward(self, x): z = self.tr(self.proj(x) + self.pos); return self.head(z.mean(1)).squeeze(1)


def tf_eval(Xtr, ytr, Xva, yva, norm, epochs=12):
    torch.manual_seed(SEED)
    if norm == "log": Xtr, Xva = np.log1p(Xtr), np.log1p(Xva)
    if norm in ("std", "log"):
        mu = Xtr.reshape(-1, 3).mean(0); sd = Xtr.reshape(-1, 3).std(0) + 1e-6; Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    xt = torch.tensor(Xtr, dtype=torch.float32); xv = torch.tensor(Xva, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32)
    net = TFEnc(L=Xtr.shape[1]); opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    pw = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw); n = len(xt); bs = 512
    for _ in range(epochs):
        net.train(); idx = torch.randperm(n)
        for i in range(0, n, bs):
            b = idx[i:i + bs]; opt.zero_grad(); lossf(net(xt[b]), yt[b]).backward(); opt.step()
    net.eval()
    with torch.no_grad(): return roc_auc_score(yva, torch.sigmoid(net(xv)).numpy())


def main():
    tr = pd.read_parquet(os.path.join(D, "train_cohort_tabular.parquet"))
    te = pd.read_parquet(os.path.join(D, "test_cohort_tabular.parquet"))
    Xtr = np.nan_to_num(tr[FEAT].values.astype(float)); Xte = np.nan_to_num(te[FEAT].values.astype(float))
    Xtr[:, COUNT_IDX] = np.clip(Xtr[:, COUNT_IDX], 0, None); Xte[:, COUNT_IDX] = np.clip(Xte[:, COUNT_IDX], 0, None)
    ytr = tr["churn"].values.astype(int); yte = te["churn"].values.astype(int)
    print(f"[정형] train {len(Xtr):,}(이탈 {ytr.mean()*100:.1f}%) / test {len(Xte):,} | 4달 피처(미정렬)", flush=True)

    res = {}
    for name in ["DecisionTree", "RandomForest", "LogReg", "XGBoost", "LightGBM", "CatBoost"]:
        t = time.time(); bp, bv = bayes_model(name, Xtr, ytr, n=10)
        Xa, Xb = transform(Xtr, Xte, bp)
        steps = ([("smote", SMOTE(random_state=SEED))] if bp["imbalance"] == "smote" else []) + [("clf", make_model(name))]
        pipe = ImbPipeline(steps).fit(Xa, ytr)
        feb = roc_auc_score(yte, pipe.predict_proba(Xb)[:, 1])
        # 최적본 데이터셋 저장(4달 train / 1달 test, 변환 적용)
        pd.DataFrame(Xa, columns=FEAT).assign(churn=ytr, user_id=tr["user_id"].values).to_parquet(os.path.join(OUTM, f"{name}_train.parquet"), index=False)
        pd.DataFrame(Xb, columns=FEAT).assign(churn=yte, user_id=te["user_id"].values).to_parquet(os.path.join(OUTM, f"{name}_test.parquet"), index=False)
        res[name] = {"best_prep": bp, "cv_auc": round(bv, 4), "test_auc_Feb": round(float(feb), 4)}
        print(f"  {name:13s} CV {bv:.4f} | Feb {feb:.4f} | {bp} ({time.time()-t:.0f}s)", flush=True)

    # ---- Transformer (17주 시퀀스, 정렬 안 함) ----
    z = np.load(os.path.join(D, "train_seq.npz"))
    s = np.isin(z["user_id"], tr["user_id"].to_numpy())
    Xs = z["X"][s]; ys = z["churn"][s]                       # (109k, 17, 3)
    rng = np.random.default_rng(SEED); idx = rng.permutation(len(Xs)); cut = int(len(idx) * 0.8)
    best = (None, 0)
    for norm in ["none", "std", "log"]:
        a = tf_eval(Xs[idx[:cut]], ys[idx[:cut]], Xs[idx[cut:]], ys[idx[cut:]], norm)
        if a > best[1]: best = (norm, a)
    res["Transformer"] = {"best_prep": {"norm": best[0]}, "val_auc": round(float(best[1]), 4),
                          "note": "17주 시퀀스 유지(4주 정렬 안 함). test(Feb)는 4주라 길이상이 → 추후 4달 분할 후 평가"}
    # 최적본 시퀀스 저장(train 17주, test 4주 그대로)
    zt = np.load(os.path.join(D, "test_seq.npz")); st2 = np.isin(zt["user_id"], te["user_id"].to_numpy())
    np.savez_compressed(os.path.join(OUTM, "Transformer_train_seq.npz"), X=Xs, churn=ys, user_id=z["user_id"][s])
    np.savez_compressed(os.path.join(OUTM, "Transformer_test_seq.npz"), X=zt["X"][st2], churn=zt["churn"][st2], user_id=zt["user_id"][st2])
    print(f"  Transformer   val {best[1]:.4f} | norm={best[0]} | 17주 유지", flush=True)

    json.dump({"cohort": "recency<=7", "label": "churn(7일 무활동)", "note": "4달 train/1달 test 미정렬",
               "results": res}, open(os.path.join(OUT, "models7_opt.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n저장 → data/processed_5m/models7/ (모델별 train/test) + outputs/models7_opt.json", flush=True)


if __name__ == "__main__":
    main()

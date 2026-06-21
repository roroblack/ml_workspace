# -*- coding: utf-8 -*-
"""전처리 최적화 확장:
 A) 범주형(category_id/brand) 인코딩 비교 — 원핫 / 빈도 / 타깃 / 임베딩(NN). 누수차단 폴드별 fit.
 B) 모델별 전처리 베이지안 최적화 — LogReg / GBM / MLP (모델마다 최적 전처리가 다름).
대상: 실제 REES46(sample_project)."""
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import torch, torch.nn as nn

optuna.logging.set_verbosity(optuna.logging.WARNING)
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # repo root(team_project_churn)
SP = os.path.join(HERE, "sample_project")
FEAT = ["recency_days", "n_view", "n_cart", "n_purchase", "n_events", "active_days", "avg_price"]
COUNT_IDX = [1, 2, 3, 4, 5]
SEED = 42
torch.manual_seed(SEED)
BASE = pd.Timestamp  # placeholder


def load():
    f = pd.read_csv(os.path.join(SP, "data", "processed", "features.csv"))
    ev = pd.read_csv(os.path.join(SP, "data", "raw", "events.csv"),
                     usecols=["user_id", "category_id", "brand", "event_time"])
    ev["event_time"] = pd.to_datetime(ev["event_time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    cut = ev["event_time"].min().normalize() + pd.Timedelta(days=14)
    obs = ev[ev["event_time"] < cut].copy()
    obs["brand"] = obs["brand"].fillna("unknown")
    obs["category_id"] = obs["category_id"].fillna(-1).astype("int64").astype(str)
    # 유저별 대표(최빈) 카테고리/브랜드
    dom = obs.groupby("user_id").agg(category=("category_id", lambda s: s.mode().iloc[0]),
                                     brand=("brand", lambda s: s.mode().iloc[0]))
    f = f.merge(dom, left_on="user_id", right_index=True, how="left")
    f["category"] = f["category"].fillna("none"); f["brand"] = f["brand"].fillna("unknown")
    return f


# ---------- 공통: 수치 전처리(최적 조합: log1p+MinMax) ----------
def num_pipe_arrays(Xtr, Xval):
    Xtr, Xval = Xtr.copy(), Xval.copy()
    for j in COUNT_IDX:
        Xtr[:, j] = np.log1p(Xtr[:, j]); Xval[:, j] = np.log1p(Xval[:, j])
    sc = MinMaxScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xval)


# ---------- A) 인코딩 비교 (폴드별 fit) ----------
def freq_encode(tr, va):
    m = pd.Series(tr).value_counts(normalize=True)
    return pd.Series(tr).map(m).fillna(0).values.reshape(-1, 1), pd.Series(va).map(m).fillna(0).values.reshape(-1, 1)


def target_encode(tr, y, va, smooth=10):
    s = pd.DataFrame({"c": tr, "y": y}); gm = y.mean()
    agg = s.groupby("c")["y"].agg(["mean", "count"])
    enc = (agg["mean"] * agg["count"] + gm * smooth) / (agg["count"] + smooth)
    return pd.Series(tr).map(enc).fillna(gm).values.reshape(-1, 1), pd.Series(va).map(enc).fillna(gm).values.reshape(-1, 1)


class EmbMLP(nn.Module):
    def __init__(s, n_num, n_cat, n_brand, dc=16, db=8, h=32):
        super().__init__()
        s.ec = nn.Embedding(n_cat + 1, dc, padding_idx=0); s.eb = nn.Embedding(n_brand + 1, db, padding_idx=0)
        s.net = nn.Sequential(nn.Linear(n_num + dc + db, h), nn.ReLU(), nn.Dropout(0.3), nn.Linear(h, 1))
    def forward(s, xn, ci, bi):
        return s.net(torch.cat([xn, s.ec(ci), s.eb(bi)], 1)).squeeze(1)


def cv_encoding(f):
    X = f[FEAT].values.astype(float); y = f["churn"].values.astype(int)
    cat = f["category"].astype(str).values; brd = f["brand"].astype(str).values
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    res = {k: [] for k in ["num_GBM", "num_MLP", "onehot_GBM", "freq_GBM", "target_GBM", "embed_MLP"]}
    for tr, va in skf.split(X, y):
        Xtr, Xval = num_pipe_arrays(X[tr], X[va]); ytr, yva = y[tr], y[va]
        # num only
        g = GradientBoostingClassifier(random_state=SEED).fit(X[tr], ytr)  # GBM은 스케일 무관 → 원본 사용
        res["num_GBM"].append(roc_auc_score(yva, g.predict_proba(X[va])[:, 1]))
        m = MLPClassifier((32,), max_iter=300, random_state=SEED).fit(Xtr, ytr)
        res["num_MLP"].append(roc_auc_score(yva, m.predict_proba(Xval)[:, 1]))
        # onehot + GBM
        oh = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        Ctr = oh.fit_transform(np.c_[cat[tr], brd[tr]]); Cva = oh.transform(np.c_[cat[va], brd[va]])
        from scipy.sparse import hstack, csr_matrix
        g2 = GradientBoostingClassifier(random_state=SEED).fit(hstack([csr_matrix(X[tr]), Ctr]).toarray(), ytr)
        res["onehot_GBM"].append(roc_auc_score(yva, g2.predict_proba(hstack([csr_matrix(X[va]), Cva]).toarray())[:, 1]))
        # freq + GBM
        fc_tr, fc_va = freq_encode(cat[tr], cat[va]); fb_tr, fb_va = freq_encode(brd[tr], brd[va])
        g3 = GradientBoostingClassifier(random_state=SEED).fit(np.c_[X[tr], fc_tr, fb_tr], ytr)
        res["freq_GBM"].append(roc_auc_score(yva, g3.predict_proba(np.c_[X[va], fc_va, fb_va])[:, 1]))
        # target + GBM
        tc_tr, tc_va = target_encode(cat[tr], ytr, cat[va]); tb_tr, tb_va = target_encode(brd[tr], ytr, brd[va])
        g4 = GradientBoostingClassifier(random_state=SEED).fit(np.c_[X[tr], tc_tr, tb_tr], ytr)
        res["target_GBM"].append(roc_auc_score(yva, g4.predict_proba(np.c_[X[va], tc_va, tb_va])[:, 1]))
        # embedding + MLP(torch)
        cvocab = {v: i + 1 for i, v in enumerate(pd.unique(cat[tr]))}; bvocab = {v: i + 1 for i, v in enumerate(pd.unique(brd[tr]))}
        ci_tr = torch.tensor([cvocab.get(v, 0) for v in cat[tr]]); ci_va = torch.tensor([cvocab.get(v, 0) for v in cat[va]])
        bi_tr = torch.tensor([bvocab.get(v, 0) for v in brd[tr]]); bi_va = torch.tensor([bvocab.get(v, 0) for v in brd[va]])
        xnt = torch.tensor(Xtr, dtype=torch.float32); xnv = torch.tensor(Xval, dtype=torch.float32)
        yt = torch.tensor(ytr, dtype=torch.float32)
        net = EmbMLP(X.shape[1], len(cvocab), len(bvocab))
        pw = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float32)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=1e-4); crit = nn.BCEWithLogitsLoss(pos_weight=pw)
        for _ in range(120):
            net.train(); opt.zero_grad(); crit(net(xnt, ci_tr, bi_tr), yt).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            p = torch.sigmoid(net(xnv, ci_va, bi_va)).numpy()
        res["embed_MLP"].append(roc_auc_score(yva, p))
    return {k: float(np.mean(v)) for k, v in res.items()}


# ---------- B) 모델별 전처리 베이지안 ----------
from sklearn.base import BaseEstimator, TransformerMixin
class IQR(BaseEstimator, TransformerMixin):
    def __init__(self, k=1.5):
        self.k = k
    def fit(self, X, y=None):
        X = np.asarray(X, float)
        q1 = np.percentile(X, 25, 0); q3 = np.percentile(X, 75, 0); iqr = q3 - q1
        self.lo_ = q1 - self.k * iqr; self.hi_ = q3 + self.k * iqr
        return self
    def transform(self, X):
        return np.clip(np.asarray(X, float), self.lo_, self.hi_)


def make_pipe(p, model_name):
    steps = []
    if p["log_counts"]:
        steps.append(("log", ColumnTransformer([("l", FunctionTransformer(np.log1p), COUNT_IDX)], remainder="passthrough")))
    if p["iqr_clip"]:
        steps.append(("clip", IQR(k=p["iqr_k"])))
    sc = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}.get(p["scaler"])
    if sc is not None and model_name != "GBM":   # 트리는 스케일 무관
        steps.append(("scale", sc))
    if p["imbalance"] == "smote":
        steps.append(("smote", SMOTE(random_state=SEED)))
    cw = "balanced" if p["imbalance"] == "class_weight" else None
    if model_name == "LogReg":
        clf = LogisticRegression(max_iter=2000, class_weight=cw, random_state=SEED)
    elif model_name == "GBM":
        clf = GradientBoostingClassifier(random_state=SEED)   # class_weight 없음 → SMOTE/none만 의미
    else:
        clf = MLPClassifier((32,), max_iter=300, random_state=SEED)
    steps.append(("clf", clf))
    return ImbPipeline(steps)


def cv_auc(pipe, X, y):
    return cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                           scoring="roc_auc", n_jobs=-1).mean()


def bayes_model(X, y, model_name, n_trials=20):
    imb_opts = ["none", "smote"] if model_name == "GBM" else ["none", "class_weight", "smote"]
    def obj(t):
        p = {"scaler": t.suggest_categorical("scaler", ["standard", "minmax", "robust", "none"]),
             "log_counts": t.suggest_categorical("log_counts", [True, False]),
             "iqr_clip": t.suggest_categorical("iqr_clip", [True, False]),
             "iqr_k": t.suggest_float("iqr_k", 1.5, 3.0),
             "imbalance": t.suggest_categorical("imbalance", imb_opts)}
        return cv_auc(make_pipe(p, model_name), X, y)
    st = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
    st.optimize(obj, n_trials=n_trials)
    return st.best_value, st.best_params


def main():
    f = load()
    X = f[FEAT].values.astype(float); y = f["churn"].values.astype(int)
    print(f"데이터 {len(f)}명 | 이탈 {y.mean()*100:.1f}% | category {f.category.nunique()}종 brand {f.brand.nunique()}종")

    print("\n[A] 범주형 인코딩 비교 (CV AUC)")
    enc = cv_encoding(f)
    for k, v in sorted(enc.items(), key=lambda x: -x[1]):
        print(f"  {k:12s} {v:.4f}")

    print("\n[B] 모델별 전처리 베이지안 최적 (CV AUC)")
    modres = {}
    for mn in ["LogReg", "GBM", "MLP"]:
        auc, bp = bayes_model(X, y, mn, n_trials=20)
        modres[mn] = {"auc": round(auc, 4), "best": bp}
        print(f"  {mn:7s} {auc:.4f} | {bp}")

    json.dump({"encoding": {k: round(v, 4) for k, v in enc.items()}, "per_model": modres},
              open(os.path.join(SP, "outputs", "preprocess_bayes2.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n저장 → sample_project/outputs/preprocess_bayes2.json")


if __name__ == "__main__":
    main()

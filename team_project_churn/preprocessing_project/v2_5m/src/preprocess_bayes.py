# -*- coding: utf-8 -*-
"""전처리 기법 베이지안 최적화 + 파일 포맷/용량 비교.
대상: sample_project 실제 REES46 피처(data/processed/features.csv).
탐색: 스케일러·로그변환·이상치클립·불균형처리 조합 (모델 고정=LogReg, 누수 차단 Pipeline+CV).
출력: outputs/preprocess_bayes.json, 콘솔에 best 조합 + 포맷 용량표."""
import os, sys, json, io
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

optuna.logging.set_verbosity(optuna.logging.WARNING)
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # repo root(team_project_churn)
FEAT_CSV = os.path.join(HERE, "sample_project", "data", "processed", "features.csv")
SEQ_NPZ = os.path.join(HERE, "sample_project", "data", "processed", "sequences.npz")
RAW_CSV = os.path.join(HERE, "sample_project", "data", "raw", "events.csv")
OUT = os.path.join(HERE, "sample_project", "outputs"); os.makedirs(OUT, exist_ok=True)
SEED = 42
FEAT = ["recency_days", "n_view", "n_cart", "n_purchase", "n_events", "active_days", "avg_price"]
COUNT_IDX = [1, 2, 3, 4, 5]   # n_view..active_days (로그변환 대상)


class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self, k=1.5): self.k = k
    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.q1_ = np.nanpercentile(X, 25, axis=0); self.q3_ = np.nanpercentile(X, 75, axis=0)
        iqr = self.q3_ - self.q1_
        self.lo_ = self.q1_ - self.k * iqr; self.hi_ = self.q3_ + self.k * iqr
        return self
    def transform(self, X):
        return np.clip(np.asarray(X, float), self.lo_, self.hi_)


def make_pipe(p):
    steps = []
    if p["log_counts"]:
        steps.append(("log", ColumnTransformer(
            [("l", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), COUNT_IDX)],
            remainder="passthrough")))
    if p["iqr_clip"]:
        steps.append(("clip", IQRClipper(k=p["iqr_k"])))
    sc = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}.get(p["scaler"])
    if sc is not None:
        steps.append(("scale", sc))
    if p["imbalance"] == "smote":
        steps.append(("smote", SMOTE(random_state=SEED)))
    cw = "balanced" if p["imbalance"] == "class_weight" else None
    steps.append(("clf", LogisticRegression(max_iter=2000, class_weight=cw, random_state=SEED)))
    return ImbPipeline(steps)


def cv_auc(pipe, X, y):
    return cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                           scoring="roc_auc", n_jobs=-1).mean()


def bayes(X, y, n_trials=40):
    def objective(t):
        p = {
            "scaler": t.suggest_categorical("scaler", ["standard", "minmax", "robust", "none"]),
            "log_counts": t.suggest_categorical("log_counts", [True, False]),
            "iqr_clip": t.suggest_categorical("iqr_clip", [True, False]),
            "iqr_k": t.suggest_float("iqr_k", 1.5, 3.0),
            "imbalance": t.suggest_categorical("imbalance", ["none", "class_weight", "smote"]),
        }
        return cv_auc(make_pipe(p), X, y)
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    return study


def file_sizes():
    """전처리 산출물 위치·포맷·용량 + 포맷 변환 비교."""
    rows = []
    def add(label, path):
        if os.path.exists(path):
            rows.append((label, os.path.relpath(path, HERE), os.path.getsize(path)))
    add("원본 이벤트(csv)", RAW_CSV)
    add("정형 피처(csv)", FEAT_CSV)
    add("시퀀스(npz, 압축)", SEQ_NPZ)
    # 포맷 변환 비교
    comp = []
    df = pd.read_csv(FEAT_CSV)
    base = os.path.getsize(FEAT_CSV)
    # csv.gz
    gz = FEAT_CSV + ".gz"; df.to_csv(gz, index=False, compression="gzip")
    comp.append(("features.csv", "csv", base))
    comp.append(("features.csv.gz", "csv+gzip", os.path.getsize(gz)))
    # parquet (가능 시)
    try:
        pq = FEAT_CSV.replace(".csv", ".parquet"); df.to_parquet(pq, index=False)
        comp.append(("features.parquet", "parquet(snappy)", os.path.getsize(pq)))
    except Exception as e:
        comp.append(("features.parquet", f"불가({str(e)[:25]})", -1))
    # 시퀀스: npz(압축) vs npy(비압축)
    z = np.load(SEQ_NPZ)["X"]
    npy = os.path.join(os.path.dirname(SEQ_NPZ), "_seq_tmp.npy"); np.save(npy, z)
    seqcomp = [("sequences.npz", "npz(압축)", os.path.getsize(SEQ_NPZ)),
               ("_seq_tmp.npy", "npy(비압축)", os.path.getsize(npy))]
    os.remove(npy)
    return rows, comp, seqcomp


def main():
    df = pd.read_csv(FEAT_CSV)
    X = df[FEAT].values.astype(float); y = df["churn"].values.astype(int)
    print(f"데이터: {df.shape[0]}명, 이탈률 {y.mean()*100:.1f}% (sample_project 실제 REES46)")

    base_p = {"scaler": "standard", "log_counts": False, "iqr_clip": False, "iqr_k": 1.5, "imbalance": "class_weight"}
    base_auc = cv_auc(make_pipe(base_p), X, y)
    print(f"[기준 전처리] StandardScaler+class_weight → CV AUC {base_auc:.4f}")

    study = bayes(X, y, n_trials=40)
    best = study.best_params; best_auc = study.best_value
    print(f"[베이지안 최적] CV AUC {best_auc:.4f}")
    print(f"  최적 전처리: {best}")
    print(f"  개선폭: {best_auc - base_auc:+.4f}")

    rows, comp, seqcomp = file_sizes()
    print("\n[전처리 파일 위치/용량]")
    for l, p, s in rows: print(f"  {l:18s} {p:42s} {s/1024:8.1f} KB")
    print("[정형 피처 포맷별 용량]")
    for l, f, s in comp: print(f"  {l:20s} {f:16s} {('-' if s<0 else f'{s/1024:.1f} KB')}")
    print("[시퀀스 포맷별 용량]")
    for l, f, s in seqcomp: print(f"  {l:18s} {f:14s} {s/1024:.1f} KB")

    json.dump({"base_auc": round(base_auc, 4), "best_auc": round(best_auc, 4),
               "best_params": best, "improve": round(best_auc - base_auc, 4),
               "files": [{"label": l, "path": p, "bytes": s} for l, p, s in rows],
               "tabular_formats": [{"name": l, "fmt": f, "bytes": s} for l, f, s in comp],
               "seq_formats": [{"name": l, "fmt": f, "bytes": s} for l, f, s in seqcomp],
               "trials": [{"auc": round(t.value, 4), "params": t.params}
                          for t in sorted(study.trials, key=lambda x: -(x.value or 0))[:5]]},
              open(os.path.join(OUT, "preprocess_bayes.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n저장 → sample_project/outputs/preprocess_bayes.json")


if __name__ == "__main__":
    main()

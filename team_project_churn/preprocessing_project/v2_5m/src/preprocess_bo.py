# -*- coding: utf-8 -*-
"""전처리 기법 베이지안 최적화 (optuna TPE).
Bank Churn 데이터에서 인코딩/스케일링/이상치/불균형 처리 조합을 탐색해
5-fold CV ROC-AUC를 최대화하는 '최적 전처리'를 찾는다. (누수 방지: 전처리는 fold마다 fit)
출력: outputs/preprocess_bo.json
"""
import os, sys, json
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # repo root(team_project_churn)
RAW = os.path.join(HERE, "data", "raw", "Churn_Modelling.csv")
OUT = os.path.join(HERE, "outputs"); os.makedirs(OUT, exist_ok=True)
SEED = 42
TARGET, DROP = "Churn", ["CustomerId", "Surname"]
CAT = ["Geography", "Gender"]


class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self, enable=True, k=1.5): self.enable = enable; self.k = k
    def fit(self, X, y=None):
        X = np.asarray(X, float); q1 = np.percentile(X, 25, 0); q3 = np.percentile(X, 75, 0)
        iqr = q3 - q1; self.lo_, self.hi_ = q1 - self.k * iqr, q3 + self.k * iqr; return self
    def transform(self, X):
        return np.clip(np.asarray(X, float), self.lo_, self.hi_) if self.enable else np.asarray(X, float)


df = pd.read_csv(RAW).drop(columns=[c for c in DROP if c in pd.read_csv(RAW, nrows=1).columns])
y = df[TARGET].astype(int).values
X = df.drop(columns=[TARGET])
NUM = [c for c in X.columns if c not in CAT]
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)


def build(p):
    scaler = {"standard": StandardScaler(), "minmax": MinMaxScaler(),
              "robust": RobustScaler(), "none": "passthrough"}[p["scaler"]]
    num_pipe = Pipeline([("clip", IQRClipper(enable=p["outlier"] == "iqr")), ("scale", scaler)])
    enc = (OneHotEncoder(handle_unknown="ignore") if p["encoding"] == "onehot"
           else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    pre = ColumnTransformer([("num", num_pipe, NUM), ("cat", enc, CAT)])
    cw = "balanced" if p["imbalance"] == "class_weight" else None
    model = {"logreg": LogisticRegression(max_iter=1000, class_weight=cw, random_state=SEED),
             "rf": RandomForestClassifier(n_estimators=300, class_weight=cw, n_jobs=-1, random_state=SEED),
             "gbm": GradientBoostingClassifier(random_state=SEED)}[p["model"]]
    steps = [("pre", pre)]
    if p["imbalance"] == "smote":
        steps.append(("smote", SMOTE(random_state=SEED)))
    steps.append(("model", model))
    return ImbPipeline(steps)


def objective(trial):
    p = {
        "encoding": trial.suggest_categorical("encoding", ["onehot", "ordinal"]),
        "scaler": trial.suggest_categorical("scaler", ["standard", "minmax", "robust", "none"]),
        "outlier": trial.suggest_categorical("outlier", ["none", "iqr"]),
        "imbalance": trial.suggest_categorical("imbalance", ["none", "class_weight", "smote"]),
        "model": trial.suggest_categorical("model", ["logreg", "rf", "gbm"]),
    }
    score = cross_val_score(build(p), X, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
    return score


def main():
    print(f"데이터 {X.shape}, 이탈률 {y.mean():.3f}")
    # baseline (전형적 기본): standard + onehot + none + gbm
    base_p = {"encoding": "onehot", "scaler": "standard", "outlier": "none", "imbalance": "none", "model": "gbm"}
    base = cross_val_score(build(base_p), X, y, cv=cv, scoring="roc_auc").mean()
    print(f"baseline(onehot+standard+none+gbm) CV AUC {base:.4f}")

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=40)
    print(f"\nbest CV AUC {study.best_value:.4f}  (baseline 대비 {study.best_value-base:+.4f})")
    print("best 전처리:", study.best_params)
    try:
        imp = optuna.importance.get_param_importances(study)
        print("전처리 요소 중요도:", {k: round(v, 3) for k, v in imp.items()})
    except Exception as e:
        imp = {}; print("중요도 생략", e)

    # 요소별 최고 CV AUC (한 요소를 고정했을 때 best)
    rows = [{**t.params, "auc": t.value} for t in study.trials if t.value is not None]
    tdf = pd.DataFrame(rows)
    by = {}
    for col in ["encoding", "scaler", "outlier", "imbalance", "model"]:
        by[col] = tdf.groupby(col)["auc"].max().round(4).to_dict()

    json.dump({"data_shape": list(X.shape), "churn_rate": round(float(y.mean()), 4),
               "baseline_auc": round(float(base), 4), "best_auc": round(float(study.best_value), 4),
               "best_params": study.best_params, "param_importance": {k: round(v, 4) for k, v in imp.items()},
               "best_per_option": by, "n_trials": len(study.trials)},
              open(os.path.join(OUT, "preprocess_bo.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("저장 → outputs/preprocess_bo.json")


if __name__ == "__main__":
    main()

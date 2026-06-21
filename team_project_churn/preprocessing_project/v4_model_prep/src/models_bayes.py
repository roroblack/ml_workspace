# -*- coding: utf-8 -*-
"""v4 계획서23-B: 7모델을 v2 피처(category/brand/세션/remove/price 살린)로 재실행 + Brier/ECE.
- 입력: processed_eventbase/{train,test}_tabular_v2.parquet (코호트 subset으로 v1과 동일 모집단 비교).
- 살린 신호를 '수치'로 사용(n_categories·cat_entropy·brand_loyalty·remove_ratio·price분포 등).
- 각 모델 폴더에 산출 + ★30행 텍스트(<Model>_first30.txt) 저장.
실행: python .../models_bayes_v2.py <Model> [n_trials]
"""
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, joblib
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

optuna.logging.set_verbosity(optuna.logging.WARNING)
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
E = os.path.join(HERE, "preprocessing_project", "v4_model_prep", "output", "processed_eventbase")
V4 = os.path.join(HERE, "preprocessing_project", "v4_model_prep", "output")
SEED = 42
TREE_MODELS = {"DecisionTree", "RandomForest", "XGBoost", "LightGBM", "CatBoost"}


# ---- 모델/HP/파이프 (자체완결: 구 models_bayes_one 의존 제거) ----
def make_model(name, hp, cw):
    if name == "DecisionTree":
        return DecisionTreeClassifier(max_depth=hp["max_depth"], min_samples_leaf=hp["min_samples_leaf"],
                                      ccp_alpha=hp["ccp_alpha"], class_weight=cw, random_state=SEED)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
                                      min_samples_leaf=hp["min_samples_leaf"], max_features=hp["max_features"],
                                      class_weight=cw, n_jobs=-1, random_state=SEED)
    if name == "LogReg":
        return LogisticRegression(C=hp["C"], penalty="l2", class_weight=cw, max_iter=2000, random_state=SEED)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"], learning_rate=hp["lr"],
                             subsample=hp["subsample"], colsample_bytree=hp["colsample"], min_child_weight=hp["min_child_weight"],
                             reg_alpha=hp["reg_alpha"], reg_lambda=hp["reg_lambda"], tree_method="hist",
                             scale_pos_weight=(hp.get("scale_pos_weight", 1) if cw else 1),
                             eval_metric="logloss", random_state=SEED, n_jobs=-1)
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=hp["n_estimators"], num_leaves=hp["num_leaves"], max_depth=hp["max_depth"],
                              learning_rate=hp["lr"], min_child_samples=hp["min_child_samples"],
                              subsample=hp["subsample"], colsample_bytree=hp["colsample"],
                              reg_alpha=hp["reg_alpha"], reg_lambda=hp["reg_lambda"],
                              class_weight=cw, random_state=SEED, n_jobs=-1, verbose=-1)
    if name == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=hp["iterations"], depth=hp["depth"], learning_rate=hp["lr"],
                                  l2_leaf_reg=hp["l2_leaf_reg"], auto_class_weights="Balanced" if cw else None,
                                  random_state=SEED, verbose=0, thread_count=-1)
    raise ValueError(name)


def sample_hp(t, name):
    if name == "DecisionTree":
        return {"max_depth": t.suggest_int("max_depth", 3, 20), "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 80),
                "ccp_alpha": t.suggest_float("ccp_alpha", 0.0, 0.02)}
    if name == "RandomForest":
        return {"n_estimators": t.suggest_int("n_estimators", 100, 400), "max_depth": t.suggest_int("max_depth", 4, 24),
                "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 50),
                "max_features": t.suggest_categorical("max_features", ["sqrt", "log2", 0.5])}
    if name == "LogReg":
        return {"C": t.suggest_float("C", 1e-3, 1e2, log=True)}
    if name == "XGBoost":
        return {"n_estimators": t.suggest_int("n_estimators", 150, 500), "max_depth": t.suggest_int("max_depth", 3, 10),
                "lr": t.suggest_float("lr", 0.02, 0.3, log=True), "subsample": t.suggest_float("subsample", 0.6, 1.0),
                "colsample": t.suggest_float("colsample", 0.6, 1.0), "min_child_weight": t.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": t.suggest_float("reg_alpha", 1e-3, 5.0, log=True), "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
                "scale_pos_weight": t.suggest_categorical("scale_pos_weight", [1, 4])}
    if name == "LightGBM":
        return {"n_estimators": t.suggest_int("n_estimators", 150, 500), "num_leaves": t.suggest_int("num_leaves", 15, 200),
                "max_depth": t.suggest_int("max_depth", -1, 12), "lr": t.suggest_float("lr", 0.02, 0.3, log=True),
                "min_child_samples": t.suggest_int("min_child_samples", 5, 100), "subsample": t.suggest_float("subsample", 0.6, 1.0),
                "colsample": t.suggest_float("colsample", 0.6, 1.0), "reg_alpha": t.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
                "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 5.0, log=True)}
    if name == "CatBoost":
        return {"iterations": t.suggest_int("iterations", 150, 400), "depth": t.suggest_int("depth", 4, 10),
                "lr": t.suggest_float("lr", 0.02, 0.3, log=True), "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1.0, 10.0)}
    return {}


def build_pipe(name, prep, hp):
    steps = []
    if prep["imbalance"] == "smote":
        steps.append(("smote", SMOTE(random_state=SEED)))
    cw = "balanced" if prep["imbalance"] == "classweight" else None
    steps.append(("clf", make_model(name, hp, cw)))
    return ImbPipeline(steps)
# v2 수치 피처(살린 category/brand/세션/remove/price 신호 포함)
FEAT_V2 = ["recency_days", "tenure_days", "ndays", "n_events", "n_view", "n_cart", "n_remove_from_cart",
           "n_purchase", "avg_price", "purch_amt", "min_price", "max_price", "std_price", "purchase_avg_price",
           "remove_ratio", "cart_purchase_ratio", "n_categories", "cat_entropy", "n_brands", "brand_loyalty",
           "n_sessions", "events_per_session"]
COUNTS = ["ndays", "n_events", "n_view", "n_cart", "n_remove_from_cart", "n_purchase", "purch_amt", "n_categories", "n_brands", "n_sessions"]
COUNT_IDX = [FEAT_V2.index(c) for c in COUNTS]


def transform(Xtr, Xte, prep):
    a, b = Xtr.copy(), Xte.copy()
    if prep["log_counts"]:
        a[:, COUNT_IDX] = np.log1p(np.clip(a[:, COUNT_IDX], 0, None)); b[:, COUNT_IDX] = np.log1p(np.clip(b[:, COUNT_IDX], 0, None))
    sc = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}.get(prep["scaler"])
    if sc is not None:
        sc.fit(a); a, b = sc.transform(a), sc.transform(b)
    return a, b, sc


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            e += abs(p[m].mean() - y[m].mean()) * m.mean()
    return float(e)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "DecisionTree"
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else (15 if name in ("XGBoost", "LightGBM", "CatBoost") else 25)
    mo = os.path.join(V4, name); os.makedirs(mo, exist_ok=True)
    tr = pd.read_parquet(os.path.join(E, "train_tabular_v2.parquet"))
    te = pd.read_parquet(os.path.join(E, "test_tabular_v2.parquet"))
    tr = tr[tr.cohort_recency7 == 1]; te = te[te.cohort_recency7 == 1]      # v1과 동일 모집단(코호트)
    Xtr = np.nan_to_num(tr[FEAT_V2].values.astype(float)); Xte = np.nan_to_num(te[FEAT_V2].values.astype(float))
    ytr = tr["churn"].values.astype(int); yte = te["churn"].values.astype(int)
    print(f"[{name}/v2] train {len(Xtr):,}(이탈 {ytr.mean()*100:.1f}%) / test {len(Xte):,} | {len(FEAT_V2)}피처(v2) | {n_trials}trial", flush=True)
    trials = []

    def objective(t):
        prep = {"scaler": t.suggest_categorical("scaler", ["none", "standard", "robust"] if name in TREE_MODELS else ["standard", "minmax", "robust"]),
                "log_counts": t.suggest_categorical("log_counts", [True, False]),
                "imbalance": t.suggest_categorical("imbalance", ["none", "classweight", "smote"])}
        hp = sample_hp(t, name)
        Xa, _, _ = transform(Xtr, Xtr, prep)
        sc = cross_val_score(build_pipe(name, prep, hp), Xa, ytr, cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                             scoring="average_precision", n_jobs=1).mean()
        trials.append({**{k: prep[k] for k in prep}, "pr_auc_cv": round(float(sc), 4)})
        return sc

    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED)); study.optimize(objective, n_trials=n_trials)
    bp = study.best_params
    prep = {"scaler": bp["scaler"], "log_counts": bp["log_counts"], "imbalance": bp["imbalance"]}
    hp = {k: bp[k] for k in bp if k not in ("scaler", "log_counts", "imbalance")}

    Xa, Xb, scaler = transform(Xtr, Xte, prep)
    Xh, Xv, yh, yv = train_test_split(Xa, ytr, test_size=0.2, stratify=ytr, random_state=SEED)
    pipe_h = build_pipe(name, prep, hp).fit(Xh, yh)
    cal = CalibratedClassifierCV(pipe_h, method="isotonic", cv="prefit").fit(Xv, yv)
    p = cal.predict_proba(Xb)[:, 1]
    pr = average_precision_score(yte, p); auc = roc_auc_score(yte, p)
    brier = brier_score_loss(yte, p); e = ece(yte, p)
    ths = np.linspace(0.1, 0.9, 81); best_th, best_f1 = max(((th, f1_score(yte, (p >= th).astype(int))) for th in ths), key=lambda x: x[1])

    def best_by(k):
        d = {}
        for r in trials: d[r[k]] = max(d.get(r[k], 0), r["pr_auc_cv"])
        return d
    per_option = {k: best_by(k) for k in ["scaler", "log_counts", "imbalance"]}
    metrics = {"cv_pr_auc": round(float(study.best_value), 4), "oot_pr_auc": round(float(pr), 4), "oot_auc": round(float(auc), 4),
               "brier": round(float(brier), 4), "ece": round(float(e), 4), "f1@thr": round(float(best_f1), 4),
               "base_rate": round(float(ytr.mean()), 4), "n_features": len(FEAT_V2)}

    joblib.dump({"model_name": f"{name}_v2", "model_type": "tree" if name in TREE_MODELS else "linear",
                 "feature_order": FEAT_V2, "prep": prep, "hp": hp, "scaler": scaler, "calibrator": cal,
                 "threshold": float(best_th), "metrics": metrics}, os.path.join(mo, f"prep_{name}_v2.joblib"))
    json.dump({"best_params": bp, "per_option_best": per_option, "metrics": metrics, "threshold": float(best_th),
               "n_trials": n_trials}, open(os.path.join(mo, f"{name}_v2_bayes.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    dfv = pd.DataFrame(Xa, columns=FEAT_V2).assign(churn=ytr, user_id=tr["user_id"].values)
    dfv.to_parquet(os.path.join(mo, f"{name}_v2_train.parquet"), index=False)

    # ★ 30행 텍스트 (엑셀에서 열듯) — 원본 v2 피처값(변환 전) + 라벨
    raw30 = tr[["user_id"] + FEAT_V2 + ["churn"]].head(30)
    with open(os.path.join(mo, f"{name}_first30.txt"), "w", encoding="utf-8") as f:
        f.write(f"# {name} v2 전처리 데이터 — 앞 30행 (코호트, {len(FEAT_V2)}피처)\n")
        f.write(f"# 성능(Feb): PR-AUC {metrics['oot_pr_auc']} | AUC {metrics['oot_auc']} | Brier {metrics['brier']} | ECE {metrics['ece']} | thr {best_th:.2f}\n")
        f.write(f"# 최적 전처리: scaler={prep['scaler']} log={prep['log_counts']} imbalance={prep['imbalance']}\n\n")
        f.write(raw30.to_string(index=False))
    print(f"[{name}/v2] CV {metrics['cv_pr_auc']} | Feb AUC {auc:.4f} PR {pr:.4f} Brier {brier:.4f} ECE {e:.4f} | thr {best_th:.2f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"RESULTV2\t{name}\t{metrics['oot_auc']}\t{metrics['oot_pr_auc']}\t{metrics['brier']}\t{metrics['ece']}\t{prep['scaler']}|{prep['imbalance']}", flush=True)


if __name__ == "__main__":
    main()

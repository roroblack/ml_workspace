# -*- coding: utf-8 -*-
"""v4-3 바운스 / v4-4 churn30 — 이진분류 모델별 전처리+HP 베이지안 최적화(+ES·isotonic·threshold).

지표: ROC-AUC·PR-AUC·Brier·F1@thr (추천 top-k 아님). 속도: early stopping·hist·작은트리·대용량시 표본탐색.
산출(폴더 output/<Model>/): prep_<Model>_bin.joblib(prep+scaler+calibrator+threshold) · <Model>_bin_train.parquet · bayes.json · first30.
실행: python models_bin_bayes.py <bounce|churn30> <Model> [n_trials]
"""
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, joblib
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

optuna.logging.set_verbosity(optuna.logging.WARNING)
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SEED = 42
TREE = {"DecisionTree", "RandomForest", "XGBoost", "LightGBM", "CatBoost"}
BOOST = {"XGBoost", "LightGBM", "CatBoost"}
N_BOOST = 400; ES = 30; MAX_BIN = 127
SAMPLE_CAP = 150000   # 탐색(objective)용 표본 상한(대용량 바운스 가속)

DSETS = {
    "bounce":  dict(dir="v4-3_bounce",  train="train_bounce.parquet",  test="test_bounce.parquet",  y="churn30",
                    drop=["user_id", "user_session"], counts=["step", "n_view_sf", "n_cart_sf", "n_purchase_sf"]),
    "churn30": dict(dir="v4-4_churn30", train="train_churn30.parquet", test="test_churn30.parquet", y="churn30",
                    drop=["user_id"], counts=["ndays", "n_events", "n_view", "n_cart", "n_remove_from_cart",
                    "n_purchase", "purch_amt", "n_categories", "n_brands", "n_sessions"]),
}


def make_model(name, hp, cw, spw):
    if name == "DecisionTree":
        return DecisionTreeClassifier(max_depth=hp["max_depth"], min_samples_leaf=hp["min_samples_leaf"], class_weight=cw, random_state=SEED)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"], min_samples_leaf=hp["min_samples_leaf"], class_weight=cw, n_jobs=-1, random_state=SEED)
    if name == "LogReg":
        return LogisticRegression(C=hp["C"], class_weight=cw, max_iter=2000, random_state=SEED, n_jobs=-1)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=N_BOOST, max_depth=hp["max_depth"], learning_rate=hp["lr"], subsample=hp["subsample"],
                             colsample_bytree=hp["colsample"], tree_method="hist", max_bin=MAX_BIN, eval_metric="logloss",
                             early_stopping_rounds=ES, scale_pos_weight=spw, random_state=SEED, n_jobs=-1)
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=N_BOOST, num_leaves=hp["num_leaves"], learning_rate=hp["lr"], min_child_samples=hp["min_child_samples"],
                              subsample=hp["subsample"], subsample_freq=1, colsample_bytree=hp["colsample"], max_bin=MAX_BIN,
                              class_weight=cw, force_col_wise=True, random_state=SEED, n_jobs=-1, verbose=-1)
    if name == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=min(N_BOOST, 300), depth=hp["depth"], learning_rate=hp["lr"], l2_leaf_reg=hp["l2_leaf_reg"],
                                  loss_function="Logloss", border_count=MAX_BIN, od_type="Iter", od_wait=ES,
                                  auto_class_weights="Balanced" if cw else None, random_state=SEED, verbose=0, thread_count=-1)
    raise ValueError(name)


def fit_es(name, model, Xtr, ytr, Xv, yv):
    if name == "XGBoost": model.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
    elif name == "LightGBM":
        import lightgbm as lgb; model.fit(Xtr, ytr, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(ES, verbose=False), lgb.log_evaluation(0)])
    elif name == "CatBoost": model.fit(Xtr, ytr, eval_set=(Xv, yv), verbose=False)
    else: model.fit(Xtr, ytr)
    return model


def sample_hp(t, name):
    if name == "DecisionTree": return {"max_depth": t.suggest_int("max_depth", 3, 20), "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 80)}
    if name == "RandomForest": return {"n_estimators": t.suggest_int("n_estimators", 100, 300), "max_depth": t.suggest_int("max_depth", 4, 24), "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 50)}
    if name == "LogReg": return {"C": t.suggest_float("C", 1e-3, 1e2, log=True)}
    if name == "XGBoost": return {"max_depth": t.suggest_int("max_depth", 3, 9), "lr": t.suggest_float("lr", 0.03, 0.3, log=True), "subsample": t.suggest_float("subsample", 0.7, 1.0), "colsample": t.suggest_float("colsample", 0.6, 1.0)}
    if name == "LightGBM": return {"num_leaves": t.suggest_int("num_leaves", 15, 128), "lr": t.suggest_float("lr", 0.03, 0.3, log=True), "min_child_samples": t.suggest_int("min_child_samples", 10, 200), "subsample": t.suggest_float("subsample", 0.7, 1.0), "colsample": t.suggest_float("colsample", 0.6, 1.0)}
    if name == "CatBoost": return {"depth": t.suggest_int("depth", 4, 8), "lr": t.suggest_float("lr", 0.03, 0.3, log=True), "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1.0, 10.0)}
    return {}


def main():
    dskey = sys.argv[1] if len(sys.argv) > 1 else "bounce"
    name = sys.argv[2] if len(sys.argv) > 2 else "LightGBM"
    n_trials = int(sys.argv[3]) if len(sys.argv) > 3 else (4 if name in BOOST else 8)
    ds = DSETS[dskey]; DIR = os.path.join(HERE, "preprocessing_project", ds["dir"], "output")
    tr = pd.read_parquet(os.path.join(DIR, ds["train"])); te = pd.read_parquet(os.path.join(DIR, ds["test"]))
    FEAT = [c for c in tr.columns if c not in ds["drop"] + [ds["y"]]]
    CIDX = [FEAT.index(c) for c in ds["counts"] if c in FEAT]
    Xtr = np.nan_to_num(tr[FEAT].values.astype(float)); Xte = np.nan_to_num(te[FEAT].values.astype(float))
    ytr = tr[ds["y"]].values.astype(int); yte = te[ds["y"]].values.astype(int)
    pos = float(ytr.mean())
    mo = os.path.join(DIR, name); os.makedirs(mo, exist_ok=True)
    # 탐색용 표본(대용량 가속) — 층화
    if len(Xtr) > SAMPLE_CAP:
        idx, _ = train_test_split(np.arange(len(Xtr)), train_size=SAMPLE_CAP, random_state=SEED, stratify=ytr)
        Xs, ys = Xtr[idx], ytr[idx]
    else: Xs, ys = Xtr, ytr
    Xh, Xv, yh, yv = train_test_split(Xs, ys, test_size=0.2, random_state=SEED, stratify=ys)
    print(f"[{dskey}/{name}] train {len(Xtr):,}(pos {pos*100:.1f}%) / test {len(Xte):,} | {len(FEAT)}피처 | {n_trials}trial", flush=True)

    def transform(Xfit, Xapply, prep):
        a, b = Xfit.copy(), Xapply.copy()
        if prep["log_counts"] and CIDX:
            a[:, CIDX] = np.log1p(np.clip(a[:, CIDX], 0, None)); b[:, CIDX] = np.log1p(np.clip(b[:, CIDX], 0, None))
        sc = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}.get(prep["scaler"])
        if sc is not None: sc.fit(a); a, b = sc.transform(a), sc.transform(b)
        return a, b, sc

    def objective(t):
        prep = {"scaler": t.suggest_categorical("scaler", ["none", "standard", "robust"] if name in TREE else ["standard", "minmax", "robust"]),
                "log_counts": t.suggest_categorical("log_counts", [True, False]),
                "imbalance": t.suggest_categorical("imbalance", ["none", "classweight"])}
        cw = "balanced" if prep["imbalance"] == "classweight" else None
        spw = (1 - pos) / max(pos, 1e-6) if (name == "XGBoost" and prep["imbalance"] == "classweight") else 1
        Xa, Xb, _ = transform(Xh, Xv, prep)
        m = fit_es(name, make_model(name, sample_hp(t, name), cw, spw), Xa, yh, Xb, yv)
        p = m.predict_proba(Xb)[:, 1]
        return average_precision_score(yv, p)

    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED)); study.optimize(objective, n_trials=n_trials)
    bp = study.best_params; prep = {k: bp[k] for k in ("scaler", "log_counts", "imbalance")}; hp = {k: bp[k] for k in bp if k not in prep}
    cw = "balanced" if prep["imbalance"] == "classweight" else None
    spw = (1 - pos) / max(pos, 1e-6) if (name == "XGBoost" and prep["imbalance"] == "classweight") else 1
    Xfa, Xfb, scaler = transform(Xtr, Xte, prep)
    Xh2, Xv2, yh2, yv2 = train_test_split(Xfa, ytr, test_size=0.15, random_state=SEED, stratify=ytr)
    base = fit_es(name, make_model(name, hp, cw, spw), Xh2, yh2, Xv2, yv2)
    cal = CalibratedClassifierCV(base, method="isotonic", cv="prefit").fit(Xv2, yv2)
    p = cal.predict_proba(Xfb)[:, 1]
    auc = roc_auc_score(yte, p); pr = average_precision_score(yte, p); brier = brier_score_loss(yte, p)
    ths = np.linspace(0.1, 0.9, 81); best_th, best_f1 = max(((th, f1_score(yte, (p >= th).astype(int))) for th in ths), key=lambda x: x[1])
    metrics = {"oot_auc": round(float(auc), 4), "oot_pr_auc": round(float(pr), 4), "brier": round(float(brier), 4),
               "f1@thr": round(float(best_f1), 4), "threshold": round(float(best_th), 3), "pos_rate": round(pos, 4), "n_features": len(FEAT)}
    joblib.dump({"model_name": f"{name}_bin_{dskey}", "model_type": "tree" if name in TREE else "linear", "task": f"binary_{dskey}",
                 "target": ds["y"], "feature_order": FEAT, "prep": prep, "hp": hp, "scaler": scaler, "calibrator": cal,
                 "threshold": float(best_th), "metrics": metrics}, os.path.join(mo, f"prep_{name}_bin.joblib"))
    json.dump({"best_params": bp, "metrics": metrics, "n_trials": n_trials}, open(os.path.join(mo, f"{name}_bin_bayes.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    pd.DataFrame(Xfa, columns=FEAT).assign(**{ds["y"]: ytr}).to_parquet(os.path.join(mo, f"{name}_bin_train.parquet"), index=False)
    with open(os.path.join(mo, f"{name}_first30.txt"), "w", encoding="utf-8") as f:
        f.write(f"# {name} {dskey}(이진) — AUC {auc:.4f} PR {pr:.4f} Brier {brier:.4f} F1 {best_f1:.4f} thr {best_th:.2f} | {prep}\n\n")
        f.write(tr[ds["drop"] + FEAT + [ds["y"]]].head(30).to_string(index=False))
    print(f"RESULTBIN\t{dskey}\t{name}\tAUC={auc:.4f}\tPR={pr:.4f}\tBrier={brier:.4f}\tF1={best_f1:.4f}\t{prep['scaler']}|{prep['imbalance']}\t({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

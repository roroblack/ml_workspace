# -*- coding: utf-8 -*-
"""v4 — 모델 '하나씩' 전처리+HP 베이지안 탐색(실시간 안전). 모델별 폴더에 산출.
- 입력: 5m 코호트 정본(realtime-safe 10피처). 정본 불변(추가형).
- 모델별: 전처리(scaler/log/imbalance) + 핵심 HP 공동탐색(optuna TPE).
- 목적: 3-fold Stratified CV PR-AUC. 이후 Feb 시간외삽 + isotonic 보정 + F1 임계값.
- 산출(output/<Model>/): prep_{Model}.joblib, {Model}_bayes.json, {Model}_{train,test}.parquet(전처리본 백업),
                        {Model}_전처리리포트.md(인수인계).
실행: python .../models_bayes_one.py <Model> [n_trials]
  Model ∈ DecisionTree RandomForest LogReg XGBoost LightGBM CatBoost
"""
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, joblib
import optuna
from optuna.samplers import TPESampler
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

optuna.logging.set_verbosity(optuna.logging.WARNING)
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
D = os.path.join(HERE, "sample_project", "data", "processed_5m")
V4 = os.path.join(HERE, "preprocessing_project", "v4_model_prep", "output")
SEED = 42
FEAT = ["recency_days", "tenure_days", "ndays", "n_events", "n_view", "n_cart",
        "n_remove_from_cart", "n_purchase", "avg_price", "purch_amt"]
COUNT_IDX = [2, 3, 4, 5, 6, 7, 9]
TREE_MODELS = {"DecisionTree", "RandomForest", "XGBoost", "LightGBM", "CatBoost"}


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
        spw = hp.get("scale_pos_weight", 1) if cw else 1
        return XGBClassifier(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"], learning_rate=hp["lr"],
                             subsample=hp["subsample"], colsample_bytree=hp["colsample"], min_child_weight=hp["min_child_weight"],
                             reg_alpha=hp["reg_alpha"], reg_lambda=hp["reg_lambda"], tree_method="hist",
                             scale_pos_weight=spw, eval_metric="logloss", random_state=SEED, n_jobs=-1)
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


def transform(Xtr, Xte, prep):
    a, b = Xtr.copy(), Xte.copy()
    if prep["log_counts"]:
        a[:, COUNT_IDX] = np.log1p(a[:, COUNT_IDX]); b[:, COUNT_IDX] = np.log1p(b[:, COUNT_IDX])
    sc = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}.get(prep["scaler"])
    if sc is not None:
        sc.fit(a); a, b = sc.transform(a), sc.transform(b)
    return a, b, sc


def build_pipe(name, prep, hp):
    steps = []
    if prep["imbalance"] == "smote":
        steps.append(("smote", SMOTE(random_state=SEED)))
    cw = "balanced" if prep["imbalance"] == "classweight" else None
    steps.append(("clf", make_model(name, hp, cw)))
    return ImbPipeline(steps)


def write_report(name, mo, study, per_option, prep, hp, metrics, best_th, base_rate):
    p = os.path.join(mo, f"{name}_전처리리포트.md")
    lines = [f"# v4 · {name} — 전처리 베이지안 결과 (인수인계용)\n",
             f"재현: `python preprocessing_project/v4_model_prep/src/models_bayes_one.py {name}` → `output/{name}/`.\n",
             "## 1. 데이터 · X/Y",
             f"- 입력: `processed_5m/{{train,test}}_cohort_tabular.parquet` (코호트 recency≤7).",
             f"- X = 10피처(realtime-safe): {', '.join(FEAT)}",
             f"- Y = churn(7일 무활동). train 이탈률 {base_rate*100:.1f}%.\n",
             "## 2. 옵션별 best CV PR-AUC"]
    for k, dd in per_option.items():
        lines.append(f"- **{k}**: " + " / ".join(f"{o} {v}" for o, v in sorted(dd.items(), key=lambda x: -x[1])))
    lines += ["\n## 3. 선택된 전처리 + HP",
              f"- 전처리: scaler=**{prep['scaler']}**, log1p=**{prep['log_counts']}**, imbalance=**{prep['imbalance']}**",
              f"- HP: {hp}",
              "\n## 4. 성능 (Feb 시간외삽)",
              f"- CV PR-AUC **{metrics['cv_pr_auc']}** | Feb PR-AUC {metrics['oot_pr_auc']}(보정 {metrics['oot_pr_auc_cal']}) | **AUC {metrics['oot_auc']}** | 임계값 {best_th:.2f}(F1 {metrics['f1@thr']})",
              f"- base rate {metrics['base_rate']} (PR-AUC는 양성다수라 높음 — AUC가 분별력 지표)",
              "\n## 5. 산출물 · 서빙",
              f"- `prep_{name}.joblib`(전처리+모델+isotonic보정+임계값+feature_order), `{name}_bayes.json`, `{name}_{{train,test}}.parquet`(전처리본 백업).",
              f"- 서빙: 최근이벤트→10피처→(scaler/log)→calibrator.predict_proba→≥{best_th:.2f}. (`realtime_compat_test.py`)",
              "\n## 6. 백엔드 제출(계약)",
              f"```json\n{{\"model_name\":\"{name}_v4\",\"model_type\":\"{'tree' if name in TREE_MODELS else 'linear'}\",\"artifact_path\":\"preprocessing_project/v4_model_prep/output/{name}/prep_{name}.joblib\",\"preprocessing_config\":{{\"scale\":\"{prep['scaler']}\",\"log1p\":{str(prep['log_counts']).lower()},\"imbalance\":\"{prep['imbalance']}\",\"label\":\"churn\",\"threshold\":{best_th:.2f},\"calibrator\":\"isotonic\"}},\"metrics\":{json.dumps(metrics)}}}\n```"]
    open(p, "w", encoding="utf-8").write("\n".join(lines))


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "DecisionTree"
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else (20 if name in ("XGBoost", "LightGBM", "CatBoost") else 30)
    mo = os.path.join(V4, name); os.makedirs(mo, exist_ok=True)
    tr = pd.read_parquet(os.path.join(D, "train_cohort_tabular.parquet"))
    te = pd.read_parquet(os.path.join(D, "test_cohort_tabular.parquet"))
    Xtr = np.nan_to_num(tr[FEAT].values.astype(float)); Xte = np.nan_to_num(te[FEAT].values.astype(float))
    Xtr[:, COUNT_IDX] = np.clip(Xtr[:, COUNT_IDX], 0, None); Xte[:, COUNT_IDX] = np.clip(Xte[:, COUNT_IDX], 0, None)
    ytr = tr["churn"].values.astype(int); yte = te["churn"].values.astype(int)
    base_rate = float(ytr.mean())
    print(f"[{name}] train {len(Xtr):,}(이탈 {base_rate*100:.1f}%) / test {len(Xte):,} | 10피처 | {n_trials}trial", flush=True)
    trials = []

    def objective(t):
        prep = {"scaler": t.suggest_categorical("scaler", ["none", "standard", "robust"] if name in TREE_MODELS else ["standard", "minmax", "robust"]),
                "log_counts": t.suggest_categorical("log_counts", [True, False]),
                "imbalance": t.suggest_categorical("imbalance", ["none", "classweight", "smote"])}
        hp = sample_hp(t, name)
        Xa, _, _ = transform(Xtr, Xtr, prep)
        sc = cross_val_score(build_pipe(name, prep, hp), Xa, ytr,
                             cv=StratifiedKFold(3, shuffle=True, random_state=SEED), scoring="average_precision", n_jobs=1).mean()
        trials.append({"scaler": prep["scaler"], "log_counts": prep["log_counts"], "imbalance": prep["imbalance"], "pr_auc_cv": round(float(sc), 4)})
        return sc

    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED)); study.optimize(objective, n_trials=n_trials)
    bp = study.best_params
    prep = {"scaler": bp["scaler"], "log_counts": bp["log_counts"], "imbalance": bp["imbalance"]}
    hp = {k: bp[k] for k in bp if k not in ("scaler", "log_counts", "imbalance")}

    Xa, Xb, scaler = transform(Xtr, Xte, prep)
    pipe = build_pipe(name, prep, hp).fit(Xa, ytr)
    proba_raw = pipe.predict_proba(Xb)[:, 1]
    Xh, Xv, yh, yv = train_test_split(Xa, ytr, test_size=0.2, stratify=ytr, random_state=SEED)
    pipe_h = build_pipe(name, prep, hp).fit(Xh, yh)
    cal = CalibratedClassifierCV(pipe_h, method="isotonic", cv="prefit").fit(Xv, yv)
    proba_cal = cal.predict_proba(Xb)[:, 1]
    pr_raw = average_precision_score(yte, proba_raw); auc = roc_auc_score(yte, proba_raw); pr_cal = average_precision_score(yte, proba_cal)
    ths = np.linspace(0.1, 0.9, 81); best_th, best_f1 = max(((th, f1_score(yte, (proba_cal >= th).astype(int))) for th in ths), key=lambda x: x[1])

    def best_by(key):
        d = {}
        for r in trials: d[r[key]] = max(d.get(r[key], 0), r["pr_auc_cv"])
        return d
    per_option = {k: best_by(k) for k in ["scaler", "log_counts", "imbalance"]}
    metrics = {"cv_pr_auc": round(float(study.best_value), 4), "oot_pr_auc": round(float(pr_raw), 4),
               "oot_pr_auc_cal": round(float(pr_cal), 4), "oot_auc": round(float(auc), 4),
               "f1@thr": round(float(best_f1), 4), "base_rate": round(base_rate, 4)}

    joblib.dump({"model_name": f"{name}_v4", "model_type": "tree" if name in TREE_MODELS else "linear",
                 "feature_order": FEAT, "prep": prep, "hp": hp, "scaler": scaler, "pipeline": pipe,
                 "calibrator": cal, "threshold": float(best_th), "metrics": metrics}, os.path.join(mo, f"prep_{name}.joblib"))
    json.dump({"best_params": bp, "best_pr_auc_cv": metrics["cv_pr_auc"], "per_option_best": per_option,
               "metrics": metrics, "threshold": float(best_th), "n_trials": n_trials, "all_trials": trials},
              open(os.path.join(mo, f"{name}_bayes.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    # 전처리본 데이터셋 백업(유실 복원용)
    pd.DataFrame(Xa, columns=FEAT).assign(churn=ytr, user_id=tr["user_id"].values).to_parquet(os.path.join(mo, f"{name}_train.parquet"), index=False)
    pd.DataFrame(Xb, columns=FEAT).assign(churn=yte, user_id=te["user_id"].values).to_parquet(os.path.join(mo, f"{name}_test.parquet"), index=False)
    write_report(name, mo, study, per_option, prep, hp, metrics, best_th, base_rate)

    print(f"[{name}] CV {metrics['cv_pr_auc']} | Feb PR {pr_raw:.4f} AUC {auc:.4f} | thr {best_th:.2f} F1 {best_f1:.3f} | {prep} ({time.time()-t0:.0f}s)", flush=True)
    print(f"RESULT\t{name}\t{metrics['cv_pr_auc']}\t{metrics['oot_auc']}\t{metrics['oot_pr_auc']}\t{best_th:.2f}\t{prep['scaler']}|{prep['imbalance']}", flush=True)


if __name__ == "__main__":
    main()

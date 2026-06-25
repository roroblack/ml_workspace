# -*- coding: utf-8 -*-
"""v4-1/v4-2 추천 — models7처럼 **모델별 전처리+HP 베이지안 최적화**(다중분류판) + 속도 최적화.

models_bayes.py(이진 churn)의 추천 버전. 이진→다중분류 조정 + 학습 속도 최적화 전면 적용:
- 지표: PR-AUC/isotonic/SMOTE → **top-1/top-5 accuracy·MRR**.
- [속도1] **Early stopping**(boosting 전부, val mlogloss 30라운드) → 트리수 자동결정. n_estimators는 상한만 크게.
- [속도2] **Histogram + max_bin↓**(LGBM force_col_wise·max_bin127 / XGB tree_method=hist·max_bin128 / Cat border_count128).
- [속도3] HP탐색은 **단일 holdout(80/20)** 로 1-fit 평가(3-fold→1-fit, 3배↓) + 그 holdout이 ES val 겸용.
- [속도4] boosting HP에서 n_estimators 제거(ES가 결정) → 탐색차원↓.
- imbalance ∈ {none, classweight}, 보정 없음. (근거: web 2025 — ES 30라운드·hist·optuna가 멀티클래스 GBDT 표준 가속.)

산출(폴더 output/<Model>/): prep_<Model>_rec.joblib · <Model>_rec_bayes.json · <Model>_rec_train.parquet · <Model>_first30.txt
실행: python models_rec_bayes.py <cat|item> <Model> [n_trials]
"""
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, joblib
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

optuna.logging.set_verbosity(optuna.logging.WARNING)
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))   # → team_project_churn
SEED = 42
TREE_MODELS = {"DecisionTree", "RandomForest", "XGBoost", "LightGBM", "CatBoost"}
BOOST = {"XGBoost", "LightGBM", "CatBoost"}
N_BOOST = 300        # 상한(ES가 실제 트리수 결정). 다중분류는 라운드당 클래스수만큼 트리 → 상한↓
ES = 20              # early stopping rounds
MAX_BIN = 127        # histogram bin↓ → 속도↑

DSETS = {
    "cat":  dict(dir="v4-1_rec_category", train="train_cat.parquet", test="test_cat.parquet", y="y_next_category"),
    "item": dict(dir="v4-2_rec_item",     train="train_item.parquet", test="test_item.parquet", y="y_next_item"),
}
FEAT_V2 = ["recency_days", "tenure_days", "ndays", "n_events", "n_view", "n_cart", "n_remove_from_cart",
           "n_purchase", "avg_price", "purch_amt", "min_price", "max_price", "std_price", "purchase_avg_price",
           "remove_ratio", "cart_purchase_ratio", "n_categories", "cat_entropy", "n_brands", "brand_loyalty",
           "n_sessions", "events_per_session"]
COUNTS = ["ndays", "n_events", "n_view", "n_cart", "n_remove_from_cart", "n_purchase", "purch_amt", "n_categories", "n_brands", "n_sessions"]
COUNT_IDX = [FEAT_V2.index(c) for c in COUNTS]


def make_model(name, hp, cw, ncls):
    if name == "DecisionTree":
        return DecisionTreeClassifier(max_depth=hp["max_depth"], min_samples_leaf=hp["min_samples_leaf"],
                                      class_weight=cw, random_state=SEED)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
                                      min_samples_leaf=hp["min_samples_leaf"], class_weight=cw, n_jobs=-1, random_state=SEED)
    if name == "LogReg":
        return LogisticRegression(C=hp["C"], penalty="l2", class_weight=cw, max_iter=1000,
                                  solver="lbfgs", multi_class="multinomial", random_state=SEED, n_jobs=-1)
    if name == "XGBoost":      # objective/num_class은 sklearn 래퍼가 자동 추론(multi:softprob)
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=N_BOOST, max_depth=hp["max_depth"], learning_rate=hp["lr"],
                             subsample=hp["subsample"], colsample_bytree=hp["colsample"], tree_method="hist",
                             max_bin=MAX_BIN, eval_metric="mlogloss",
                             early_stopping_rounds=ES, random_state=SEED, n_jobs=-1)
    if name == "LightGBM":     # objective/num_class 자동 추론
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=N_BOOST, num_leaves=hp["num_leaves"], learning_rate=hp["lr"],
                              min_child_samples=hp["min_child_samples"], subsample=hp["subsample"], subsample_freq=1,
                              colsample_bytree=hp["colsample"], max_bin=MAX_BIN,
                              class_weight=cw, force_col_wise=True, random_state=SEED, n_jobs=-1, verbose=-1)
    if name == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=min(N_BOOST, 40), depth=hp["depth"], learning_rate=hp["lr"],
                                  l2_leaf_reg=hp["l2_leaf_reg"], loss_function="MultiClass", border_count=MAX_BIN,
                                  od_type="Iter", od_wait=ES, auto_class_weights="Balanced" if cw else None,
                                  random_state=SEED, verbose=0, thread_count=-1)
    raise ValueError(name)


def fit_es(name, model, Xtr, ytr, Xval, yval):
    """boosting은 early stopping(val) 적용, 나머지는 일반 fit.
    ES val에는 train 폴드 미관측 라벨이 섞이면 라벨인코더가 깨지므로 **관측 라벨만 필터**."""
    if name in BOOST:
        seen = np.isin(yval, np.unique(ytr))
        Xval, yval = Xval[seen], yval[seen]
    if name == "XGBoost":
        model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    elif name == "LightGBM":
        import lightgbm as lgb
        model.fit(Xtr, ytr, eval_set=[(Xval, yval)],
                  callbacks=[lgb.early_stopping(ES, verbose=False), lgb.log_evaluation(0)])
    elif name == "CatBoost":
        model.fit(Xtr, ytr, eval_set=(Xval, yval), verbose=False)
    else:
        model.fit(Xtr, ytr)
    return model


def sample_hp(t, name):
    if name == "DecisionTree":
        return {"max_depth": t.suggest_int("max_depth", 4, 24), "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 50)}
    if name == "RandomForest":
        return {"n_estimators": t.suggest_int("n_estimators", 100, 300), "max_depth": t.suggest_int("max_depth", 6, 28),
                "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 30)}
    if name == "LogReg":
        return {"C": t.suggest_float("C", 1e-2, 1e2, log=True)}
    if name == "XGBoost":      # n_estimators 없음(ES가 결정)
        return {"max_depth": t.suggest_int("max_depth", 4, 10), "lr": t.suggest_float("lr", 0.03, 0.3, log=True),
                "subsample": t.suggest_float("subsample", 0.7, 1.0), "colsample": t.suggest_float("colsample", 0.6, 1.0)}
    if name == "LightGBM":     # 다중분류 속도: 작은 트리(num_leaves↓)·높은 lr로 빠르게 수렴
        return {"num_leaves": t.suggest_int("num_leaves", 15, 63), "lr": t.suggest_float("lr", 0.05, 0.3, log=True),
                "min_child_samples": t.suggest_int("min_child_samples", 10, 80), "subsample": t.suggest_float("subsample", 0.7, 1.0),
                "colsample": t.suggest_float("colsample", 0.6, 1.0)}
    if name == "CatBoost":
        return {"depth": t.suggest_int("depth", 4, 8), "lr": t.suggest_float("lr", 0.03, 0.3, log=True),
                "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1.0, 10.0)}
    return {}


def transform(Xfit, Xapply, prep):
    a, b = Xfit.copy(), Xapply.copy()
    if prep["log_counts"]:
        a[:, COUNT_IDX] = np.log1p(np.clip(a[:, COUNT_IDX], 0, None)); b[:, COUNT_IDX] = np.log1p(np.clip(b[:, COUNT_IDX], 0, None))
    sc = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}.get(prep["scaler"])
    if sc is not None:
        sc.fit(a); a, b = sc.transform(a), sc.transform(b)
    return a, b, sc


def topk(model, X, y, k=5):
    proba = model.predict_proba(X); cls = np.asarray(model.classes_)
    kk = min(k, proba.shape[1])
    idx = np.argpartition(-proba, kth=kk - 1, axis=1)[:, :kk]
    toplabels = cls[idx]
    return float(np.mean([y[i] in toplabels[i] for i in range(len(y))]))


def mrr(model, X, y):
    proba = model.predict_proba(X); cls = np.asarray(model.classes_)
    order = np.argsort(-proba, axis=1)
    rr = 0.0
    for i, yt in enumerate(y):
        rank = np.where(cls[order[i]] == yt)[0]
        if len(rank): rr += 1.0 / (rank[0] + 1)
    return float(rr / len(y))


def main():
    dskey = sys.argv[1] if len(sys.argv) > 1 else "cat"
    name = sys.argv[2] if len(sys.argv) > 2 else "LightGBM"
    ds = DSETS[dskey]
    DIR = os.path.join(HERE, "preprocessing_project", ds["dir"], "output")
    tr = pd.read_parquet(os.path.join(DIR, ds["train"])); te = pd.read_parquet(os.path.join(DIR, ds["test"]))
    # ★ 희소클래스 제거: train 빈도 ≥ MIN_COUNT만 유지 → 층화분할·XGBoost(0..n-1 연속) 안정 + 속도↑
    MIN_COUNT = 10
    vc = tr[ds["y"]].value_counts(); keep = set(vc[vc >= MIN_COUNT].index)
    tr = tr[tr[ds["y"]].isin(keep)]; te = te[te[ds["y"]].isin(keep)]
    # ★ y 라벨 인코딩(0..n-1). classes=원본 id 매핑(추론시 역변환).
    le = LabelEncoder().fit(tr[ds["y"]].values)
    classes = le.classes_.tolist(); ncls = len(classes)
    global N_BOOST
    N_BOOST = 50 if ncls > 400 else 70            # 다중분류 보팅 속도 우선: 트리상한 대폭↓(ES가 추가 단축)
    n_trials = int(sys.argv[3]) if len(sys.argv) > 3 else (6 if name in BOOST else 8)
    Xtr = np.nan_to_num(tr[FEAT_V2].values.astype(float)); Xte = np.nan_to_num(te[FEAT_V2].values.astype(float))
    ytr = le.transform(tr[ds["y"]].values); yte = le.transform(te[ds["y"]].values)
    mo = os.path.join(DIR, name); os.makedirs(mo, exist_ok=True)
    # 탐색용 단일 holdout(ES val 겸용) — 층화로 전 클래스가 train에 존재
    Xh, Xv, yh, yv = train_test_split(Xtr, ytr, test_size=0.2, random_state=SEED, stratify=ytr)
    print(f"[{dskey}/{name}] train {len(Xtr):,} / test {len(Xte):,} | {ncls}클래스 | {n_trials}trial | ES{ES} bin{MAX_BIN}", flush=True)
    trials = []

    def objective(t):
        prep = {"scaler": t.suggest_categorical("scaler", ["none", "standard", "robust"] if name in TREE_MODELS else ["standard", "minmax", "robust"]),
                "log_counts": t.suggest_categorical("log_counts", [True, False]),
                "imbalance": t.suggest_categorical("imbalance", ["none", "classweight"])}
        hp = sample_hp(t, name)
        cw = "balanced" if prep["imbalance"] == "classweight" else None
        Xa, Xb, _ = transform(Xh, Xv, prep)
        m = fit_es(name, make_model(name, hp, cw, ncls), Xa, yh, Xb, yv)
        s = topk(m, Xb, yv, k=5)
        trials.append({**prep, "top5_cv": round(s, 4)})
        return s

    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED)); study.optimize(objective, n_trials=n_trials)
    bp = study.best_params
    prep = {"scaler": bp["scaler"], "log_counts": bp["log_counts"], "imbalance": bp["imbalance"]}
    hp = {k: bp[k] for k in bp if k not in ("scaler", "log_counts", "imbalance")}
    cw = "balanced" if prep["imbalance"] == "classweight" else None

    # 최종: 전체 train으로 적합(boosting은 10% 내부 val로 ES) → test 평가
    Xfa, Xfb, scaler = transform(Xtr, Xte, prep)
    if name in BOOST:
        Xh2, Xv2, yh2, yv2 = train_test_split(Xfa, ytr, test_size=0.1, random_state=SEED, stratify=ytr)
        clf = fit_es(name, make_model(name, hp, cw, ncls), Xh2, yh2, Xv2, yv2)
    else:
        clf = make_model(name, hp, cw, ncls).fit(Xfa, ytr)
    t1 = topk(clf, Xfb, yte, k=1); t5 = topk(clf, Xfb, yte, k=5); mr = mrr(clf, Xfb, yte)

    def best_by(k):
        d = {}
        for r in trials: d[r[k]] = max(d.get(r[k], 0), r["top5_cv"])
        return d
    per_option = {k: best_by(k) for k in ["scaler", "log_counts", "imbalance"]}
    best_iter = int(getattr(clf, "best_iteration", None) or getattr(clf, "best_iteration_", None) or getattr(clf, "tree_count_", 0) or 0)
    metrics = {"val_top5": round(float(study.best_value), 4), "oot_top1": round(t1, 4), "oot_top5": round(t5, 4),
               "oot_mrr": round(mr, 4), "n_classes": ncls, "n_features": len(FEAT_V2), "best_iter_es": best_iter,
               "majority_top1": round(float((yte == pd.Series(ytr).mode()[0]).mean()), 4)}
    joblib.dump({"model_name": f"{name}_rec_{dskey}", "model_type": "tree" if name in TREE_MODELS else "linear",
                 "task": "multiclass_recommendation", "target": ds["y"], "feature_order": FEAT_V2,
                 "prep": prep, "hp": hp, "scaler": scaler, "classifier": clf, "classes": classes,
                 "metrics": metrics}, os.path.join(mo, f"prep_{name}_rec.joblib"))
    json.dump({"best_params": bp, "per_option_best": per_option, "metrics": metrics, "n_trials": n_trials},
              open(os.path.join(mo, f"{name}_rec_bayes.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    pd.DataFrame(Xfa, columns=FEAT_V2).assign(**{ds["y"]: ytr, "user_id": tr["user_id"].values}).to_parquet(
        os.path.join(mo, f"{name}_rec_train.parquet"), index=False)
    raw30 = tr[["user_id"] + FEAT_V2 + [ds["y"]]].head(30)
    with open(os.path.join(mo, f"{name}_first30.txt"), "w", encoding="utf-8") as f:
        f.write(f"# {name} 추천({dskey}) 전처리 — 앞 30행 | {ncls}클래스 {len(FEAT_V2)}피처\n")
        f.write(f"# 성능: top1 {t1:.4f} | top5 {t5:.4f} | MRR {mr:.4f} (majority top1 {metrics['majority_top1']:.4f}) | ES트리 {best_iter}\n")
        f.write(f"# 최적 전처리: scaler={prep['scaler']} log={prep['log_counts']} imbalance={prep['imbalance']}\n\n")
        f.write(raw30.to_string(index=False))
    print(f"RESULTREC\t{dskey}\t{name}\ttop1={t1:.4f}\ttop5={t5:.4f}\tMRR={mr:.4f}\tES{best_iter}\t{prep['scaler']}|{prep['imbalance']}\t({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

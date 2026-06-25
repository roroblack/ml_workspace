# -*- coding: utf-8 -*-
"""Online Retail 이탈 — 머신러닝 학습/평가 (정형 RFM 피처)."""
import os, sys, json
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT, MODELS = (os.path.join(HERE, d) for d in ("data", "outputs", "models"))
os.makedirs(MODELS, exist_ok=True)
SEED = 42


def get_split():
    df = pd.read_csv(os.path.join(DATA, "tabular.csv"))
    feat_cols = [c for c in df.columns if c not in ("customer_id", "churn")]
    X = df[feat_cols].astype(float).values
    y = df["churn"].values.astype(int)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
    np.savez(os.path.join(DATA, "split.npz"), tr=tr, te=te)   # DL과 동일 분할 공유
    sc = StandardScaler().fit(X[tr])
    return sc.transform(X[tr]), y[tr], sc.transform(X[te]), y[te], feat_cols


def ev(name, model, Xte, yte, cmap):
    proba = model.predict_proba(Xte)[:, 1]; pred = (proba >= 0.5).astype(int)
    m = {"accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred),
         "recall": recall_score(yte, pred), "f1": f1_score(yte, pred),
         "roc_auc": roc_auc_score(yte, proba)}
    cm = confusion_matrix(yte, pred)
    plt.figure(figsize=(3.6, 3.2)); plt.imshow(cm, cmap=cmap)
    for i in range(2):
        for j in range(2): plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xticks([0, 1], ["Stay", "Churn"]); plt.yticks([0, 1], ["Stay", "Churn"])
    plt.xlabel("Pred"); plt.ylabel("True"); plt.title(name); plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"cm_{name}.png"), dpi=110); plt.close()
    return m, proba


def main():
    Xtr, ytr, Xte, yte, feats = get_split()
    print(f"train {len(ytr)} / test {len(yte)} | 이탈률 train {ytr.mean():.3f}")
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=SEED),
        "GradientBoosting": GradientBoostingClassifier(random_state=SEED),
    }
    res = {}; plt.figure(figsize=(6, 5))
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr); m, proba = ev(name, mdl, Xte, yte, "Blues")
        res[name] = m; joblib.dump(mdl, os.path.join(MODELS, f"{name}.joblib"))
        fpr, tpr, _ = roc_curve(yte, proba); plt.plot(fpr, tpr, label=f"{name} ({m['roc_auc']:.3f})")
        print(f"[{name}] acc {m['accuracy']:.3f} recall {m['recall']:.3f} f1 {m['f1']:.3f} AUC {m['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=.5); plt.legend(); plt.grid(alpha=.3)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC - ML"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "roc_ml.png"), dpi=110); plt.close()
    # 피처 중요도
    rf = models["RandomForest"]
    imp = pd.Series(rf.feature_importances_, index=feats).sort_values()
    plt.figure(figsize=(7, 5)); imp.plot(kind="barh"); plt.title("Feature Importance (RF)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "feature_importance.png"), dpi=110); plt.close()
    res["_feature_importance"] = imp.sort_values(ascending=False).round(4).to_dict()
    json.dump(res, open(os.path.join(OUT, "metrics_ml.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("ML 완료 → outputs/metrics_ml.json")


if __name__ == "__main__":
    main()

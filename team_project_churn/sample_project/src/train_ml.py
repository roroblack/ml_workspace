# -*- coding: utf-8 -*-
"""정형 ML 학습(LogReg/GBM) → 최적 모델·스케일러 저장 + DB model_registry 등록."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score
import config
from db import db_client as db

FEAT_COLS = ["recency_days", "n_view", "n_cart", "n_purchase", "n_events", "active_days", "avg_price"]


def main():
    df = pd.read_csv(os.path.join(config.PROC, "features.csv"))
    X = df[FEAT_COLS].values.astype(float); y = df["churn"].values.astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=config.SEED, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    models = {
        "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.SEED),
        "GBM": GradientBoostingClassifier(random_state=config.SEED),
    }
    best, best_auc, metrics = None, -1, {}
    for name, m in models.items():
        m.fit(Xtr_s, ytr); p = m.predict_proba(Xte_s)[:, 1]
        auc = roc_auc_score(yte, p)
        metrics[name] = {"auc": round(float(auc), 4), "f1": round(float(f1_score(yte, (p >= .5))), 4),
                         "recall": round(float(recall_score(yte, (p >= .5))), 4)}
        print(f"[train_ml] {name} AUC {auc:.3f}")
        if auc > best_auc:
            best_auc, best = auc, (name, m)

    name, model = best
    joblib.dump(model, os.path.join(config.MODELS, "tabular_model.joblib"))
    joblib.dump(sc, os.path.join(config.MODELS, "tabular_scaler.joblib"))
    json.dump({"feat_cols": FEAT_COLS, "best": name, "metrics": metrics},
              open(os.path.join(config.MODELS, "tabular_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    db.register_model(f"tabular_{name}", "tabular", os.path.join(config.MODELS, "tabular_model.joblib"),
                      metrics[name], active=True)
    print(f"[train_ml] 최적 {name} 등록 (AUC {best_auc:.3f})")


if __name__ == "__main__":
    main()

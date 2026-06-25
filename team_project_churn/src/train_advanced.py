# -*- coding: utf-8 -*-
"""
더 강한 모델 탐색 (Q2 실증)
============================
- HistGradientBoosting (sklearn 내장, 빠르고 강력)
- RandomizedSearchCV로 HGB/RF 경량 튜닝 (scoring=roc_auc, cv=3)
- SMOTE 오버샘플링 + HGB
- 임계값(threshold) 조정이 Recall에 주는 효과
처리된 데이터(data/processed)를 사용합니다.
"""
import os, sys, json
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (roc_auc_score, recall_score, precision_score,
                             f1_score, accuracy_score, precision_recall_curve)
from imblearn.over_sampling import SMOTE

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(HERE, "data", "processed")
SEED = 42

tr = pd.read_csv(os.path.join(PROC, "train.csv"))
te = pd.read_csv(os.path.join(PROC, "test.csv"))
Xtr, ytr = tr.drop(columns=["Churn"]).values, tr["Churn"].values
Xte, yte = te.drop(columns=["Churn"]).values, te["Churn"].values


def report(name, proba):
    pred = (proba >= 0.5).astype(int)
    return {"name": name, "accuracy": accuracy_score(yte, pred),
            "precision": precision_score(yte, pred), "recall": recall_score(yte, pred),
            "f1": f1_score(yte, pred), "roc_auc": roc_auc_score(yte, proba)}


rows = []

# 1) HistGradientBoosting 기본
hgb = HistGradientBoostingClassifier(random_state=SEED)
hgb.fit(Xtr, ytr)
rows.append(report("HistGB_default", hgb.predict_proba(Xte)[:, 1]))

# 2) HGB 경량 튜닝
hgb_space = {"max_depth": [None, 3, 5, 8], "learning_rate": [0.03, 0.05, 0.1],
             "max_iter": [200, 400], "l2_regularization": [0.0, 1.0],
             "max_leaf_nodes": [15, 31, 63]}
hgb_search = RandomizedSearchCV(HistGradientBoostingClassifier(random_state=SEED),
                                hgb_space, n_iter=20, scoring="roc_auc", cv=3,
                                random_state=SEED, n_jobs=-1)
hgb_search.fit(Xtr, ytr)
best_hgb = hgb_search.best_estimator_
proba_hgb = best_hgb.predict_proba(Xte)[:, 1]
rows.append(report("HistGB_tuned", proba_hgb))
print("HGB best params:", hgb_search.best_params_)

# 3) RandomForest 경량 튜닝
rf_space = {"n_estimators": [300, 500], "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 0.5]}
rf_search = RandomizedSearchCV(RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
                               rf_space, n_iter=15, scoring="roc_auc", cv=3,
                               random_state=SEED, n_jobs=-1)
rf_search.fit(Xtr, ytr)
rows.append(report("RF_tuned", rf_search.best_estimator_.predict_proba(Xte)[:, 1]))

# 4) SMOTE + HGB
sm = SMOTE(random_state=SEED)
Xsm, ysm = sm.fit_resample(Xtr, ytr)
hgb_sm = HistGradientBoostingClassifier(random_state=SEED, **{k: v for k, v in hgb_search.best_params_.items()})
hgb_sm.fit(Xsm, ysm)
rows.append(report("HistGB_tuned+SMOTE", hgb_sm.predict_proba(Xte)[:, 1]))

print("\n=== 테스트 성능 ===")
for r in sorted(rows, key=lambda x: -x["roc_auc"]):
    print(f"{r['name']:22s} acc {r['accuracy']:.3f} | recall {r['recall']:.3f} | "
          f"f1 {r['f1']:.3f} | AUC {r['roc_auc']:.3f}")

# 5) 최고 AUC 모델에서 임계값 조정 효과 (F1 최대 + recall 목표)
best = max(rows, key=lambda x: x["roc_auc"])
print(f"\n=== 임계값 조정 (모델: {best['name']}) ===")
prec, rec, thr = precision_recall_curve(yte, proba_hgb)
f1s = 2 * prec * rec / (prec + rec + 1e-9)
bi = np.nanargmax(f1s)
print(f"F1 최대 임계값 {thr[bi]:.2f} → precision {prec[bi]:.3f} recall {rec[bi]:.3f} f1 {f1s[bi]:.3f}")
for t in [0.5, 0.4, 0.3, 0.25]:
    p = (proba_hgb >= t).astype(int)
    print(f"  thr {t:.2f}: recall {recall_score(yte,p):.3f} precision {precision_score(yte,p):.3f} f1 {f1_score(yte,p):.3f}")

with open(os.path.join(HERE, "outputs", "metrics_advanced.json"), "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

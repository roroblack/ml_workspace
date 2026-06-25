# -*- coding: utf-8 -*-
"""
가입 고객 이탈 예측 — 머신러닝 모델 학습/평가
================================================
전처리된 데이터(data/processed)로 여러 ML 모델을 학습하고
이탈 예측에 중요한 지표(ROC-AUC, Recall, F1 등)로 평가합니다.

- 클래스 불균형 대응: class_weight="balanced" 사용
- 모델: 로지스틱 회귀 / 랜덤 포레스트 / 그래디언트 부스팅
- 산출물: outputs/metrics_ml.json, outputs/roc_ml.png,
          outputs/cm_<model>.png, outputs/feature_importance.png,
          models/<model>.joblib
"""
import os, json, sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
import joblib

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(HERE, "data", "processed")
MODELS = os.path.join(HERE, "models")
OUT = os.path.join(HERE, "outputs")
SEED = 42


def load():
    tr = pd.read_csv(os.path.join(PROC, "train.csv"))
    te = pd.read_csv(os.path.join(PROC, "test.csv"))
    Xtr, ytr = tr.drop(columns=["Churn"]).values, tr["Churn"].values
    Xte, yte = te.drop(columns=["Churn"]).values, te["Churn"].values
    return Xtr, ytr, Xte, yte, list(tr.drop(columns=["Churn"]).columns)


def evaluate(name, model, Xte, yte):
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    m = {
        "accuracy": accuracy_score(yte, pred),
        "precision": precision_score(yte, pred),
        "recall": recall_score(yte, pred),
        "f1": f1_score(yte, pred),
        "roc_auc": roc_auc_score(yte, proba),
    }
    cm = confusion_matrix(yte, pred)
    # 혼동행렬 저장
    plt.figure(figsize=(4, 3.5))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xticks([0, 1], ["Stay", "Churn"]); plt.yticks([0, 1], ["Stay", "Churn"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"cm_{name}.png"), dpi=110); plt.close()
    return m, proba


def main():
    Xtr, ytr, Xte, yte, feat_names = load()
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                               max_depth=None, n_jobs=-1, random_state=SEED),
        "GradientBoosting": GradientBoostingClassifier(random_state=SEED),
    }
    results = {}
    plt.figure(figsize=(6, 5))
    for name, model in models.items():
        model.fit(Xtr, ytr)
        m, proba = evaluate(name, model, Xte, yte)
        results[name] = m
        joblib.dump(model, os.path.join(MODELS, f"{name}.joblib"))
        fpr, tpr, _ = roc_curve(yte, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={m['roc_auc']:.3f})")
        print(f"[{name}] acc {m['accuracy']:.3f} | recall {m['recall']:.3f} | "
              f"f1 {m['f1']:.3f} | AUC {m['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - ML models"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "roc_ml.png"), dpi=110); plt.close()

    # 랜덤포레스트 피처 중요도
    rf = models["RandomForest"]
    imp = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=True)
    plt.figure(figsize=(7, 5))
    imp.plot(kind="barh")
    plt.title("Feature Importance (RandomForest)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "feature_importance.png"), dpi=110); plt.close()
    results["_feature_importance"] = imp.sort_values(ascending=False).round(4).to_dict()

    with open(os.path.join(OUT, "metrics_ml.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("ML 학습/평가 완료 → outputs/metrics_ml.json")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""SHAP 대안 — 모델 네이티브 피처 중요도(트리=feature_importances_, 선형=|coef|). numpy 충돌 없음.
시각화 13번(SHAP 자리)을 '피처 중요도'로 대체. 산출: output/evaluation/feature_importance.json
실행: python preprocessing_project/v4_model_prep/src/pp_feature_importance.py
"""
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, joblib

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
V4 = os.path.join(HERE, "preprocessing_project", "v4_model_prep", "output")
OUT = os.path.join(V4, "evaluation")
MODELS = ["DecisionTree", "RandomForest", "LogReg", "XGBoost", "LightGBM", "CatBoost"]


def get_clf(art):
    # v2 joblib: calibrator(CalibratedClassifierCV prefit) → 내부 pipe → clf
    cal = art.get("calibrator")
    try:
        est = cal.calibrated_classifiers_[0].estimator
        return est.named_steps["clf"]
    except Exception:
        pass
    if "pipeline" in art:
        return art["pipeline"].named_steps["clf"]
    return None


def main():
    FEAT = None
    out = {}
    for m in MODELS:
        jp = os.path.join(V4, m, f"prep_{m}_v2.joblib")
        if not os.path.exists(jp):
            continue
        art = joblib.load(jp); FEAT = art["feature_order"]
        clf = get_clf(art)
        if clf is None:
            continue
        if hasattr(clf, "feature_importances_"):
            imp = np.asarray(clf.feature_importances_, dtype=float)
        elif hasattr(clf, "coef_"):
            imp = np.abs(np.asarray(clf.coef_, dtype=float)).ravel()
        else:
            continue
        if imp.sum() > 0:
            imp = imp / imp.sum()
        order = np.argsort(-imp)[:10]
        out[m] = {FEAT[i]: round(float(imp[i]), 4) for i in order}
        print(f"  [{m}] top: {list(out[m].items())[:3]}")
    json.dump(out, open(os.path.join(OUT, "feature_importance.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[FI] {len(out)}모델 → {os.path.join(OUT,'feature_importance.json')}")


if __name__ == "__main__":
    main()

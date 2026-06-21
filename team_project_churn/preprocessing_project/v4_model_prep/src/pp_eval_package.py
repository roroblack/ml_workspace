# -*- coding: utf-8 -*-
"""검증 산출물 패키지 생성(발표 15시각화의 데이터원). 재학습 없이 v2 joblib로 test 예측만.
산출(output/evaluation/):
  eval_predictions.parquet  user_id·model_name·split·y_true·y_score·y_pred·cohort_flag·revenue·top_category·top_brand
  metrics_summary.json      모델별 AUC/PR-AUC/Brier/ECE/F1@thr/threshold (메인 비교화면)
  curves.json               모델별 roc/pr/threshold/calibration 곡선
실행: python preprocessing_project/v4_model_prep/src/pp_eval_package.py
"""
import os, sys, json, glob
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, joblib
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             roc_curve, precision_recall_curve, f1_score)

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
V4 = os.path.join(HERE, "preprocessing_project", "v4_model_prep", "output")
E = os.path.join(V4, "processed_eventbase")
OUT = os.path.join(V4, "evaluation"); os.makedirs(OUT, exist_ok=True)
TAB_MODELS = ["DecisionTree", "RandomForest", "LogReg", "XGBoost", "LightGBM", "CatBoost"]
FEAT_V2 = ["recency_days", "tenure_days", "ndays", "n_events", "n_view", "n_cart", "n_remove_from_cart",
           "n_purchase", "avg_price", "purch_amt", "min_price", "max_price", "std_price", "purchase_avg_price",
           "remove_ratio", "cart_purchase_ratio", "n_categories", "cat_entropy", "n_brands", "brand_loyalty",
           "n_sessions", "events_per_session"]
COUNTS = ["ndays", "n_events", "n_view", "n_cart", "n_remove_from_cart", "n_purchase", "purch_amt", "n_categories", "n_brands", "n_sessions"]
CIDX = [FEAT_V2.index(c) for c in COUNTS]


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p <= edges[i + 1]) if i == bins - 1 else (p >= edges[i]) & (p < edges[i + 1])
        if m.sum(): e += abs(p[m].mean() - y[m].mean()) * m.mean()
    return float(e)


def transform(X, art):
    a = X.copy()
    if art["prep"]["log_counts"]:
        a[:, CIDX] = np.log1p(np.clip(a[:, CIDX], 0, None))
    if art.get("scaler") is not None:
        a = art["scaler"].transform(a)
    return a


def calib_curve(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); xs, ys = [], []
    for i in range(bins):
        m = (p >= edges[i]) & (p <= edges[i + 1]) if i == bins - 1 else (p >= edges[i]) & (p < edges[i + 1])
        if m.sum() > 0:
            xs.append(round(float(p[m].mean()), 4)); ys.append(round(float(y[m].mean()), 4))
    return {"pred": xs, "true": ys}


def main():
    te = pd.read_parquet(os.path.join(E, "test_tabular_v2.parquet"))
    te = te[te.cohort_recency7 == 1].reset_index(drop=True)
    X = np.nan_to_num(te[FEAT_V2].values.astype(float)); X[:, CIDX] = np.clip(X[:, CIDX], 0, None)
    y = te["churn"].values.astype(int)
    rows, metrics, curves = [], {}, {}
    for m in TAB_MODELS:
        jp = os.path.join(V4, m, f"prep_{m}_v2.joblib")
        if not os.path.exists(jp): print(f"  [skip] {m} (joblib 없음)"); continue
        art = joblib.load(jp)
        p = art["calibrator"].predict_proba(transform(X, art))[:, 1]
        thr = float(art["threshold"]); yp = (p >= thr).astype(int)
        rows.append(pd.DataFrame({"user_id": te.user_id.values, "model_name": m, "split": "test",
                                  "y_true": y, "y_score": p, "y_pred": yp, "cohort_flag": 1,
                                  "revenue": te.purch_amt.values, "top_category": te.top_category_id.values,
                                  "top_brand": te.top_brand.values}))
        metrics[m] = {"auc": round(roc_auc_score(y, p), 4), "pr_auc": round(average_precision_score(y, p), 4),
                      "brier": round(brier_score_loss(y, p), 4), "ece": round(ece(y, p), 4),
                      "f1": round(f1_score(y, yp), 4), "threshold": round(thr, 3), "n": int(len(y)),
                      "base_rate": round(float(y.mean()), 4)}
        fpr, tpr, _ = roc_curve(y, p); pr, rc, _ = precision_recall_curve(y, p)
        ths = np.linspace(0.05, 0.95, 37)
        tcurve = [{"t": round(float(t), 3),
                   "precision": round(float(((yp2 := (p >= t)).sum() and (y[yp2].sum() / yp2.sum())) or 0), 4),
                   "recall": round(float(y[p >= t].sum() / max(y.sum(), 1)), 4),
                   "f1": round(float(f1_score(y, (p >= t).astype(int))), 4)} for t in ths]
        ds = max(1, len(fpr) // 100)
        curves[m] = {"roc": {"fpr": [round(float(v), 4) for v in fpr[::ds]], "tpr": [round(float(v), 4) for v in tpr[::ds]]},
                     "pr": {"recall": [round(float(v), 4) for v in rc[::max(1, len(rc)//100)]], "precision": [round(float(v), 4) for v in pr[::max(1, len(pr)//100)]]},
                     "threshold": tcurve, "calibration": calib_curve(y, p)}
        print(f"  [{m}] AUC {metrics[m]['auc']} PR {metrics[m]['pr_auc']} Brier {metrics[m]['brier']} ECE {metrics[m]['ece']} F1 {metrics[m]['f1']}")

    # Transformer(시퀀스)는 val 기준 요약만(길이상이로 Feb 직접 test 보류)
    tjson = os.path.join(V4, "Transformer", "transformer_meta.json")
    if os.path.exists(tjson):
        tm = json.load(open(tjson, encoding="utf-8")); metrics["Transformer"] = {"val_auc": tm.get("val_auc"), "note": "시퀀스 val 기준(길이정렬 후 Feb 예정)"}

    pd.concat(rows).to_parquet(os.path.join(OUT, "eval_predictions.parquet"), index=False)
    json.dump(metrics, open(os.path.join(OUT, "metrics_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(curves, open(os.path.join(OUT, "curves.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    # SHAP(있으면 트리계열 top피처)
    try:
        import shap
        sh = {}
        Xs = transform(X[:2000], joblib.load(os.path.join(V4, "CatBoost", "prep_CatBoost_v2.joblib")))
        for m in ["CatBoost", "LightGBM", "XGBoost", "RandomForest", "DecisionTree"]:
            art = joblib.load(os.path.join(V4, m, f"prep_{m}_v2.joblib")); clf = art["pipeline"].named_steps["clf"] if "pipeline" in art else None
            if clf is None: continue
            ex = shap.TreeExplainer(clf); sv = ex.shap_values(transform(X[:2000], art))
            sv = sv[1] if isinstance(sv, list) else sv
            imp = np.abs(sv).mean(0); sh[m] = {FEAT_V2[i]: round(float(imp[i]), 4) for i in np.argsort(-imp)[:10]}
        json.dump(sh, open(os.path.join(OUT, "shap_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("  [shap] shap_summary.json 생성")
    except Exception as ex:
        print(f"  [shap] 생략({type(ex).__name__}) — 설치 시 자동 생성")

    print(f"\n[eval] 산출 → {OUT} (eval_predictions {sum(len(r) for r in rows):,}행 · metrics · curves)")


if __name__ == "__main__":
    main()

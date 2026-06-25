# -*- coding: utf-8 -*-
"""진짜 SHAP(TreeSHAP) 생성 — 격리 numpy2 env 전용(26-8 §4). 트리 모델 per-prediction 기여 → 전역 mean|shap|.
CatBoost/LightGBM/XGBoost 번들에서 raw 트리모델 추출 → shap.TreeExplainer → shap_summary.json 덮어씀.
실행(별도 venv): pip install "numpy>=2" "shap>=0.47" catboost lightgbm xgboost scikit-learn pandas joblib
        python gen_real_shap.py
"""
import json, sys, warnings
from pathlib import Path
warnings.simplefilter("ignore")
import numpy as np, pandas as pd, joblib

HERE = Path(__file__).resolve().parents[4]   # ml_workspace (src→v4→preproc→team→ml_workspace)
MODELS = HERE / "SKN32-2nd_GAJIMA_Dev" / "models" / "preprocessors"
EVALCH = HERE / "SKN32-2nd_GAJIMA_Dev" / "data" / "processed" / "evaluation" / "churn"
SAMPLE = HERE / "SKN32-2nd_GAJIMA_Dev" / "data" / "processed" / "churn" / "test_tabular_v2.parquet"
TREE = {"CatBoost": "catboost", "LightGBM": "lightgbm", "XGBoost": "xgboost"}


def raw_clf(bundle):
    cal = bundle.get("calibrator")
    try:
        est = cal.calibrated_classifiers_[0].estimator
        return est.named_steps["clf"], est
    except Exception:
        return bundle.get("model"), None


def main():
    if not SAMPLE.exists():
        print("샘플 없음:", SAMPLE); return 1
    import shap
    feat_order = None
    Xs = pd.read_parquet(SAMPLE)
    for model, mkey in TREE.items():
        jp = MODELS / f"prep_{model}_v2.joblib"
        if not jp.exists():
            print("skip(no bundle):", model); continue
        b = joblib.load(jp); feat_order = b["feature_order"]
        clf, pipe = raw_clf(b)
        X = Xs.reindex(columns=feat_order).head(2000).fillna(0)
        # 파이프라인 전처리가 있으면 clf 입력공간으로 변환(스케일 등)
        try:
            if pipe is not None:
                pre = pipe[:-1]
                X = pd.DataFrame(pre.transform(X), columns=feat_order)
        except Exception:
            pass
        try:
            expl = shap.TreeExplainer(clf)
            sv = expl.shap_values(X)
            sv = sv[1] if isinstance(sv, list) and len(sv) == 2 else sv
            mab = np.abs(np.asarray(sv)).mean(axis=0).ravel()
            order = np.argsort(-mab)
            out = {"feature": [feat_order[i] for i in order],
                   "mean_abs_shap": [round(float(mab[i]), 4) for i in order],
                   "rank": list(range(1, len(order) + 1)), "method": "TreeSHAP"}
            d = EVALCH / mkey; d.mkdir(parents=True, exist_ok=True)
            (d / "shap_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  [OK] {model}: TreeSHAP top {out['feature'][:3]}")
        except Exception as e:
            print(f"  [X] {model}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

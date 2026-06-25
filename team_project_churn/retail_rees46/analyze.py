# -*- coding: utf-8 -*-
"""이탈 영향요인 분석 — 중요도 + 방향성(어떤 값이 이탈을 키우는가)."""
import os, sys, json
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT, MODELS = (os.path.join(HERE, d) for d in ("data", "outputs", "models"))
SEED = 42

df = pd.read_csv(os.path.join(DATA, "tabular.csv"))
feat = [c for c in df.columns if c not in ("customer_id", "user_id", "churn")]
X = df[feat].astype(float).values; y = df["churn"].values.astype(int)
sp = np.load(os.path.join(DATA, "split.npz")); tr, te = sp["tr"], sp["te"]
sc = StandardScaler().fit(X[tr]); Xte_s = sc.transform(X[te])

gbm = joblib.load(os.path.join(MODELS, "GradientBoosting.joblib"))

# 1) permutation importance (최적 모델 기준)
pi = permutation_importance(gbm, Xte_s, y[te], n_repeats=20, random_state=SEED, scoring="roc_auc")
perm = pd.Series(pi.importances_mean, index=feat).sort_values(ascending=False)

# 2) 방향성: 이탈/유지 그룹 원본값 평균 + 상관
raw = df[feat]
churn_mean = raw[df.churn == 1].mean(); stay_mean = raw[df.churn == 0].mean()
corr = raw.corrwith(df["churn"]).sort_values(ascending=False)

# 3) (가능 시) SHAP
shap_done = False
try:
    import shap
    expl = shap.TreeExplainer(gbm)
    sv = expl.shap_values(Xte_s)
    sv = sv[1] if isinstance(sv, list) else sv
    shap_imp = pd.Series(np.abs(sv).mean(0), index=feat).sort_values(ascending=False)
    plt.figure(figsize=(7, 5)); shap_imp.sort_values().plot(kind="barh")
    plt.title("Mean |SHAP| (GBM)"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "shap_importance.png"), dpi=110); plt.close()
    shap_done = True
except Exception as e:
    print("SHAP 생략:", str(e)[:60])

# 그래프: permutation importance
plt.figure(figsize=(7, 5)); perm.sort_values().plot(kind="barh")
plt.title("Permutation Importance (GBM, AUC)"); plt.tight_layout()
plt.savefig(os.path.join(OUT, "permutation_importance.png"), dpi=110); plt.close()

out = {
    "permutation_importance": perm.round(4).to_dict(),
    "corr_with_churn": corr.round(3).to_dict(),
    "group_mean_churn": churn_mean.round(2).to_dict(),
    "group_mean_stay": stay_mean.round(2).to_dict(),
    "shap_done": shap_done,
}
json.dump(out, open(os.path.join(OUT, "drivers.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print("=== 이탈 영향요인 (permutation importance 상위) ===")
for k, v in perm.head(8).items():
    direction = "이탈↑일수록 높음" if churn_mean[k] > stay_mean[k] else "이탈↑일수록 낮음"
    print(f"  {k:20s} imp {v:.4f} | 이탈평균 {churn_mean[k]:.1f} vs 유지평균 {stay_mean[k]:.1f} ({direction})")
print("분석 완료 → outputs/drivers.json")

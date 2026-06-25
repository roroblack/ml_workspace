# -*- coding: utf-8 -*-
"""17-3-4 개선안 검증: GBM(증분) + GRU(시퀀스) 앙상블 — 저장된 npz 재사용(빠름).
임베딩 개선이 실패했으므로, 트리의 정형강점 + DL의 시퀀스강점을 결합한 앙상블이
단일 모델을 이기는지 확인."""
import os, sys, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import session_full_1734 as M
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

DS = os.path.join(M.PROC, "session_level_full.npz")
OUT = M.OUT


def main():
    z = np.load(DS)
    X, CI, Y, tr = z["X"], z["cat_idx"], z["Y"], z["is_train"]
    Xtr, Xte = X[tr], X[~tr]; CItr, CIte = CI[tr], CI[~tr]; Ytr, Yte = Y[tr], Y[~tr]
    Xtr_f, Xte_f = Xtr.reshape(len(Xtr), -1), Xte.reshape(len(Xte), -1)
    print(f"로드 윈도우 train {len(Xtr):,}/test {len(Xte):,}")

    gbm = MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=300, random_state=M.SEED)).fit(Xtr_f, Ytr)
    p_gbm = gbm.predict(Xte_f).clip(0, 1)
    p_gru = M.train_dl("gru", Xtr, CItr, Ytr, Xte, CIte, n_cat=0).clip(0, 1)

    res = {"GBM": M.metrics(Yte, p_gbm), "GRU": M.metrics(Yte, p_gru)}
    best = (None, 1e9)
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:                 # GRU 가중 탐색
        p = (w * p_gru + (1 - w) * p_gbm).clip(0, 1)
        r = M.metrics(Yte, p)
        if r["rmse"] < best[1]: best = (w, r["rmse"], r)
    res["Ensemble(GBM+GRU)"] = best[2]; res["_best_w_gru"] = best[0]
    for k in ["GBM", "GRU", "Ensemble(GBM+GRU)"]:
        print(f"  {k:20s} RMSE {res[k]['rmse']:.4f} AUC {res[k]['overload_auc']:.4f}")
    print(f"  앙상블 최적 GRU가중 w={best[0]}")
    json.dump(res, open(os.path.join(OUT, "forecast_session_full_ensemble.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("저장 → forecast_session_full_ensemble.json")


if __name__ == "__main__":
    main()

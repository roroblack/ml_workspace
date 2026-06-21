# -*- coding: utf-8 -*-
"""17-5 최적본 저장: 베이지안으로 찾은 ML/DL 각각의 최적 전처리를 적용해 별도 파일로 저장.
ML 최적: log1p(counts)+minmax (+SMOTE는 학습시 적용). DL 최적: 정규화 없음(raw).
"""
import os, sys, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, joblib
from sklearn.preprocessing import MinMaxScaler

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # repo root(team_project_churn)
D = os.path.join(HERE, "sample_project", "data", "processed_5m")
FEAT = ["recency_days", "tenure_days", "ndays", "n_events", "n_view", "n_cart",
        "n_remove_from_cart", "n_purchase", "avg_price", "purch_amt"]
COUNT_IDX = [2, 3, 4, 5, 6, 7, 9]


def main():
    tr = pd.read_parquet(os.path.join(D, "train_cohort_tabular.parquet"))
    te = pd.read_parquet(os.path.join(D, "test_cohort_tabular.parquet"))
    Xtr = np.nan_to_num(tr[FEAT].values.astype(float)); Xte = np.nan_to_num(te[FEAT].values.astype(float))
    Xtr[:, COUNT_IDX] = np.clip(Xtr[:, COUNT_IDX], 0, None); Xte[:, COUNT_IDX] = np.clip(Xte[:, COUNT_IDX], 0, None)

    # === ML 최적: log1p(counts) + minmax ===
    Xtr_m = Xtr.copy(); Xte_m = Xte.copy()
    Xtr_m[:, COUNT_IDX] = np.log1p(Xtr_m[:, COUNT_IDX]); Xte_m[:, COUNT_IDX] = np.log1p(Xte_m[:, COUNT_IDX])
    sc = MinMaxScaler().fit(Xtr_m)
    Xtr_m, Xte_m = sc.transform(Xtr_m), sc.transform(Xte_m)
    pd.DataFrame(Xtr_m, columns=FEAT).assign(churn=tr["churn"].values, user_id=tr["user_id"].values
        ).to_parquet(os.path.join(D, "train_tabular_ML_opt.parquet"), index=False)
    pd.DataFrame(Xte_m, columns=FEAT).assign(churn=te["churn"].values, user_id=te["user_id"].values
        ).to_parquet(os.path.join(D, "test_tabular_ML_opt.parquet"), index=False)
    joblib.dump(sc, os.path.join(D, "ML_opt_scaler.joblib"))

    # === DL 최적: 정규화 없음(raw) — 코호트 최근4주 시퀀스 그대로 ===
    z = np.load(os.path.join(D, "train_seq.npz")); zt = np.load(os.path.join(D, "test_seq.npz"))
    tr_ids = set(tr["user_id"]); te_ids = set(te["user_id"])
    s1 = np.isin(z["user_id"], list(tr_ids)); s2 = np.isin(zt["user_id"], list(te_ids))
    np.savez_compressed(os.path.join(D, "train_seq_DL_opt.npz"),
                        X=z["X"][s1][:, -4:, :], churn=z["churn"][s1], user_id=z["user_id"][s1])
    np.savez_compressed(os.path.join(D, "test_seq_DL_opt.npz"),
                        X=zt["X"][s2][:, -4:, :], churn=zt["churn"][s2], user_id=zt["user_id"][s2])

    sizes = {f: round(os.path.getsize(os.path.join(D, f)) / 1024, 1) for f in
             ["train_tabular_ML_opt.parquet", "test_tabular_ML_opt.parquet", "ML_opt_scaler.joblib",
              "train_seq_DL_opt.npz", "test_seq_DL_opt.npz"]}
    print("최적본 저장(processed_5m):")
    for f, kb in sizes.items(): print(f"  {f:34s} {kb:8.1f} KB")


if __name__ == "__main__":
    main()

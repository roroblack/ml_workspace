# -*- coding: utf-8 -*-
"""churn_no_purchase 복원(배포본 일관성). 결과창 구매유저만 다시 읽어 정확 재계산 후
processed_5m의 테이블/시퀀스에 컬럼/키를 도로 추가."""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # repo root(team_project_churn)
SRC = os.path.join(HERE, "src"); D = os.path.join(HERE, "sample_project", "data", "processed_5m")
DT = {"event_type": "category", "user_id": "int64"}


def purch_set(zipf, c0, c1):
    df = pd.read_csv(os.path.join(SRC, zipf), usecols=["event_time", "event_type", "user_id"], dtype=DT)
    df["event_time"] = pd.to_datetime(df["event_time"], format="%Y-%m-%d %H:%M:%S UTC", errors="coerce")
    m = (df.event_time >= pd.Timestamp(c0)) & (df.event_time < pd.Timestamp(c1)) & (df.event_type == "purchase")
    return set(df.loc[m, "user_id"].unique())


def main():
    tr_p = purch_set("2020-Jan.csv.zip", "2020-01-25", "2020-02-01")   # train 결과창 구매
    te_p = purch_set("2020-Feb.csv.zip", "2020-02-22", "2020-03-01")   # test 결과창 구매
    print(f"결과창 구매유저: train {len(tr_p):,} / test {len(te_p):,}", flush=True)

    for f, ps in [("train_tabular.parquet", tr_p), ("test_tabular.parquet", te_p),
                  ("train_cohort_tabular.parquet", tr_p), ("test_cohort_tabular.parquet", te_p)]:
        p = os.path.join(D, f)
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p)
        d["churn_no_purchase"] = (~d["user_id"].isin(ps)).astype("int32")
        d.to_parquet(p, index=False)
        print(f"  {f}: churn_no_purchase 복원 ({d.churn_no_purchase.mean()*100:.1f}%)", flush=True)

    for f, ps in [("train_seq.npz", tr_p), ("test_seq.npz", te_p)]:
        p = os.path.join(D, f)
        if not os.path.exists(p):
            continue
        z = np.load(p)
        cnp = (~np.isin(z["user_id"], list(ps))).astype("int32")
        np.savez_compressed(p, X=z["X"], user_id=z["user_id"], churn=z["churn"], churn_np=cnp)
        print(f"  {f}: churn_np 복원", flush=True)


if __name__ == "__main__":
    main()

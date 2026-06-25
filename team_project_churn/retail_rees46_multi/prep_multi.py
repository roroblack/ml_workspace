# -*- coding: utf-8 -*-
"""
REES46 화장품 — 5개월 전체 연결 + 다개월 시계열 라벨링
=======================================================
관찰기간: 2019-10-01 ~ 2020-01-31 (4개월)  → 피처/시퀀스
결과기간: 2020-02-01 ~ 2020-02-28 (1개월)  → 이 기간 활동 없음 = 이탈(1)
  (긴 결과창 + 긴 관찰 → recency로만 환원되지 않는 라벨, 시퀀스 DL에 유리한 조건)
시퀀스: 주별(약 17주) view/cart/purchase 횟수.
산출물: data/{tabular.csv, seq.npz, split.npz}, outputs/label_summary.json
"""
import os, sys, glob, json
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")           # 월별 zip 위치
DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "outputs")
os.makedirs(DATA, exist_ok=True); os.makedirs(OUT, exist_ok=True)

SEED = 42
CUT = pd.Timestamp("2020-02-01")                # 관찰/결과 경계
OBS_START = pd.Timestamp("2019-10-01")
USER_CAP = 80000
EVENTS = ["view", "cart", "purchase"]
USECOLS = ["event_time", "event_type", "product_id", "brand", "price", "user_id"]


def load_all():
    # 파일별로 파싱·다운캐스트 후 concat (메모리 절약). date는 나중에 표본에만 부여.
    parts = []
    for z in sorted(glob.glob(os.path.join(SRC, "2019-*.csv.zip")) + glob.glob(os.path.join(SRC, "2020-*.csv.zip"))):
        print("  읽는 중:", os.path.basename(z), flush=True)
        d = pd.read_csv(z, usecols=USECOLS)
        # "2019-11-01 00:00:02 UTC" → format 지정(빠르고 str.replace 불필요)
        d["event_time"] = pd.to_datetime(d["event_time"], format="%Y-%m-%d %H:%M:%S UTC", errors="coerce")
        d = d.dropna(subset=["event_time", "user_id"])
        d["user_id"] = d["user_id"].astype(np.int64)
        d["product_id"] = d["product_id"].astype(np.int32)
        d["price"] = d["price"].astype(np.float32)
        d["event_type"] = d["event_type"].astype("category")
        d["brand"] = d["brand"].astype("category")
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def main():
    print("5개월 로딩...", flush=True)
    df = load_all()
    print(f"  전체 이벤트 {len(df):,}", flush=True)
    obs = df[df["event_time"] < CUT]
    out = df[df["event_time"] >= CUT]

    pop = obs["user_id"].unique()
    rng = np.random.RandomState(SEED)
    if len(pop) > USER_CAP:
        pop = rng.choice(pop, USER_CAP, replace=False)
    pop = np.sort(pop)
    out_users = set(out["user_id"].unique())
    churn = np.array([0 if u in out_users else 1 for u in pop], dtype=int)

    obs = obs[obs["user_id"].isin(pop)].copy()     # 표본만 남겨 가속
    obs["date"] = obs["event_time"].dt.normalize()  # 무거운 연산은 표본에만

    # ---- 정형 피처 ----
    g = obs.groupby("user_id")
    f = pd.DataFrame(index=pop); f.index.name = "user_id"
    last = g["event_time"].max(); first = g["event_time"].min()
    f["recency_days"] = (CUT - last).dt.total_seconds() / 86400
    f["tenure_days"] = (last - first).dt.total_seconds() / 86400
    f["active_days"] = g["date"].nunique()
    f["n_events"] = g.size()
    for et in EVENTS:
        f[f"n_{et}"] = obs[obs["event_type"] == et].groupby("user_id").size().reindex(pop).fillna(0).values
    f["n_unique_products"] = g["product_id"].nunique()
    f["n_unique_brands"] = g["brand"].nunique()
    f["avg_price"] = g["price"].mean()
    f["view2purchase"] = (f["n_purchase"] / f["n_view"].replace(0, np.nan)).fillna(0)
    f["active_weeks"] = (g["date"].apply(lambda s: ((s - OBS_START).dt.days // 7).nunique()))
    f = f.fillna(0); f["churn"] = churn
    f.reset_index().to_csv(os.path.join(DATA, "tabular.csv"), index=False)

    # ---- 주별 시퀀스 [N, W, 3] ----
    obs = obs.assign(week=((obs["date"] - OBS_START).dt.days // 7).astype(int))
    n_weeks = int(((CUT - OBS_START).days - 1) // 7) + 1
    uidx = {u: i for i, u in enumerate(pop)}
    X = np.zeros((len(pop), n_weeks, len(EVENTS)), dtype=np.float32)
    wk = obs.groupby(["user_id", "week", "event_type"]).size()
    for (u, w, et), c in wk.items():
        if et in EVENTS and 0 <= w < n_weeks and u in uidx:
            X[uidx[u], w, EVENTS.index(et)] = c
    np.savez(os.path.join(DATA, "seq.npz"), X=X, y=churn, user_id=pop)

    idx = np.arange(len(churn))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=churn)
    np.savez(os.path.join(DATA, "split.npz"), tr=tr, te=te)

    summary = {
        "files": "5 months (2019-10 ~ 2020-02)", "total_events": int(len(df)),
        "obs_period": ["2019-10-01", "2020-01-31"], "outcome_period": ["2020-02-01", "2020-02-28"],
        "population": int(len(pop)), "churn_count": int(churn.sum()),
        "churn_rate": round(float(churn.mean()), 4),
        "tabular_features": [c for c in f.columns if c not in ("churn",)],
        "seq_shape": list(X.shape), "n_weeks": n_weeks, "seq_feature_order": EVENTS,
    }
    json.dump(summary, open(os.path.join(OUT, "label_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"완료 | 모집단 {len(pop)} | 이탈률 {summary['churn_rate']*100:.2f}% | "
          f"정형 {len(summary['tabular_features'])-1}피처 | 시퀀스 {X.shape}")


if __name__ == "__main__":
    main()

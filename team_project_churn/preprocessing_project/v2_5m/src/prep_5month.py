# -*- coding: utf-8 -*-
"""17-5 Phase1. 5개월 REES46 풀데이터 → 이탈 라벨 테이블(ML) + 주별 시퀀스(DL).
- 라벨: 결과 7일 무활동=churn / 7일 무purchase=churn_no_purchase
- 기간: train(Oct-Jan, 결과 01-25~31) / test(Feb, 결과 02-22~28)
- 풀데이터(샘플링 없음). 월별 순차 처리로 메모리 안전.
저장: sample_project/data/processed_5m/{train,test}_tabular.parquet, {train,test}_seq.npz, meta_5m.json
"""
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # repo root(team_project_churn)
SRC = os.path.join(HERE, "src")
OUT = os.path.join(HERE, "sample_project", "data", "processed_5m"); os.makedirs(OUT, exist_ok=True)
MONTHS = ["2019-Oct.csv.zip", "2019-Nov.csv.zip", "2019-Dec.csv.zip", "2020-Jan.csv.zip", "2020-Feb.csv.zip"]
EVENTS = ["view", "cart", "purchase"]          # 시퀀스 피처(canonical과 동일)
ALLTYPES = ["view", "cart", "remove_from_cart", "purchase"]
DT = {"event_type": "category", "price": "float32", "user_id": "int64"}

BLOCKS = {
    "train": {"obs": ("2019-10-01", "2020-01-25"), "out": ("2020-01-25", "2020-02-01"), "wk0": "2019-10-01"},
    "test":  {"obs": ("2020-02-01", "2020-02-22"), "out": ("2020-02-22", "2020-03-01"), "wk0": "2020-02-01"},
}
for b in BLOCKS.values():
    b["obs"] = (pd.Timestamp(b["obs"][0]), pd.Timestamp(b["obs"][1]))
    b["out"] = (pd.Timestamp(b["out"][0]), pd.Timestamp(b["out"][1]))
    b["wk0"] = pd.Timestamp(b["wk0"])
NWK = {k: int((v["obs"][1] - v["wk0"]).days // 7) + 1 for k, v in BLOCKS.items()}  # train≈17, test=3


def agg_tabular(obs):
    """관찰 이벤트 → 유저별 부분집계(월별 합산 가능한 형태)."""
    obs = obs.copy(); obs["date"] = obs["event_time"].dt.normalize()
    g = obs.groupby("user_id")
    part = g.agg(n_events=("price", "size"), sum_price=("price", "sum"),
                 max_time=("event_time", "max"), min_time=("event_time", "min"),
                 ndays=("date", "nunique"))
    for t in ALLTYPES:
        part[f"n_{t}"] = obs[obs.event_type == t].groupby("user_id").size().reindex(part.index).fillna(0).astype("int64")
    part["purch_amt"] = obs[obs.event_type == "purchase"].groupby("user_id")["price"].sum().reindex(part.index).fillna(0)
    return part


def agg_weekly(obs, wk0):
    obs = obs[obs.event_type.isin(EVENTS)].copy()
    obs["week"] = ((obs["event_time"].dt.normalize() - wk0).dt.days // 7).astype(int)
    obs["et"] = obs["event_type"].astype(str)
    w = obs.groupby(["user_id", "week", "et"]).size().reset_index(name="c")
    return w


def main():
    tab_parts = {"train": [], "test": []}
    seq_parts = {"train": [], "test": []}
    out_active = {"train": [], "test": []}
    out_purch = {"train": [], "test": []}

    for m in MONTHS:
        t0 = time.time()
        df = pd.read_csv(os.path.join(SRC, m), usecols=list(DT) + ["event_time"], dtype=DT)
        df["event_time"] = pd.to_datetime(df["event_time"], format="%Y-%m-%d %H:%M:%S UTC", errors="coerce")
        df = df.dropna(subset=["event_time", "user_id"])
        for blk, cfg in BLOCKS.items():
            o0, o1 = cfg["obs"]; c0, c1 = cfg["out"]
            obs = df[(df.event_time >= o0) & (df.event_time < o1)]
            if len(obs):
                tab_parts[blk].append(agg_tabular(obs))
                seq_parts[blk].append(agg_weekly(obs, cfg["wk0"]))
            out = df[(df.event_time >= c0) & (df.event_time < c1)]
            if len(out):
                out_active[blk].append(out["user_id"].unique())
                out_purch[blk].append(out[out.event_type == "purchase"]["user_id"].unique())
        print(f"  [{m}] {len(df):,}행 처리 ({time.time()-t0:.0f}s)", flush=True)
        del df

    meta = {"periods": {k: [str(v["obs"][0].date()), str(v["obs"][1].date()), "out7d",
                            str(v["out"][0].date()), str(v["out"][1].date())] for k, v in BLOCKS.items()},
            "n_weeks": NWK, "datasets": {}}

    for blk in ["train", "test"]:
        # ---- 정형 테이블 ----
        tab = pd.concat(tab_parts[blk])
        agg = {c: "sum" for c in tab.columns if c not in ("max_time", "min_time")}
        agg["max_time"] = "max"; agg["min_time"] = "min"
        tab = tab.groupby(level=0).agg(agg)
        obs_end = BLOCKS[blk]["obs"][1]
        tab["recency_days"] = (obs_end - tab["max_time"]).dt.total_seconds() / 86400
        tab["tenure_days"] = (tab["max_time"] - tab["min_time"]).dt.total_seconds() / 86400
        tab["avg_price"] = tab["sum_price"] / tab["n_events"].clip(lower=1)
        active = set(np.concatenate(out_active[blk])) if out_active[blk] else set()
        purch = set(np.concatenate(out_purch[blk])) if out_purch[blk] else set()
        idx = tab.index.to_numpy()
        tab["churn"] = (~np.isin(idx, list(active))).astype(int)            # 결과창 7일 무활동=이탈(주 라벨)
        tab["churn_no_purchase"] = (~np.isin(idx, list(purch))).astype(int)  # 보조: 7일 무구매(배포본 호환 유지)
        feat_cols = ["recency_days", "tenure_days", "ndays", "n_events", "n_view", "n_cart",
                     "n_remove_from_cart", "n_purchase", "avg_price", "purch_amt"]
        out_tab = tab.reset_index()[["user_id"] + feat_cols + ["churn", "churn_no_purchase"]]
        out_tab.to_parquet(os.path.join(OUT, f"{blk}_tabular.parquet"), index=False)

        # ---- 주별 시퀀스 ----
        users = out_tab["user_id"].to_numpy()
        uidx = {u: i for i, u in enumerate(users)}
        nwk = NWK[blk]
        X = np.zeros((len(users), nwk, len(EVENTS)), np.float32)
        sl = pd.concat(seq_parts[blk]).groupby(["user_id", "week", "et"], as_index=False)["c"].sum()
        eidx = {e: i for i, e in enumerate(EVENTS)}
        for u, wk, et, c in sl.itertuples(index=False):
            if u in uidx and 0 <= wk < nwk and et in eidx:
                X[uidx[u], wk, eidx[et]] = c
        np.savez_compressed(os.path.join(OUT, f"{blk}_seq.npz"), X=X, user_id=users,
                            churn=out_tab["churn"].to_numpy(), churn_np=out_tab["churn_no_purchase"].to_numpy())

        meta["datasets"][blk] = {"n_users": int(len(out_tab)),
                                 "churn_rate": round(float(out_tab["churn"].mean()), 4),
                                 "churn_np_rate": round(float(out_tab["churn_no_purchase"].mean()), 4),
                                 "seq_shape": list(X.shape),
                                 "tabular_KB": round(os.path.getsize(os.path.join(OUT, f"{blk}_tabular.parquet")) / 1024, 1),
                                 "seq_KB": round(os.path.getsize(os.path.join(OUT, f"{blk}_seq.npz")) / 1024, 1)}
        print(f"[{blk}] 유저 {len(out_tab):,} | 이탈률 {out_tab.churn.mean()*100:.1f}% | "
              f"무구매이탈 {out_tab.churn_no_purchase.mean()*100:.1f}% | 시퀀스 {X.shape}", flush=True)

    meta["features"] = feat_cols
    json.dump(meta, open(os.path.join(OUT, "meta_5m.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("저장 → sample_project/data/processed_5m/ (train/test _tabular.parquet, _seq.npz, meta_5m.json)", flush=True)


if __name__ == "__main__":
    main()

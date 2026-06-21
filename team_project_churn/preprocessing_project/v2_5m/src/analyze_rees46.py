# -*- coding: utf-8 -*-
"""REES46 화장품 데이터셋 실측 분석 (벡터화·고속판).
월별 zip을 통째로(필요 컬럼만) 로드해 pandas 벡터연산으로 집계 → reports/_rees46_stats.json.
유저/세션 dict 파이썬 루프를 제거(groupby+describe)하여 빠름. 5개월 순차, 월마다 메모리 해제.
실행: C:\\Users\\playdata2\\anaconda3\\python.exe analyze_rees46.py [월...]
"""
import sys, os, json, zipfile, time, gc
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(HERE, "src")
OUT_JSON = os.path.join(HERE, "reports", "_rees46_stats.json")
MONTHS = ["2019-Oct", "2019-Nov", "2019-Dec", "2020-Jan", "2020-Feb"]
DT = {"event_type": "category", "product_id": "int64", "category_id": "int64",
      "category_code": "object", "brand": "object", "price": "float32",
      "user_id": "int64", "user_session": "object"}
USE = list(DT) + ["event_time"]


def dist(a):
    a = np.asarray(a, dtype=np.float64)
    if not len(a): return {}
    return {k: round(float(v), 2) for k, v in {
        "mean": a.mean(), "median": np.median(a), "p25": np.percentile(a, 25),
        "p75": np.percentile(a, 75), "p90": np.percentile(a, 90),
        "p99": np.percentile(a, 99), "max": a.max()}.items()}


def analyze_month(month):
    zp = os.path.join(SRC, f"{month}.csv.zip"); t0 = time.time()
    with zipfile.ZipFile(zp) as zf:
        with zf.open(f"{month}.csv") as f:
            df = pd.read_csv(f, usecols=USE, dtype=DT)
    df["event_time"] = pd.to_datetime(df["event_time"].str.slice(0, 19),
                                      format="%Y-%m-%d %H:%M:%S", errors="coerce")
    n = len(df)
    et = df["event_type"].astype(str)
    p = df["price"].astype("float64")
    g = df.groupby("user_id", sort=False)
    ev_per_user = g.size().to_numpy()
    purch = df.loc[et == "purchase", "user_id"]
    pcnt = purch.groupby(purch).size()
    date = df["event_time"].dt.normalize()
    active_days = df.assign(d=date).groupby("user_id")["d"].nunique().to_numpy()
    sess_len = df.groupby("user_session", sort=False).size().to_numpy()
    # 구매 간격(구매 서브셋만, 벡터화)
    pdf = df.loc[et == "purchase", ["user_id"]].assign(d=date[et == "purchase"]).drop_duplicates()
    pdf = pdf.sort_values(["user_id", "d"])
    gap = pdf.groupby("user_id")["d"].diff().dt.days.dropna()
    gap = gap[gap > 0].to_numpy()
    res = {
        "month": month, "rows": int(n),
        "event_time_min": str(df["event_time"].min()), "event_time_max": str(df["event_time"].max()),
        "event_type_counts": {k: int(v) for k, v in et.value_counts().items()},
        "brand_missing": int(df["brand"].isna().sum()), "brand_missing_pct": round(df["brand"].isna().mean() * 100, 1),
        "catcode_missing": int(df["category_code"].isna().sum()), "catcode_missing_pct": round(df["category_code"].isna().mean() * 100, 1),
        "price": {"mean": round(float(p.mean()), 2), "median": round(float(p.median()), 2),
                  "min": round(float(p.min()), 2), "max": round(float(p.max()), 2),
                  "std": round(float(p.std()), 2), "neg": int((p < 0).sum()), "zero": int((p == 0).sum())},
        "uniques": {"user_id": int(df["user_id"].nunique()), "user_session": int(df["user_session"].nunique()),
                    "product_id": int(df["product_id"].nunique()), "category_id": int(df["category_id"].nunique()),
                    "brand": int(df["brand"].nunique())},
        "dist_user_events": dist(ev_per_user), "dist_session_len": dist(sess_len),
        "dist_user_active_days": dist(active_days),
        "purchasers": int((pcnt >= 1).sum()), "repeat_purchasers": int((pcnt >= 2).sum()),
        "repeat_purchase_rate": round(float((pcnt >= 2).sum() / max((pcnt >= 1).sum(), 1)), 3),
        "purchase_gap_days": dist(gap) if len(gap) else {},
        "top_catcode": [[str(k), int(v)] for k, v in df["category_code"].value_counts().head(10).items()],
        "top_brand": [[str(k), int(v)] for k, v in df["brand"].value_counts().head(10).items()],
        "dow_counts": np.bincount(df["event_time"].dt.dayofweek.dropna().astype(int), minlength=7).tolist(),
        "hour_counts": np.bincount(df["event_time"].dt.hour.dropna().astype(int), minlength=24).tolist(),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    uu = df["user_id"].unique(); pu = df["product_id"].unique()
    del df, g; gc.collect()
    return res, uu, pu


def main():
    months = sys.argv[1:] or MONTHS
    all_res, gu, gp = [], [], []
    for m in months:
        sys.stderr.write(f"[*] {m} ...\n"); sys.stderr.flush()
        r, uu, pu = analyze_month(m); all_res.append(r); gu.append(uu); gp.append(pu)
        sys.stderr.write(f"    rows={r['rows']:,} users={r['uniques']['user_id']:,} {r['elapsed_sec']}s\n"); sys.stderr.flush()
    summary = {"months": all_res,
               "global_unique_users": int(len(np.unique(np.concatenate(gu)))) if gu else 0,
               "global_unique_products": int(len(np.unique(np.concatenate(gp)))) if gp else 0,
               "total_rows": int(sum(r["rows"] for r in all_res))}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(summary, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    sys.stderr.write(f"[OK] {OUT_JSON} (total_rows={summary['total_rows']:,}, users={summary['global_unique_users']:,})\n")


if __name__ == "__main__":
    main()

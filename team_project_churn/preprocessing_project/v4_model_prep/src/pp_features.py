# -*- coding: utf-8 -*-
"""v4 정식 전처리(raw→피처) — 기존 prep_5month이 '안 읽고 버린' 컬럼을 전부 살린다.
살리는 것: category_id, brand, product_id, user_session, remove_from_cart, price 분포.
산출: processed_eventbase/train_tabular_v2.parquet  (전체 모집단, 시간외삽 라벨)
설계 근거: reports/22(화이트페이퍼)·23(개선안). prep_5month(10피처)의 상위호환(추가 컬럼만).

[적용한 데이터 전처리 기술 — 코드에 주석으로 명시]
 (T1) dtype 다운캐스트(int32/float32/category) → 메모리 1/2~1/4
 (T2) 월별 순차 처리 → 2천만행 동시적재 회피(OOM 방지)
 (T3) datetime 파싱(문자열 'YYYY-MM-DD HH:MM:SS UTC' → datetime64) + 결측 제거
 (T4) 시간외삽 분할(관찰/결과) → 데이터 누수 차단
 (T5) 분산 누적식 표준편차(sum, sumsq, n) → 전체 std를 월별 합산으로 계산
 (T6) cross-month 고유개수: 월별 distinct (user,값) 쌍 누적 → 합쳐서 nunique
 (T7) 결측 처리: brand 결측 45% → 'UNK', category_code 98% 결측 → 미사용(category_id 사용)
 (T8) 고카디널리티 범주(category_id ~500, brand ~280): 인코딩은 모델 단계로 위임,
      여기선 원값(last_cat_id/top_brand) + 분포요약(entropy/loyalty)으로 보존
"""
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(HERE, "src")
OUT = os.path.join(HERE, "preprocessing_project", "v4_model_prep", "output", "processed_eventbase"); os.makedirs(OUT, exist_ok=True)
# (T1) dtype — ★category_id/product_id는 18자리(int32 초과) → int64 필수. user_id도 int64 안전. price float32
DT = {"event_type": "category", "product_id": "int64", "category_id": "int64",
      "brand": "category", "price": "float32", "user_id": "int64", "user_session": "category"}
USE = list(DT) + ["event_time"]                       # category_code는 98% 결측이라 의도적으로 미로딩(T7)
OBS = (pd.Timestamp("2019-10-01"), pd.Timestamp("2020-01-25"))   # (T4) 관찰=피처
OUTC = (pd.Timestamp("2020-01-25"), pd.Timestamp("2020-02-01"))  # (T4) 결과=라벨(7일)
MONTHS = ["2019-Oct", "2019-Nov", "2019-Dec", "2020-Jan"]


def process_block(tag, OBS, OUTC, MONTHS):
    import zipfile
    num_parts, cat_parts, brand_parts, sess_parts, last_parts = [], [], [], [], []
    out_active, out_purch = [], []
    t0 = time.time()
    for m in MONTHS:                                   # (T2) 월별 순차
        with zipfile.ZipFile(os.path.join(SRC, f"{m}.csv.zip")) as zf:
            df = pd.read_csv(zf.open(f"{m}.csv"), usecols=USE, dtype=DT)
        # (T3) datetime 파싱 + 결측 제거
        df["event_time"] = pd.to_datetime(df["event_time"].str.slice(0, 19), format="%Y-%m-%d %H:%M:%S", errors="coerce")
        df = df.dropna(subset=["event_time", "user_id"])
        obs = df[(df.event_time >= OBS[0]) & (df.event_time < OBS[1])].copy()
        out = df[(df.event_time >= OUTC[0]) & (df.event_time < OUTC[1])]
        if len(out):
            out_active.append(out["user_id"].unique())
            out_purch.append(out.loc[out.event_type == "purchase", "user_id"].unique())
        if not len(obs):
            del df; continue
        obs["date"] = obs["event_time"].dt.normalize()
        et = obs["event_type"].astype(str)
        p = obs["price"].astype("float64")
        g = obs.groupby("user_id", sort=False)
        # 숫자 부분집계 (월별 합산 가능; T5용 sum/sumsq 포함)  ※ price·remove_from_cart 살림
        part = pd.DataFrame({
            "n_events": g.size(),
            "sum_price": p.groupby(obs.user_id).sum(), "sumsq_price": (p * p).groupby(obs.user_id).sum(),
            "min_price": p.groupby(obs.user_id).min(), "max_price": p.groupby(obs.user_id).max(),
            "max_time": g["event_time"].max(), "min_time": g["event_time"].min(),
            "ndays": g["date"].nunique(),
        })
        # 이벤트타입 카운트: 4회 마스킹 groupby → 1회 groupby+unstack(벡터화)
        tc = obs.assign(_et=et).groupby(["user_id", "_et"]).size().unstack(fill_value=0).reindex(part.index, fill_value=0)
        for t in ["view", "cart", "remove_from_cart", "purchase"]:    # remove_from_cart 포함
            part[f"n_{t}"] = tc[t].astype("int32") if t in tc.columns else np.int32(0)
        part["purch_amt"] = obs.loc[et == "purchase", "price"].groupby(obs.loc[et == "purchase"].user_id).sum().reindex(part.index).fillna(0)
        num_parts.append(part)
        # (T6) cross-month 고유쌍: category/brand/session  ※ category/brand 살림
        cat_parts.append(obs.groupby(["user_id", "category_id"]).size().rename("c").reset_index())
        brand_parts.append(obs.assign(brand=obs.brand.astype("object").fillna("UNK")).groupby(["user_id", "brand"]).size().rename("c").reset_index())
        sess_parts.append(obs[["user_id", "user_session"]].drop_duplicates())
        # last_cat/last_brand: 월별 유저-마지막 이벤트
        li = g["event_time"].idxmax()
        lp = obs.loc[li, ["user_id", "event_time", "category_id", "brand"]].copy()
        lp["brand"] = lp["brand"].astype("object").fillna("UNK")          # 결측 brand → UNK
        last_parts.append(lp)
        print(f"  [{m}] obs {len(obs):,}행 처리 ({time.time()-t0:.0f}s)", flush=True)
        del df, obs

    # ---- 숫자 피처 합치기 ----
    tab = pd.concat(num_parts)
    agg = {c: "sum" for c in ["n_events", "sum_price", "sumsq_price", "ndays", "n_view", "n_cart", "n_remove_from_cart", "n_purchase", "purch_amt"]}
    agg.update({"min_price": "min", "max_price": "max", "max_time": "max", "min_time": "min"})
    tab = tab.groupby(level=0).agg(agg)
    tab["recency_days"] = (OBS[1] - tab["max_time"]).dt.total_seconds() / 86400
    tab["tenure_days"] = (tab["max_time"] - tab["min_time"]).dt.total_seconds() / 86400
    tab["avg_price"] = tab["sum_price"] / tab["n_events"].clip(lower=1)
    # (T5) 분산누적식 표준편차
    mean = tab["sum_price"] / tab["n_events"].clip(lower=1)
    tab["std_price"] = np.sqrt((tab["sumsq_price"] / tab["n_events"].clip(lower=1) - mean**2).clip(lower=0))
    tab["purchase_avg_price"] = tab["purch_amt"] / tab["n_purchase"].clip(lower=1)
    # 행동 비율(살린 신호로 파생)
    tab["remove_ratio"] = tab["n_remove_from_cart"] / tab["n_cart"].clip(lower=1)      # 장바구니 변심
    tab["cart_purchase_ratio"] = tab["n_purchase"] / tab["n_cart"].clip(lower=1)        # 구매 전환

    # ---- category 분포 → n_categories, top_cat, cat_entropy ----
    cat = pd.concat(cat_parts).groupby(["user_id", "category_id"])["c"].sum().reset_index()
    gtot = cat.groupby("user_id")["c"].transform("sum")
    cat["p"] = cat["c"] / gtot
    ncat = cat.groupby("user_id")["category_id"].nunique().rename("n_categories")
    # 엔트로피: .apply(파이썬 람다, 유저마다 루프) → 벡터화(plogp 컬럼 후 groupby.sum)
    cat["_plogp"] = -cat["p"] * np.log(cat["p"].where(cat["p"] > 0, 1.0))
    ent = cat.groupby("user_id")["_plogp"].sum().rename("cat_entropy")
    # top_cat: 전체 sort_values+tail → groupby.idxmax(부분정렬, 더 빠름)
    topcat = cat.loc[cat.groupby("user_id")["c"].idxmax()].set_index("user_id")["category_id"].rename("top_category_id")

    # ---- brand 분포 → top_brand, brand_loyalty ----
    br = pd.concat(brand_parts).groupby(["user_id", "brand"])["c"].sum().reset_index()
    bt = br.groupby("user_id")["c"].transform("sum")
    br["p"] = br["c"] / bt
    topbrand = br.loc[br.groupby("user_id")["c"].idxmax()].set_index("user_id")   # sort+tail → idxmax
    loyalty = topbrand["p"].rename("brand_loyalty"); topbrand_id = topbrand["brand"].rename("top_brand")
    nbrand = br.groupby("user_id")["brand"].nunique().rename("n_brands")

    # ---- session → n_sessions, events_per_session ----
    nsess = pd.concat(sess_parts).drop_duplicates().groupby("user_id").size().rename("n_sessions")

    # ---- last_cat/last_brand (전월 통합 마지막 이벤트) ----
    last = pd.concat(last_parts).sort_values("event_time").groupby("user_id").tail(1).set_index("user_id")
    last_cat = last["category_id"].rename("last_cat_id"); last_brand = last["brand"].astype(str).rename("last_brand")

    # ---- 합치기 ----
    feat = tab.join([ncat, ent, topcat, nbrand, loyalty, topbrand_id, nsess, last_cat, last_brand])
    feat["events_per_session"] = feat["n_events"] / feat["n_sessions"].clip(lower=1)
    feat["n_brands"] = feat["n_brands"].fillna(0); feat["n_categories"] = feat["n_categories"].fillna(0)

    # ---- 라벨(시간외삽) ----
    active = set(np.concatenate(out_active)) if out_active else set()
    purch = set(np.concatenate(out_purch)) if out_purch else set()
    idx = feat.index.to_numpy()
    feat["churn"] = (~np.isin(idx, list(active))).astype("int8")
    feat["churn_no_purchase"] = (~np.isin(idx, list(purch))).astype("int8")
    feat["cohort_recency7"] = (feat["recency_days"] <= 7).astype("int8")   # 코호트 정렬 플래그(살림: 전체+표식)

    cols = ["recency_days", "tenure_days", "ndays", "n_events", "n_view", "n_cart", "n_remove_from_cart",
            "n_purchase", "avg_price", "purch_amt", "min_price", "max_price", "std_price", "purchase_avg_price",
            "remove_ratio", "cart_purchase_ratio", "n_categories", "cat_entropy", "top_category_id", "last_cat_id",
            "n_brands", "brand_loyalty", "top_brand", "last_brand", "n_sessions", "events_per_session",
            "cohort_recency7", "churn", "churn_no_purchase"]
    out = feat.reset_index()[["user_id"] + cols]
    path = os.path.join(OUT, f"{tag}_tabular_v2.parquet")
    out.to_parquet(path, index=False)
    meta = {"block": tag, "rows": int(len(out)), "cohort_rows": int(out.cohort_recency7.sum()),
            "churn_rate": round(float(out.churn.mean()), 4), "n_features": len(cols) - 2,
            "kept_now_that_prep5m_dropped": ["category_id(n_categories/top/last/entropy)", "brand(top/loyalty/last/n)",
            "user_session(n_sessions/events_per_session)", "remove_from_cart(remove_ratio)", "price 분포(min/max/std/purchase_avg)"],
            "techniques": ["dtype 다운캐스트", "월별순차(OOM방지)", "datetime파싱+결측제거", "시간외삽분할(누수차단)",
            "분산누적std", "cross-month distinct nunique", "brand결측→UNK/category_code미사용", "고카디널리티=원값+분포요약(인코딩은 모델단계)"]}
    json.dump(meta, open(os.path.join(OUT, f"tabular_v2_meta_{tag}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    print(f"\n[v2:{tag}] 전체 {len(out):,}명(코호트 {meta['cohort_rows']:,}) | {meta['n_features']}피처 | 이탈률 {meta['churn_rate']*100:.1f}%", flush=True)
    print(f"[v2:{tag}] 저장 → {path}", flush=True)
    print("[v2] 신규 보존 피처 미리보기:\n", out[["user_id", "n_remove_from_cart", "remove_ratio", "n_categories",
          "cat_entropy", "top_category_id", "top_brand", "brand_loyalty", "n_sessions", "last_cat_id"]].head(8).to_string(index=False), flush=True)


def main():
    # train: 관찰 Oct~Jan25, 결과 Jan25~Feb01 / test: 관찰 Feb01~Feb22, 결과 Feb22~Mar01 (시간외삽)
    process_block("train", (pd.Timestamp("2019-10-01"), pd.Timestamp("2020-01-25")),
                  (pd.Timestamp("2020-01-25"), pd.Timestamp("2020-02-01")),
                  ["2019-Oct", "2019-Nov", "2019-Dec", "2020-Jan"])
    process_block("test", (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-02-22")),
                  (pd.Timestamp("2020-02-22"), pd.Timestamp("2020-03-01")), ["2020-Feb"])


if __name__ == "__main__":
    main()

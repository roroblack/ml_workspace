# -*- coding: utf-8 -*-
"""추천용 카탈로그·매핑 사전 구축 (category 누락 대비 + 유사분류 추천 토대).
산출(processed_eventbase/):
  product_catalog.parquet   product_id → category_id, brand, price_median, n_events
  category_catalog.parquet  category_id → category_code(있으면), n_products, top_brand, price_median, n_events
  category_similar.parquet  category_id → 유사 category_id top-10 (행동 동시출현, cosine)   ← '유사분류 추천' 사전
근거: category_code는 96.6% 결측(복원불가)이라 텍스트 계층 대신 **행동 co-occurrence로 유사도** 산출.
"""
import os, sys, json, zipfile, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
import scipy.sparse as sp

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(HERE, "src")
OUT = os.path.join(HERE, "preprocessing_project", "v4_model_prep", "output", "processed_eventbase"); os.makedirs(OUT, exist_ok=True)
MONTHS = ["2019-Oct", "2019-Nov", "2019-Dec", "2020-Jan", "2020-Feb"]
DT = {"product_id": "int64", "category_id": "int64", "category_code": "object", "brand": "category",
      "price": "float32", "user_session": "category", "event_type": "category"}


def main():
    prod_parts, cat_parts, code_parts, sc_parts = [], [], [], []
    t0 = time.time()
    for m in MONTHS:
        with zipfile.ZipFile(os.path.join(SRC, f"{m}.csv.zip")) as zf:
            df = pd.read_csv(zf.open(f"{m}.csv"), usecols=list(DT), dtype=DT)
        df["brand"] = df["brand"].astype("object").fillna("UNK")
        # 상품 카탈로그 부분집계
        prod_parts.append(df.groupby("product_id").agg(category_id=("category_id", "first"),
                          price_sum=("price", "sum"), n=("price", "size")).reset_index())
        # 브랜드 카운트(상품/카테고리 대표 브랜드용)
        cat_parts.append(df.groupby("category_id").agg(n_events=("price", "size"),
                          price_sum=("price", "sum")).reset_index())
        code_parts.append(df.dropna(subset=["category_code"])[["category_id", "category_code"]].drop_duplicates())
        # 카테고리 대표 브랜드: (category_id, brand) 카운트
        cb = df.groupby(["category_id", "brand"]).size().rename("c").reset_index(); cb["__m"] = m
        # 세션×카테고리 (동시출현용 distinct)
        sc_parts.append(df[["user_session", "category_id"]].drop_duplicates())
        # 누적 brand 카운트 저장
        if m == MONTHS[0]: brand_cnt = cb
        else: brand_cnt = pd.concat([brand_cnt, cb])
        print(f"  [{m}] {len(df):,}행 ({time.time()-t0:.0f}s)", flush=True); del df

    # 상품 카탈로그
    prod = pd.concat(prod_parts).groupby("product_id").agg(category_id=("category_id", "first"),
            price_sum=("price_sum", "sum"), n=("n", "sum")).reset_index()
    prod["price_median"] = (prod["price_sum"] / prod["n"]).round(2)   # 근사(평균) — 정확median은 비용 큼
    prod[["product_id", "category_id", "price_median", "n"]].rename(columns={"n": "n_events"}).to_parquet(os.path.join(OUT, "product_catalog.parquet"), index=False)

    # 카테고리 카탈로그
    cat = pd.concat(cat_parts).groupby("category_id").agg(n_events=("n_events", "sum"), price_sum=("price_sum", "sum")).reset_index()
    cat["price_median"] = (cat["price_sum"] / cat["n_events"]).round(2)
    nprod = prod.groupby("category_id").size().rename("n_products")
    code = pd.concat(code_parts).drop_duplicates("category_id").set_index("category_id")["category_code"]
    topbrand = pd.concat([brand_cnt]).groupby(["category_id", "brand"])["c"].sum().reset_index()
    topbrand = topbrand.sort_values("c").groupby("category_id").tail(1).set_index("category_id")["brand"].rename("top_brand")
    catC = cat.set_index("category_id").join([nprod, code.rename("category_code"), topbrand]).reset_index()
    catC["n_products"] = catC["n_products"].fillna(0).astype(int)
    catC.to_parquet(os.path.join(OUT, "category_catalog.parquet"), index=False)

    # ---- 유사분류: 세션 동시출현 → cosine ----
    sc = pd.concat(sc_parts).drop_duplicates().dropna(subset=["user_session", "category_id"])  # 결측 세션 제거(factorize -1 방지)
    s_idx, s_uni = pd.factorize(sc["user_session"]); c_idx, c_uni = pd.factorize(sc["category_id"])
    A = sp.csr_matrix((np.ones(len(sc), np.float32), (s_idx, c_idx)), shape=(len(s_uni), len(c_uni)))  # 세션×카테고리(이진)
    C = (A.T @ A).toarray()                              # 카테고리×카테고리 동시출현
    diag = np.sqrt(np.diag(C)); diag[diag == 0] = 1
    S = C / diag[:, None] / diag[None, :]               # cosine 유사도
    np.fill_diagonal(S, 0)
    rows = []
    for i in range(len(c_uni)):
        top = np.argsort(-S[i])[:10]
        for r, j in enumerate(top):
            if S[i, j] > 0: rows.append((int(c_uni[i]), r + 1, int(c_uni[j]), round(float(S[i, j]), 4)))
    sim = pd.DataFrame(rows, columns=["category_id", "rank", "similar_category_id", "cosine"])
    sim.to_parquet(os.path.join(OUT, "category_similar.parquet"), index=False)

    print(f"\n[catalog] 상품 {len(prod):,} | 카테고리 {len(catC)} | 유사쌍 {len(sim):,}", flush=True)
    json.dump({"products": int(len(prod)), "categories": int(len(catC)),
               "categories_with_code": int(catC.category_code.notna().sum()),
               "similar_pairs": int(len(sim)), "method": "session co-occurrence cosine"},
              open(os.path.join(OUT, "catalog_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    # 데모: 임의 카테고리의 유사분류 + 프로필
    demo = catC.sort_values("n_events", ascending=False).head(1)["category_id"].iloc[0]
    print(f"\n=== 유사분류 추천 데모: category_id={demo} ===", flush=True)
    prof = catC.set_index("category_id")
    for _, r in sim[sim.category_id == demo].iterrows():
        s = r["similar_category_id"]; p = prof.loc[s]
        print(f"  rank{int(r['rank'])} cos={r['cosine']:.3f}  cat={s}  code={p['category_code']}  top_brand={p['top_brand']}  price~{p['price_median']}", flush=True)


if __name__ == "__main__":
    main()

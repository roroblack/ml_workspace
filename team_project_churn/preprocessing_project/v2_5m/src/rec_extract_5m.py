# -*- coding: utf-8 -*-
"""17-6-3. 추천 전용 데이터셋 — 5개월 풀데이터에서 user×item(category/brand/product) 추출.
룰 §4: 추천은 REES46 필드만(user_id·event_type·category_id·brand·product_id). 가중 view1/cart3/purchase5.
월별 순차 처리(메모리 안전). 산출:
  data/processed_rec/rec_user_interest_5m.parquet   (유저별 top 카테고리/브랜드 + 점수)
  data/processed_rec/rec_user_top_categories.parquet(유저×상위5 카테고리 — 추천 후보)
  data/processed_rec/rec_item_popularity.parquet    (글로벌 상품/카테고리 인기 — 콜드스타트)
  data/processed_rec/meta_rec.json
"""
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # repo root(team_project_churn)
SRC = os.path.join(HERE, "src")
OUT = os.path.join(HERE, "sample_project", "data", "processed_rec"); os.makedirs(OUT, exist_ok=True)
MONTHS = ["2019-Oct.csv.zip", "2019-Nov.csv.zip", "2019-Dec.csv.zip", "2020-Jan.csv.zip", "2020-Feb.csv.zip"]
WEIGHT = {"view": 1.0, "cart": 3.0, "remove_from_cart": -1.0, "purchase": 5.0}
DT = {"event_type": "category", "category_id": "int32", "brand": "category", "product_id": "int32", "user_id": "int64"}


def main():
    cat_parts, brand_parts, prod_parts = [], [], []
    n_events = 0
    for m in MONTHS:
        t = time.time()
        df = pd.read_csv(os.path.join(SRC, m), usecols=list(DT), dtype=DT)
        df = df.dropna(subset=["user_id"])
        df["w"] = df["event_type"].astype(str).map(WEIGHT).fillna(0).astype("float32")
        df["brand"] = df["brand"].astype(str)
        n_events += len(df)
        cat_parts.append(df.groupby(["user_id", "category_id"], observed=True)["w"].sum().reset_index())
        brand_parts.append(df.groupby(["user_id", "brand"], observed=True)["w"].sum().reset_index())
        prod_parts.append(df.groupby("product_id")["w"].sum().reset_index())
        print(f"  [{m}] {len(df):,}행 ({time.time()-t:.0f}s)", flush=True)
        del df

    # 유저×카테고리 최종 집계 → top-5 + top-1
    cat = pd.concat(cat_parts).groupby(["user_id", "category_id"], as_index=False)["w"].sum()
    cat = cat.sort_values("w", ascending=False)
    top5 = cat.groupby("user_id").head(5).copy()
    top5["rank"] = top5.groupby("user_id").cumcount() + 1
    top5.to_parquet(os.path.join(OUT, "rec_user_top_categories.parquet"), index=False)
    top_cat = cat.groupby("user_id").head(1).rename(columns={"category_id": "top_category_id", "w": "cat_score"})

    brand = pd.concat(brand_parts).groupby(["user_id", "brand"], as_index=False)["w"].sum().sort_values("w", ascending=False)
    top_brand = brand.groupby("user_id").head(1).rename(columns={"brand": "top_brand", "w": "brand_score"})

    interest = top_cat.merge(top_brand, on="user_id", how="outer")
    interest.to_parquet(os.path.join(OUT, "rec_user_interest_5m.parquet"), index=False)

    prod = pd.concat(prod_parts).groupby("product_id", as_index=False)["w"].sum().sort_values("w", ascending=False)
    prod.head(1000).to_parquet(os.path.join(OUT, "rec_item_popularity.parquet"), index=False)

    meta = {"source": "REES46 5개월 풀", "n_events": int(n_events), "weight": WEIGHT,
            "n_users": int(interest["user_id"].nunique()),
            "n_categories": int(cat["category_id"].nunique()), "n_brands": int(brand["brand"].nunique()),
            "n_products": int(prod["product_id"].nunique()),
            "files": {
                "interest": "rec_user_interest_5m.parquet (유저별 top 카테고리/브랜드)",
                "top_categories": "rec_user_top_categories.parquet (유저×상위5 카테고리)",
                "item_popularity": "rec_item_popularity.parquet (글로벌 인기 상품 top1000)"},
            "sizes_KB": {f: round(os.path.getsize(os.path.join(OUT, f)) / 1024, 1) for f in
                         ["rec_user_interest_5m.parquet", "rec_user_top_categories.parquet", "rec_item_popularity.parquet"]}}
    json.dump(meta, open(os.path.join(OUT, "meta_rec.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n추천 데이터셋 저장(processed_rec): 유저 {meta['n_users']:,} | 카테고리 {meta['n_categories']} | "
          f"브랜드 {meta['n_brands']} | 상품 {meta['n_products']:,}", flush=True)
    for f, kb in meta["sizes_KB"].items(): print(f"  {f:38s} {kb:8.1f} KB", flush=True)


if __name__ == "__main__":
    main()

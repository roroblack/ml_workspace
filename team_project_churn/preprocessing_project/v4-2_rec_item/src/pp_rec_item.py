# -*- coding: utf-8 -*-
"""v4-2 추천 전처리 — Y=다음 아이템(next item, product_id).

v4_model_prep의 X(유저별 22피처)를 **그대로 재사용**하고, 라벨을 **결과창에서 그 유저가 가장 많이 본 product_id** 로 한다.
product_id는 카디널리티가 매우 큼(수만) → **train 빈도 상위 TOPK 아이템만 클래스로** 두고 나머지는 제외(추천 다중분류 tractable).

시간 분할(v4와 동일):
  train: 관찰 ~2020-01-25 → 결과 01-25~02-01 [2020-Jan.csv]
  test : 관찰 02-01~02-22 → 결과 02-22~03-01 [2020-Feb.csv]

산출: output/{train,test}_item.parquet (user_id + 22피처 + y_next_item), classes_item.json, meta_item.json, first30_item.txt
"""
import os, sys, json, zipfile, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(HERE, "src")
V4OUT = os.path.join(HERE, "preprocessing_project", "v4_model_prep", "output", "processed_eventbase")
OUT = os.path.join(HERE, "preprocessing_project", "v4-2_rec_item", "output"); os.makedirs(OUT, exist_ok=True)

TARGET = "product_id"
TOPK = 1000                      # train 결과창 빈도 상위 1000 아이템만 클래스로
DROP = ["churn", "churn_no_purchase"]

BLOCKS = {
    "train": dict(xparq="train_tabular_v2.parquet", month="2020-Jan",
                  start=pd.Timestamp("2020-01-25"), end=pd.Timestamp("2020-02-01")),
    "test":  dict(xparq="test_tabular_v2.parquet",  month="2020-Feb",
                  start=pd.Timestamp("2020-02-22"), end=pd.Timestamp("2020-03-01")),
}


def read_outcome(month, start, end):
    dt = {"user_id": "int64", TARGET: "int64"}
    with zipfile.ZipFile(os.path.join(SRC, f"{month}.csv.zip")) as zf:
        df = pd.read_csv(zf.open(f"{month}.csv"), usecols=["event_time", "user_id", TARGET], dtype=dt)
    df["event_time"] = pd.to_datetime(df["event_time"].str.slice(0, 19), format="%Y-%m-%d %H:%M:%S", errors="coerce")
    m = (df.event_time >= start) & (df.event_time < end)
    return df.loc[m, ["user_id", TARGET]].dropna()


def top_per_user(ev):
    cnt = ev.groupby(["user_id", TARGET]).size().rename("c").reset_index()
    top = cnt.loc[cnt.groupby("user_id")["c"].idxmax()].set_index("user_id")[TARGET]
    return top.rename("y_next_item")


def build(tag, cfg, topk_classes=None):
    t0 = time.time()
    X = pd.read_parquet(os.path.join(V4OUT, cfg["xparq"]))
    feat_cols = [c for c in X.columns if c not in DROP + ["user_id"]]
    ev = read_outcome(cfg["month"], cfg["start"], cfg["end"])
    y = top_per_user(ev)
    df = X[["user_id"] + feat_cols].merge(y, left_on="user_id", right_index=True, how="inner")
    users_with_target = len(df)
    if topk_classes is None:                         # train: 상위 TOPK 아이템 선정
        topk_classes = df["y_next_item"].value_counts().head(TOPK).index.tolist()
    df = df[df["y_next_item"].isin(topk_classes)]    # 상위 아이템으로 한정
    cov_active = users_with_target / len(X)
    cov_topk = len(df) / max(users_with_target, 1)
    path = os.path.join(OUT, f"{tag}_item.parquet")
    df.to_parquet(path, index=False)
    meta = {"block": tag, "rows": int(len(df)), "x_users": int(len(X)),
            "users_with_outcome": int(users_with_target),
            "coverage_outcome_active": round(cov_active, 4), "coverage_in_topk": round(cov_topk, 4),
            "n_classes": int(len(topk_classes)), "TOPK": TOPK, "n_features": len(feat_cols),
            "target": f"y_next_item(=outcome 최빈 product_id, train 상위 {TOPK})",
            "top10_items": df["y_next_item"].value_counts().head(10).to_dict()}
    json.dump(meta, open(os.path.join(OUT, f"meta_item_{tag}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=int)
    print(f"[v4-2:{tag}] {len(df):,}행 / X유저 {len(X):,} (outcome활성 {cov_active*100:.1f}%, "
          f"topK내 {cov_topk*100:.1f}%) | 클래스 {len(topk_classes)} | 피처 {len(feat_cols)} | {time.time()-t0:.0f}s", flush=True)
    if tag == "train":
        json.dump([int(c) for c in topk_classes], open(os.path.join(OUT, "classes_item.json"), "w"), indent=0)
        with open(os.path.join(OUT, "first30_item.txt"), "w", encoding="utf-8") as f:
            f.write(df.head(30).to_string(index=False))
    return topk_classes


def main():
    cls = build("train", BLOCKS["train"], topk_classes=None)
    build("test", BLOCKS["test"], topk_classes=set(cls))
    print(f"[v4-2] 완료 → {OUT}", flush=True)


if __name__ == "__main__":
    main()

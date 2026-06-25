# -*- coding: utf-8 -*-
"""v4-1 추천 전처리 — Y=다음 카테고리(next category).

v4_model_prep의 X(유저별 22피처, processed_eventbase/{train,test}_tabular_v2.parquet)를 **그대로 재사용**하고,
라벨만 churn(이탈 이진) → **결과창(outcome window)에서 그 유저가 가장 많이 본 category_id** 로 교체한다.
즉 "과거 행동 요약(X) → 다음 기간 주요 관심 카테고리(Y)"의 다중분류(추천) 데이터셋.

시간 분할(v4와 동일, 누수 차단):
  train: 관찰 2019-10-01~2020-01-25 (X) → 결과 2020-01-25~02-01 (Y)  [2020-Jan.csv]
  test : 관찰 2020-02-01~02-22       (X) → 결과 2020-02-22~03-01     [2020-Feb.csv]

산출: output/{train,test}_cat.parquet (user_id + 22피처 + y_next_category), classes_cat.json, meta_cat.json, first30_cat.txt
"""
import os, sys, json, zipfile, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(HERE, "src")
V4OUT = os.path.join(HERE, "preprocessing_project", "v4_model_prep", "output", "processed_eventbase")
OUT = os.path.join(HERE, "preprocessing_project", "v4-1_rec_category", "output"); os.makedirs(OUT, exist_ok=True)

TARGET = "category_id"
# X에서 제외할 비피처 컬럼(라벨/식별자)
DROP = ["churn", "churn_no_purchase"]

BLOCKS = {
    "train": dict(xparq="train_tabular_v2.parquet", month="2020-Jan",
                  start=pd.Timestamp("2020-01-25"), end=pd.Timestamp("2020-02-01")),
    "test":  dict(xparq="test_tabular_v2.parquet",  month="2020-Feb",
                  start=pd.Timestamp("2020-02-22"), end=pd.Timestamp("2020-03-01")),
}


def read_outcome(month, start, end):
    """결과창 이벤트(user_id, TARGET)만 슬라이스로 읽음."""
    dt = {"user_id": "int64", TARGET: "int64"}
    with zipfile.ZipFile(os.path.join(SRC, f"{month}.csv.zip")) as zf:
        df = pd.read_csv(zf.open(f"{month}.csv"), usecols=["event_time", "user_id", TARGET], dtype=dt)
    df["event_time"] = pd.to_datetime(df["event_time"].str.slice(0, 19), format="%Y-%m-%d %H:%M:%S", errors="coerce")
    m = (df.event_time >= start) & (df.event_time < end)
    return df.loc[m, ["user_id", TARGET]].dropna()


def top_per_user(ev):
    """유저별 결과창 최빈 TARGET = 다음 주요 관심 카테고리(동률은 첫번째)."""
    cnt = ev.groupby(["user_id", TARGET]).size().rename("c").reset_index()
    top = cnt.loc[cnt.groupby("user_id")["c"].idxmax()].set_index("user_id")[TARGET]
    return top.rename("y_next_category")


def build(tag, cfg, train_classes=None):
    t0 = time.time()
    X = pd.read_parquet(os.path.join(V4OUT, cfg["xparq"]))
    feat_cols = [c for c in X.columns if c not in DROP + ["user_id"]]
    ev = read_outcome(cfg["month"], cfg["start"], cfg["end"])
    y = top_per_user(ev)
    df = X[["user_id"] + feat_cols].merge(y, left_on="user_id", right_index=True, how="inner")
    # train에서 본 클래스만 유지(test의 미관측 클래스 제거 → 분류 일관)
    if train_classes is not None:
        df = df[df["y_next_category"].isin(train_classes)]
    classes = sorted(df["y_next_category"].unique().tolist()) if train_classes is None else train_classes
    cov = len(df) / len(X)
    path = os.path.join(OUT, f"{tag}_cat.parquet")
    df.to_parquet(path, index=False)
    meta = {"block": tag, "rows": int(len(df)), "x_users": int(len(X)),
            "coverage_outcome_active": round(cov, 4), "n_classes": int(len(classes)),
            "n_features": len(feat_cols), "target": "y_next_category(=outcome 최빈 category_id)",
            "top10_classes": df["y_next_category"].value_counts().head(10).to_dict()}
    json.dump(meta, open(os.path.join(OUT, f"meta_cat_{tag}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=int)
    print(f"[v4-1:{tag}] {len(df):,}행 / X유저 {len(X):,} (커버리지 {cov*100:.1f}%) | "
          f"클래스 {len(classes)} | 피처 {len(feat_cols)} | {time.time()-t0:.0f}s", flush=True)
    if tag == "train":
        json.dump([int(c) for c in classes], open(os.path.join(OUT, "classes_cat.json"), "w"), indent=0)
        with open(os.path.join(OUT, "first30_cat.txt"), "w", encoding="utf-8") as f:
            f.write(df.head(30).to_string(index=False))
    return classes


def main():
    classes = build("train", BLOCKS["train"], train_classes=None)
    build("test", BLOCKS["test"], train_classes=set(classes))
    print(f"[v4-1] 완료 → {OUT}", flush=True)


if __name__ == "__main__":
    main()

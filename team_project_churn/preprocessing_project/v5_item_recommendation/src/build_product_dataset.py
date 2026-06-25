# -*- coding: utf-8 -*-
"""v5 item recommendation dataset builder.

Builds a next-product prediction dataset from REES46-style events.

Default input is the small project raw sample:
  sample_project/data/raw/events.csv

Output:
  output/product_nextitem_dataset.parquet
  output/product_vocab.json
  output/dataset_meta.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[3]
PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT = PROJECT_DIR / "output"
OUT.mkdir(parents=True, exist_ok=True)

EVENT_MAP = {
    "view": "view",
    "cart": "cart",
    "remove": "remove_from_cart",
    "remove_from_cart": "remove_from_cart",
    "purchase": "purchase",
}
EVENT_TYPES = ["view", "cart", "remove_from_cart", "purchase"]


def _read_events(path: Path) -> pd.DataFrame:
    usecols = ["event_time", "event_type", "product_id", "category_id", "brand", "price", "user_id"]
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            inner = next((n for n in zf.namelist() if n.endswith(".csv")), zf.namelist()[0])
            head = pd.read_csv(zf.open(inner), nrows=0)
            session_col = "user_session" if "user_session" in head.columns else "session_id"
            df = pd.read_csv(zf.open(inner), usecols=usecols + [session_col])
    else:
        head = pd.read_csv(path, nrows=0)
        session_col = "user_session" if "user_session" in head.columns else "session_id"
        df = pd.read_csv(path, usecols=usecols + [session_col])

    df = df.rename(columns={session_col: "session_id"})
    df["event_type"] = df["event_type"].astype(str).map(EVENT_MAP).fillna(df["event_type"].astype(str))
    df = df[df["event_type"].isin(EVENT_TYPES)].copy()
    df["event_time"] = pd.to_datetime(
        df["event_time"].astype(str).str.replace(" UTC", "", regex=False),
        errors="coerce",
    )
    df = df.dropna(subset=["event_time", "session_id", "user_id", "product_id", "category_id"])
    for col in ["product_id", "category_id", "user_id"]:
        df[col] = df[col].astype("int64")
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0).clip(lower=0).astype("float32")
    df["brand"] = df["brand"].astype("object").fillna("UNK").astype(str)
    return df.sort_values(["session_id", "event_time"]).reset_index(drop=True)


def _safe_get(values, offset, default):
    return values[offset] if len(values) >= abs(offset) else default


def _make_windows(df: pd.DataFrame, window: int, max_windows: int | None, seed: int) -> pd.DataFrame:
    rows = []
    groups = list(df.groupby("session_id", sort=False))
    if max_windows:
        rng = np.random.default_rng(seed)
        rng.shuffle(groups)
    for session_id, g in groups:
        g = g.sort_values("event_time").reset_index(drop=True)
        if len(g) < 2:
            continue
        t = g["event_time"].astype("int64") // 10**9
        gaps = t.diff().fillna(0).clip(lower=0).to_numpy()

        for i in range(1, len(g)):
            hist = g.iloc[max(0, i - window):i].copy()
            hist_gaps = gaps[max(0, i - window):i]
            target = g.iloc[i]

            products = hist["product_id"].astype("int64").tolist()
            cats = hist["category_id"].astype("int64").tolist()
            events = hist["event_type"].astype(str).tolist()
            prices = hist["price"].astype(float).to_numpy()

            ev_counts = {f"n_{e}": int((hist["event_type"] == e).sum()) for e in EVENT_TYPES}
            row = {
                "user_id": int(target["user_id"]),
                "session_id": str(session_id),
                "target_time": str(target["event_time"]),
                "seq_len": int(len(hist)),
                "last_product_id": int(_safe_get(products, -1, -1)),
                "prev2_product_id": int(_safe_get(products, -2, -1)),
                "prev3_product_id": int(_safe_get(products, -3, -1)),
                "last_category_id": int(_safe_get(cats, -1, -1)),
                "prev2_category_id": int(_safe_get(cats, -2, -1)),
                "prev3_category_id": int(_safe_get(cats, -3, -1)),
                "last_event_type": str(_safe_get(events, -1, "UNK")),
                "prev2_event_type": str(_safe_get(events, -2, "UNK")),
                "prev3_event_type": str(_safe_get(events, -3, "UNK")),
                "unique_products": int(len(set(products))),
                "unique_categories": int(len(set(cats))),
                "price_last": float(prices[-1]) if len(prices) else 0.0,
                "price_mean": float(np.mean(prices)) if len(prices) else 0.0,
                "price_std": float(np.std(prices)) if len(prices) else 0.0,
                "price_min": float(np.min(prices)) if len(prices) else 0.0,
                "price_max": float(np.max(prices)) if len(prices) else 0.0,
                "gap_last_sec": float(hist_gaps[-1]) if len(hist_gaps) else 0.0,
                "gap_mean_sec": float(np.mean(hist_gaps)) if len(hist_gaps) else 0.0,
                "target_product_id": int(target["product_id"]),
                "target_category_id": int(target["category_id"]),
                "target_event_type": str(target["event_type"]),
            }
            row.update(ev_counts)
            rows.append(row)
            if max_windows and len(rows) >= max_windows:
                return pd.DataFrame(rows)

    out = pd.DataFrame(rows)
    return out


def _assign_split(df: pd.DataFrame, test_size: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    users = df["user_id"].drop_duplicates().to_numpy()
    rng.shuffle(users)
    n_test = max(1, int(len(users) * test_size))
    test_users = set(users[:n_test])
    df["is_train"] = ~df["user_id"].isin(test_users)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "sample_project" / "data" / "raw" / "events.csv"))
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--target-products", type=int, default=50)
    ap.add_argument("--max-windows", type=int, default=20000)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    print(f"[v5 build] input={input_path}", flush=True)
    df = _read_events(input_path)
    print(f"[v5 build] events={len(df):,} sessions={df.session_id.nunique():,} products={df.product_id.nunique():,}", flush=True)

    windows = _make_windows(df, args.window, args.max_windows, args.seed)
    windows = _assign_split(windows, args.test_size, args.seed)
    train = windows[windows["is_train"]]
    target_products = (
        train["target_product_id"].value_counts().head(args.target_products).index.astype("int64").tolist()
    )
    target_to_idx = {str(pid): i for i, pid in enumerate(target_products)}
    windows = windows[windows["target_product_id"].isin(target_products)].copy().reset_index(drop=True)
    windows["y_idx"] = windows["target_product_id"].map(lambda x: target_to_idx[str(int(x))]).astype("int32")
    windows["row_id"] = np.arange(len(windows), dtype=np.int64)

    dataset_path = OUT / "product_nextitem_dataset.parquet"
    vocab_path = OUT / "product_vocab.json"
    meta_path = OUT / "dataset_meta.json"
    windows.to_parquet(dataset_path, index=False)

    product_vocab = {
        "target_products": target_products,
        "target_to_idx": target_to_idx,
        "idx_to_target": {str(v): int(k) for k, v in target_to_idx.items()},
    }
    vocab_path.write_text(json.dumps(product_vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "task": "next_product_prediction",
        "input": str(input_path),
        "window_size": args.window,
        "target_products": args.target_products,
        "max_windows": args.max_windows,
        "raw_events": int(len(df)),
        "raw_sessions": int(df["session_id"].nunique()),
        "raw_products": int(df["product_id"].nunique()),
        "windows_after_target_filter": int(len(windows)),
        "train_rows": int(windows["is_train"].sum()),
        "test_rows": int((~windows["is_train"]).sum()),
        "classes": int(len(target_products)),
        "seed": args.seed,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v5 build] saved={dataset_path}", flush=True)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

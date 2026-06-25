# -*- coding: utf-8 -*-
"""Train and compare 11 next-product recommendation models.

Input:
  output/product_nextitem_dataset.parquet

Output:
  output/benchmark_metrics.json
  output/benchmark_predictions.parquet
  output/model_registry.json
  output/models/*.joblib
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT = PROJECT_DIR / "output"
MODEL_DIR = OUT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CAT_COLS = [
    "last_product_id",
    "prev2_product_id",
    "prev3_product_id",
    "last_category_id",
    "prev2_category_id",
    "prev3_category_id",
    "last_event_type",
    "prev2_event_type",
    "prev3_event_type",
]
NUM_COLS = [
    "seq_len",
    "unique_products",
    "unique_categories",
    "price_last",
    "price_mean",
    "price_std",
    "price_min",
    "price_max",
    "gap_last_sec",
    "gap_mean_sec",
    "n_view",
    "n_cart",
    "n_remove_from_cart",
    "n_purchase",
]


def _load_vocab() -> tuple[list[int], dict[int, int]]:
    vocab = json.loads((OUT / "product_vocab.json").read_text(encoding="utf-8"))
    idx_to_pid = [int(vocab["idx_to_target"][str(i)]) for i in range(len(vocab["target_products"]))]
    pid_to_idx = {pid: i for i, pid in enumerate(idx_to_pid)}
    return idx_to_pid, pid_to_idx


def _cap_train(train: pd.DataFrame, max_train: int | None, seed: int) -> pd.DataFrame:
    if not max_train or len(train) <= max_train:
        return train
    # Keep at least one row per class, then fill randomly.
    first = train.groupby("y_idx", group_keys=False).sample(n=1, random_state=seed)
    remaining = train.drop(index=first.index)
    need = max(0, max_train - len(first))
    fill = remaining.sample(min(need, len(remaining)), random_state=seed)
    return pd.concat([first, fill]).sample(frac=1, random_state=seed).reset_index(drop=True)


def _topk_from_scores(scores: np.ndarray, k: int = 10) -> np.ndarray:
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    order = np.take_along_axis(scores, part, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(part, order, axis=1)


def _metrics_from_topk(topk: np.ndarray, y: np.ndarray) -> dict:
    top1 = accuracy_score(y, topk[:, 0])
    hit = (topk == y[:, None])
    hit10 = float(hit.any(axis=1).mean())
    rr = []
    for row in hit:
        pos = np.flatnonzero(row)
        rr.append(1.0 / (int(pos[0]) + 1) if len(pos) else 0.0)
    return {"top1": round(float(top1), 6), "hit10": round(hit10, 6), "mrr10": round(float(np.mean(rr)), 6)}


def _rank_with_primary(primary: np.ndarray, popularity_order: np.ndarray, n_classes: int) -> np.ndarray:
    rows = []
    pop = [int(x) for x in popularity_order[: min(50, len(popularity_order))]]
    for p in primary:
        recs = []
        if 0 <= int(p) < n_classes:
            recs.append(int(p))
        for idx in pop:
            if idx not in recs:
                recs.append(idx)
            if len(recs) >= 10:
                break
        rows.append(recs[:10])
    return np.asarray(rows, dtype=np.int32)


def _scores_or_topk(pipe: Pipeline, X: pd.DataFrame, n_classes: int) -> np.ndarray:
    if hasattr(pipe, "predict_proba"):
        scores = pipe.predict_proba(X)
        # sklearn may omit columns if a class was unseen; align defensively.
        classes = pipe.named_steps["clf"].classes_
        if scores.shape[1] != n_classes:
            aligned = np.full((len(X), n_classes), -1e9, dtype=np.float32)
            aligned[:, classes.astype(int)] = scores
            scores = aligned
        return _topk_from_scores(scores, 10)
    if hasattr(pipe, "decision_function"):
        scores = pipe.decision_function(X)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        classes = pipe.named_steps["clf"].classes_
        if scores.shape[1] != n_classes:
            aligned = np.full((len(X), n_classes), -1e9, dtype=np.float32)
            aligned[:, classes.astype(int)] = scores
            scores = aligned
        return _topk_from_scores(scores, 10)
    pred = pipe.predict(X).astype(int)
    return pred[:, None]


def _preprocess_standard() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", dtype=np.float32), CAT_COLS),
            ("num", Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler(with_mean=False))]), NUM_COLS),
        ],
        sparse_threshold=1.0,
    )


def _preprocess_nb() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", dtype=np.float32), CAT_COLS),
            ("num", Pipeline([("impute", SimpleImputer()), ("scale", MinMaxScaler())]), NUM_COLS),
        ],
        sparse_threshold=1.0,
    )


def _model_specs(seed: int, n_classes: int) -> list[tuple[str, Pipeline | None]]:
    specs: list[tuple[str, Pipeline | None]] = [
        (
            "LogisticRegression",
            Pipeline([
                ("prep", _preprocess_standard()),
                ("clf", LogisticRegression(max_iter=120, solver="saga", n_jobs=-1, random_state=seed)),
            ]),
        ),
        (
            "SGDLogistic",
            Pipeline([
                ("prep", _preprocess_standard()),
                ("clf", SGDClassifier(loss="log_loss", max_iter=35, tol=1e-3, random_state=seed, n_jobs=-1)),
            ]),
        ),
        (
            "RidgeClassifier",
            Pipeline([
                ("prep", _preprocess_standard()),
                ("clf", RidgeClassifier(random_state=seed)),
            ]),
        ),
        (
            "ComplementNB",
            Pipeline([
                ("prep", _preprocess_nb()),
                ("clf", ComplementNB()),
            ]),
        ),
        (
            "DecisionTree",
            Pipeline([
                ("prep", _preprocess_standard()),
                ("clf", DecisionTreeClassifier(max_depth=24, min_samples_leaf=5, random_state=seed)),
            ]),
        ),
        (
            "RandomForest",
            Pipeline([
                ("prep", _preprocess_standard()),
                ("clf", RandomForestClassifier(n_estimators=80, max_depth=24, min_samples_leaf=3, n_jobs=-1, random_state=seed)),
            ]),
        ),
        (
            "ExtraTrees",
            Pipeline([
                ("prep", _preprocess_standard()),
                ("clf", ExtraTreesClassifier(n_estimators=120, max_depth=28, min_samples_leaf=2, n_jobs=-1, random_state=seed)),
            ]),
        ),
    ]
    if LGBMClassifier is not None:
        specs.append(
            (
                "LightGBM",
                Pipeline([
                    ("prep", _preprocess_standard()),
                    (
                        "clf",
                        LGBMClassifier(
                            objective="multiclass",
                            num_class=n_classes,
                            n_estimators=80,
                            learning_rate=0.08,
                            num_leaves=31,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=seed,
                            n_jobs=-1,
                            verbose=-1,
                        ),
                    ),
                ]),
            )
        )
    else:
        specs.append(("LightGBM", None))
    return specs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(OUT / "product_nextitem_dataset.parquet"))
    ap.add_argument("--max-train", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    idx_to_pid, pid_to_idx = _load_vocab()
    n_classes = len(idx_to_pid)
    df = pd.read_parquet(args.dataset)
    train = _cap_train(df[df["is_train"]].copy(), args.max_train, args.seed)
    test = df[~df["is_train"]].copy().reset_index(drop=True)
    X_train, y_train = train[CAT_COLS + NUM_COLS], train["y_idx"].to_numpy(dtype=int)
    X_test, y_test = test[CAT_COLS + NUM_COLS], test["y_idx"].to_numpy(dtype=int)

    pop_counts = train["y_idx"].value_counts().reindex(range(n_classes), fill_value=0)
    popularity_order = pop_counts.sort_values(ascending=False).index.to_numpy(dtype=int)

    results = []
    pred_rows = []
    registry = []

    def save_result(name: str, topk: np.ndarray, elapsed: float, model_path: Path | None, error: str | None = None):
        metric = _metrics_from_topk(topk, y_test) if error is None else {"top1": None, "hit10": None, "mrr10": None}
        row = {
            "model": name,
            **metric,
            "elapsed_sec": round(float(elapsed), 2),
            "status": "error" if error else "ok",
            "error": error,
            "model_path": str(model_path.relative_to(PROJECT_DIR)) if model_path else None,
        }
        results.append(row)
        registry.append({"model": name, "path": row["model_path"], "status": row["status"], "error": error})
        if error is None:
            sample_n = min(5000, len(test))
            for i in range(sample_n):
                rec_idx = [int(x) for x in topk[i].tolist()]
                rec_pid = [idx_to_pid[j] for j in rec_idx]
                pred_rows.append({
                    "model": name,
                    "row_id": int(test.iloc[i]["row_id"]),
                    "true_product_id": int(test.iloc[i]["target_product_id"]),
                    "top10_product_ids": json.dumps(rec_pid, ensure_ascii=False),
                })
        print(f"[v5 train] {name}: {row}")

    # 1. Popularity
    t = time.time()
    topk = np.tile(popularity_order[:10], (len(test), 1))
    path = MODEL_DIR / "popularity.joblib"
    joblib.dump({"type": "popularity", "popularity_order": popularity_order, "idx_to_pid": idx_to_pid}, path)
    save_result("Popularity", topk, time.time() - t, path)

    # 2. LastProduct
    t = time.time()
    primary = test["last_product_id"].map(lambda p: pid_to_idx.get(int(p), -1)).to_numpy(dtype=int)
    topk = _rank_with_primary(primary, popularity_order, n_classes)
    path = MODEL_DIR / "last_product.joblib"
    joblib.dump({"type": "last_product", "idx_to_pid": idx_to_pid}, path)
    save_result("LastProduct", topk, time.time() - t, path)

    # 3. LastCategoryPopularity
    t = time.time()
    cat_map = {}
    for cat, g in train.groupby("last_category_id"):
        order = g["y_idx"].value_counts().index.to_numpy(dtype=int)
        cat_map[int(cat)] = order
    rows = []
    for cat in test["last_category_id"].astype(int):
        recs = []
        for idx in cat_map.get(int(cat), []):
            if int(idx) not in recs:
                recs.append(int(idx))
            if len(recs) >= 10:
                break
        for idx in popularity_order:
            if int(idx) not in recs:
                recs.append(int(idx))
            if len(recs) >= 10:
                break
        rows.append(recs[:10])
    topk = np.asarray(rows, dtype=np.int32)
    path = MODEL_DIR / "last_category_popularity.joblib"
    joblib.dump({"type": "last_category_popularity", "cat_map": cat_map, "idx_to_pid": idx_to_pid}, path)
    save_result("LastCategoryPopularity", topk, time.time() - t, path)

    # 4-11. ML models
    for name, pipe in _model_specs(args.seed, n_classes):
        t = time.time()
        path = MODEL_DIR / f"{name.lower()}.joblib"
        if pipe is None:
            save_result(name, np.tile(popularity_order[:10], (len(test), 1)), time.time() - t, None, "package not installed")
            continue
        try:
            pipe.fit(X_train, y_train)
            topk = _scores_or_topk(pipe, X_test, n_classes)
            if topk.shape[1] < 10:
                topk = np.asarray([_rank_with_primary(np.array([r[0]]), popularity_order, n_classes)[0] for r in topk])
            joblib.dump(pipe, path)
            save_result(name, topk, time.time() - t, path)
        except Exception as e:
            save_result(name, np.tile(popularity_order[:10], (len(test), 1)), time.time() - t, None, f"{type(e).__name__}: {e}")

    metrics = {
        "task": "next_product_prediction",
        "n_classes": n_classes,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "max_train": args.max_train,
        "metrics": sorted(results, key=lambda r: (-1 if r["hit10"] is None else -r["hit10"], r["model"])),
    }
    (OUT / "benchmark_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(pred_rows).to_parquet(OUT / "benchmark_predictions.parquet", index=False)
    (OUT / "model_registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== v5 benchmark metrics ===")
    print(pd.DataFrame(results).sort_values(["status", "hit10"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()

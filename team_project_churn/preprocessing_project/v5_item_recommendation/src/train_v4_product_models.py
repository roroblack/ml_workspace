# -*- coding: utf-8 -*-
"""Train v4-style 7 models on the v5 next-product dataset.

v4 churn suite:
  DecisionTree, RandomForest, LogReg, XGBoost, LightGBM, CatBoost, Transformer

This script keeps the previous 11-model benchmark outputs intact and writes:
  output/v4_benchmark_metrics.json
  output/v4_benchmark_predictions.parquet
  output/v4_model_registry.json
  output/models_v4/*
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
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

from train_product_models import (
    CAT_COLS,
    NUM_COLS,
    OUT,
    PROJECT_DIR,
    _cap_train,
    _load_vocab,
    _metrics_from_topk,
    _preprocess_standard,
    _scores_or_topk,
)

sys.stdout.reconfigure(encoding="utf-8")

V4_MODEL_DIR = OUT / "models_v4"
V4_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _topk_from_scores(scores: np.ndarray, k: int = 10) -> np.ndarray:
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    order = np.take_along_axis(scores, part, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(part, order, axis=1)


def _v4_model_specs(seed: int, n_classes: int) -> list[tuple[str, Pipeline | None]]:
    specs: list[tuple[str, Pipeline | None]] = [
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
                ("clf", RandomForestClassifier(n_estimators=100, max_depth=24, min_samples_leaf=3, n_jobs=-1, random_state=seed)),
            ]),
        ),
        (
            "LogReg",
            Pipeline([
                ("prep", _preprocess_standard()),
                ("clf", LogisticRegression(max_iter=160, solver="saga", n_jobs=-1, random_state=seed)),
            ]),
        ),
    ]

    if XGBClassifier is not None:
        specs.append(
            (
                "XGBoost",
                Pipeline([
                    ("prep", _preprocess_standard()),
                    (
                        "clf",
                        XGBClassifier(
                            objective="multi:softprob",
                            num_class=n_classes,
                            n_estimators=90,
                            max_depth=5,
                            learning_rate=0.08,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            eval_metric="mlogloss",
                            tree_method="hist",
                            n_jobs=-1,
                            random_state=seed,
                        ),
                    ),
                ]),
            )
        )
    else:
        specs.append(("XGBoost", None))

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
                            n_estimators=100,
                            learning_rate=0.08,
                            num_leaves=31,
                            subsample=0.85,
                            colsample_bytree=0.85,
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

    if CatBoostClassifier is not None:
        specs.append(
            (
                "CatBoost",
                Pipeline([
                    ("prep", _preprocess_standard()),
                    (
                        "clf",
                        CatBoostClassifier(
                            loss_function="MultiClass",
                            iterations=100,
                            learning_rate=0.08,
                            depth=6,
                            random_seed=seed,
                            verbose=False,
                            allow_writing_files=False,
                        ),
                    ),
                ]),
            )
        )
    else:
        specs.append(("CatBoost", None))

    return specs


class TinyProductTransformer(nn.Module):
    def __init__(self, n_prod: int, n_cat: int, n_evt: int, n_classes: int):
        super().__init__()
        self.prod_emb = nn.Embedding(n_prod, 24, padding_idx=0)
        self.cat_emb = nn.Embedding(n_cat, 12, padding_idx=0)
        self.evt_emb = nn.Embedding(n_evt, 4, padding_idx=0)
        self.proj = nn.Linear(40, 64)
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(64)
        self.head = nn.Linear(64, n_classes)

    def forward(self, prod, cat, evt):
        x = torch.cat([self.prod_emb(prod), self.cat_emb(cat), self.evt_emb(evt)], dim=-1)
        x = self.proj(x)
        x = self.encoder(x)
        x = self.norm(x[:, -1, :])
        return self.head(x)


def _encode_sequence(train: pd.DataFrame, test: pd.DataFrame):
    prod_cols = ["prev3_product_id", "prev2_product_id", "last_product_id"]
    cat_cols = ["prev3_category_id", "prev2_category_id", "last_category_id"]
    evt_cols = ["prev3_event_type", "prev2_event_type", "last_event_type"]

    prod_vocab = {}
    for v in pd.unique(train[prod_cols].to_numpy().ravel()):
        iv = int(v)
        if iv >= 0 and iv not in prod_vocab:
            prod_vocab[iv] = len(prod_vocab) + 1
    cat_vocab = {}
    for v in pd.unique(train[cat_cols].to_numpy().ravel()):
        iv = int(v)
        if iv >= 0 and iv not in cat_vocab:
            cat_vocab[iv] = len(cat_vocab) + 1
    evt_vocab = {}
    for v in pd.unique(train[evt_cols].to_numpy().ravel()):
        sv = str(v)
        if sv != "UNK" and sv not in evt_vocab:
            evt_vocab[sv] = len(evt_vocab) + 1

    def enc(df: pd.DataFrame):
        prod = np.zeros((len(df), 3), dtype=np.int64)
        cat = np.zeros((len(df), 3), dtype=np.int64)
        evt = np.zeros((len(df), 3), dtype=np.int64)
        for j, c in enumerate(prod_cols):
            prod[:, j] = df[c].map(lambda x: prod_vocab.get(int(x), 0)).to_numpy(dtype=np.int64)
        for j, c in enumerate(cat_cols):
            cat[:, j] = df[c].map(lambda x: cat_vocab.get(int(x), 0)).to_numpy(dtype=np.int64)
        for j, c in enumerate(evt_cols):
            evt[:, j] = df[c].map(lambda x: evt_vocab.get(str(x), 0)).to_numpy(dtype=np.int64)
        return prod, cat, evt

    return enc(train), enc(test), prod_vocab, cat_vocab, evt_vocab


def _train_transformer(train: pd.DataFrame, test: pd.DataFrame, n_classes: int, seed: int, epochs: int, batch_size: int):
    if torch is None:
        raise RuntimeError("torch is not installed")

    torch.manual_seed(seed)
    np.random.seed(seed)

    (tr_prod, tr_cat, tr_evt), (te_prod, te_cat, te_evt), prod_vocab, cat_vocab, evt_vocab = _encode_sequence(train, test)
    y_train = train["y_idx"].to_numpy(dtype=np.int64)
    y_test = test["y_idx"].to_numpy(dtype=np.int64)

    model = TinyProductTransformer(len(prod_vocab) + 1, len(cat_vocab) + 1, len(evt_vocab) + 1, n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    ds = TensorDataset(
        torch.tensor(tr_prod),
        torch.tensor(tr_cat),
        torch.tensor(tr_evt),
        torch.tensor(y_train),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    last_loss = None
    for _ in range(epochs):
        losses = []
        for prod, cat, evt, y in loader:
            opt.zero_grad()
            logits = model(prod, cat, evt)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        last_loss = float(np.mean(losses)) if losses else None

    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(test), batch_size):
            prod = torch.tensor(te_prod[start:start + batch_size])
            cat = torch.tensor(te_cat[start:start + batch_size])
            evt = torch.tensor(te_evt[start:start + batch_size])
            logits = model(prod, cat, evt)
            scores.append(logits.detach().cpu().numpy())
    scores_arr = np.vstack(scores)
    topk = _topk_from_scores(scores_arr, 10)
    return topk, model, {
        "epochs": epochs,
        "batch_size": batch_size,
        "last_train_loss": last_loss,
        "prod_vocab_size": len(prod_vocab) + 1,
        "cat_vocab_size": len(cat_vocab) + 1,
        "event_vocab_size": len(evt_vocab) + 1,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(OUT / "product_nextitem_dataset.parquet"))
    ap.add_argument("--max-train", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--transformer-epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    idx_to_pid, _ = _load_vocab()
    n_classes = len(idx_to_pid)
    df = pd.read_parquet(args.dataset)
    train = _cap_train(df[df["is_train"]].copy(), args.max_train, args.seed)
    test = df[~df["is_train"]].copy().reset_index(drop=True)
    X_train, y_train = train[CAT_COLS + NUM_COLS], train["y_idx"].to_numpy(dtype=int)
    X_test, y_test = test[CAT_COLS + NUM_COLS], test["y_idx"].to_numpy(dtype=int)

    results = []
    pred_rows = []
    registry = []

    def save_result(name: str, topk: np.ndarray, elapsed: float, model_path: Path | None, error: str | None = None, extra: dict | None = None):
        metric = _metrics_from_topk(topk, y_test) if error is None else {"top1": None, "hit10": None, "mrr10": None}
        row = {
            "model": name,
            **metric,
            "elapsed_sec": round(float(elapsed), 2),
            "status": "error" if error else "ok",
            "error": error,
            "model_path": str(model_path.relative_to(PROJECT_DIR)) if model_path else None,
        }
        if extra:
            row.update(extra)
        results.append(row)
        registry.append({"model": name, "path": row["model_path"], "status": row["status"], "error": error, "extra": extra or {}})
        if error is None:
            for i in range(min(5000, len(test))):
                rec_pid = [idx_to_pid[int(j)] for j in topk[i].tolist()]
                pred_rows.append({
                    "model": name,
                    "row_id": int(test.iloc[i]["row_id"]),
                    "true_product_id": int(test.iloc[i]["target_product_id"]),
                    "top10_product_ids": json.dumps(rec_pid, ensure_ascii=False),
                })
        print(f"[v5 v4-suite] {name}: {row}", flush=True)

    for name, pipe in _v4_model_specs(args.seed, n_classes):
        t = time.time()
        path = V4_MODEL_DIR / f"{name.lower()}.joblib"
        if pipe is None:
            save_result(name, np.zeros((len(test), 10), dtype=int), time.time() - t, None, "package not installed")
            continue
        try:
            pipe.fit(X_train, y_train)
            topk = _scores_or_topk(pipe, X_test, n_classes)
            joblib.dump(pipe, path)
            save_result(name, topk, time.time() - t, path)
        except Exception as e:
            save_result(name, np.zeros((len(test), 10), dtype=int), time.time() - t, None, f"{type(e).__name__}: {e}")

    # Transformer as the v4 sequence-model counterpart.
    t = time.time()
    try:
        topk, model, meta = _train_transformer(train, test, n_classes, args.seed, args.transformer_epochs, args.batch_size)
        path = V4_MODEL_DIR / "transformer.pt"
        torch.save({"state_dict": model.state_dict(), "meta": meta}, path)
        save_result("Transformer", topk, time.time() - t, path, extra=meta)
    except Exception as e:
        save_result("Transformer", np.zeros((len(test), 10), dtype=int), time.time() - t, None, f"{type(e).__name__}: {e}")

    metrics = {
        "task": "next_product_prediction",
        "suite": "v4_7_models",
        "v4_models": ["DecisionTree", "RandomForest", "LogReg", "XGBoost", "LightGBM", "CatBoost", "Transformer"],
        "n_classes": n_classes,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "max_train": args.max_train,
        "metrics": sorted(results, key=lambda r: (-1 if r["hit10"] is None else -r["hit10"], r["model"])),
    }
    (OUT / "v4_benchmark_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(pred_rows).to_parquet(OUT / "v4_benchmark_predictions.parquet", index=False)
    (OUT / "v4_model_registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== v5 v4-suite metrics ===")
    print(pd.DataFrame(results).sort_values(["status", "hit10"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()

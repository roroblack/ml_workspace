"""연관규칙 분석 실행기.

apriori로 규칙을 뽑아 CSV로 저장하고, support–confidence 산점도(점 크기=lift)를 그린다.
지도학습 metrics 대신 규칙 개수/상위 규칙을 결과로 돌려준다.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ..association import association_rules  # noqa: E402


def run_association(config, bundle, run_dir: Path) -> dict:
    search_cfg = config.get("search", {}) or {}
    params = search_cfg.get("params", {}) or {}
    train_cfg = config.get("train", {}) or {}

    min_support = float(params.get("min_support", train_cfg.get("min_support", 0.01)))
    min_confidence = float(params.get("min_confidence", train_cfg.get("min_confidence", 0.3)))
    max_len = int(params.get("max_len", train_cfg.get("max_len", 2)))

    print(f"[assoc] min_support={min_support} min_confidence={min_confidence} max_len={max_len}")
    rules = association_rules(
        bundle.transactions,
        min_support=min_support,
        min_confidence=min_confidence,
        max_len=max_len,
    )
    print(f"[assoc] 거래 수={len(bundle.transactions)}  생성된 규칙 수={len(rules)}")

    rules_df = pd.DataFrame(rules)
    rules_csv = run_dir / "association_rules.csv"
    rules_df.to_csv(rules_csv, index=False)

    plots = []
    if not rules_df.empty:
        plot_dir = run_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        sizes = (rules_df["lift"] / rules_df["lift"].max() * 200).clip(lower=10)
        sc = ax.scatter(rules_df["support"], rules_df["confidence"],
                        s=sizes, c=rules_df["lift"], cmap="viridis", alpha=0.7)
        fig.colorbar(sc, ax=ax, label="lift")
        ax.set_xlabel("support")
        ax.set_ylabel("confidence")
        ax.set_title("Association Rules (size/color = lift)")
        path = plot_dir / "rules_scatter.png"
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        plots.append(str(path))

    top = rules[:10]
    return {
        "experiment_name": config.get("experiment_name"),
        "framework": "special",
        "task": "association_rules",
        "model": config.get("model", "apriori"),
        "dataset": config["dataset"],
        "search_method": "apriori",
        "params": {"min_support": min_support, "min_confidence": min_confidence, "max_len": max_len},
        "metrics": {"n_rules": len(rules),
                    "n_transactions": len(bundle.transactions)},
        "top_rules": top,
        "rules_csv": str(rules_csv),
        "plots": plots,
    }

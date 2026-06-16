"""플롯 저장 유틸. 모든 함수는 PNG 경로를 반환한다."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 헤드리스 환경
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402


def _save(fig, out_dir: Path, name: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return str(path)


def plot_confusion_matrix(y_true, y_pred, out_dir: Path, labels=None) -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ticks = range(cm.shape[0])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if labels and len(labels) == cm.shape[0]:
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    return _save(fig, out_dir, "confusion_matrix")


def plot_residuals(y_true, y_pred, out_dir: Path) -> str:
    resid = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_pred, resid, alpha=0.6, edgecolors="none")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (true - pred)")
    ax.set_title("Residual Plot")
    return _save(fig, out_dir, "residual_plot")


def plot_prediction_scatter(y_true, y_pred, out_dir: Path) -> str:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title("Prediction vs True")
    return _save(fig, out_dir, "prediction_scatter")


def plot_loss_curve(history: dict, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    if history.get("train_loss"):
        ax.plot(history["train_loss"], label="train")
    if history.get("valid_loss"):
        ax.plot(history["valid_loss"], label="valid")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curve")
    ax.legend()
    return _save(fig, out_dir, "loss_curve")


def plot_accuracy_curve(history: dict, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    if history.get("train_acc"):
        ax.plot(history["train_acc"], label="train")
    if history.get("valid_acc"):
        ax.plot(history["valid_acc"], label="valid")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Curve")
    ax.legend()
    return _save(fig, out_dir, "accuracy_curve")


def plot_search_history(trials: list[dict], out_dir: Path) -> str:
    """탐색 trial별 점수 추이."""
    scores = [t.get("score") for t in trials if t.get("score") is not None]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(scores) + 1), scores, marker="o", linestyle="-", alpha=0.7)
    if scores:
        best = np.maximum.accumulate(scores)
        ax.plot(range(1, len(scores) + 1), best, color="green",
                linestyle="--", label="best so far")
        ax.legend()
    ax.set_xlabel("Trial")
    ax.set_ylabel("Score")
    ax.set_title("Search History")
    return _save(fig, out_dir, "search_history")

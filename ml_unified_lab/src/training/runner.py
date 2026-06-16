"""실험 오케스트레이터.

설정 -> 데이터 로딩 -> 모델/탐색 -> 학습 -> 평가 -> 결과 저장.
framework=sklearn 은 완전 동작, framework=torch 는 torch 설치 시 동작.
결과는 runs/{timestamp}_{experiment_name}/ 에 저장된다.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from ..config import PROJECT_ROOT, load_experiment
from ..datasets import load_dataset
from ..models import build_sklearn_model
from ..reporting import (
    classification_metrics,
    plot_accuracy_curve,
    plot_confusion_matrix,
    plot_loss_curve,
    plot_prediction_scatter,
    plot_residuals,
    plot_search_history,
    regression_metrics,
)
from ..search import run_sklearn_search

RUNS_DIR = PROJECT_ROOT / "runs"


def _make_run_dir(experiment_name: str, timestamp: str | None) -> Path:
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{ts}_{experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def run_experiment(config_path, *, timestamp: str | None = None) -> dict:
    config = load_experiment(config_path)
    exp_name = config.get("experiment_name", "experiment")
    framework = config.get("framework", "sklearn")
    task = config.get("task")
    dataset_name = config["dataset"]
    model_name = config["model"]

    # torch 실험이면 run 디렉터리를 만들기 전에 의존성부터 확인
    if framework == "torch":
        _require_torch(exp_name)

    run_dir = _make_run_dir(exp_name, timestamp)
    print(f"[run] {exp_name}  (framework={framework}, task={task})")
    print(f"[run] 결과 저장 위치: {run_dir}")

    # config 스냅샷
    snapshot = {k: v for k, v in config.items() if not k.startswith("_")}
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(snapshot, f, allow_unicode=True, sort_keys=False)

    # 데이터
    print(f"[data] '{dataset_name}' 로딩 중...")
    bundle = load_dataset(dataset_name, config)

    if task == "association_rules" or framework == "special":
        from .association_runner import run_association
        result = run_association(config, bundle, run_dir)
    elif framework == "sklearn":
        result = _run_sklearn(config, bundle, run_dir, task, model_name)
    elif framework == "torch":
        result = _run_torch(config, bundle, run_dir, task, model_name)
    else:
        raise ValueError(f"알 수 없는 framework: '{framework}'")

    result["run_dir"] = str(run_dir)
    _save_json(result, run_dir / "metrics.json")
    print(f"[done] 주요 지표: {result.get('metrics')}")
    return result


# --------------------------------------------------------------------------- #
# sklearn 경로
# --------------------------------------------------------------------------- #
def _run_sklearn(config, bundle, run_dir: Path, task, model_name) -> dict:
    search_cfg = config.get("search", {}) or {}
    method = search_cfg.get("method", "none")
    n_trials = search_cfg.get("trials", config.get("_search_meta", {}).get("default_trials", 20))
    param_grid = search_cfg.get("params")
    random_state = (config.get("train", {}) or {}).get("random_state", 42)

    # 단일 실행이면서 파라미터가 지정됐으면 첫 값으로 추정기를 구성(GUI 단일값 반영)
    if method == "none" and param_grid:
        single = {k: (v[0] if isinstance(v, list) else v) for k, v in param_grid.items()}
        estimator = build_sklearn_model(model_name, task, single)
    else:
        estimator = build_sklearn_model(model_name, task)

    print(f"[search] method={method}  (모델={model_name})")
    search = run_sklearn_search(
        estimator, bundle.X_train, bundle.y_train,
        model_name=model_name, task=task, method=method,
        param_grid=param_grid, n_trials=n_trials, random_state=random_state,
    )
    best = search.best_estimator
    print(f"[search] best_params={search.best_params}  cv_score={search.best_score:.4f}")

    # 평가
    y_pred = best.predict(bundle.X_test)
    if task == "classification":
        metrics = classification_metrics(bundle.y_test, y_pred)
    else:
        metrics = regression_metrics(bundle.y_test, y_pred)

    plots = _make_plots(config, bundle, run_dir, task, y_pred, search.trials)

    # 산출물 저장
    joblib.dump(best, run_dir / "model.joblib")
    if search.trials:
        pd.DataFrame([
            {"trial": t["trial"], "score": t.get("score"), **{f"param_{k}": v for k, v in t["params"].items()}}
            for t in search.trials
        ]).to_csv(run_dir / "search_trials.csv", index=False)

    return {
        "experiment_name": config.get("experiment_name"),
        "framework": "sklearn",
        "task": task,
        "model": model_name,
        "dataset": config["dataset"],
        "search_method": search.method,
        "best_params": search.best_params,
        "cv_best_score": None if np.isnan(search.best_score) else search.best_score,
        "metrics": metrics,
        "plots": plots,
        "n_train": int(len(bundle.y_train)),
        "n_test": int(len(bundle.y_test)),
    }


# --------------------------------------------------------------------------- #
# torch 경로 (torch 설치 시 동작)
# --------------------------------------------------------------------------- #
def _require_torch(exp_name: str) -> None:
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"'{exp_name}' 실험은 PyTorch가 필요합니다.\n"
            "  설치: pip install torch torchvision\n"
            "  (sklearn 실험은 torch 없이도 바로 실행됩니다: "
            "iris_logistic_grid, breast_cancer_svm_random, concrete_gbr_grid)"
        ) from e


def _run_torch(config, bundle, run_dir: Path, task, model_name) -> dict:
    from .torch_runner import train_torch  # torch는 여기서만 import

    return train_torch(config, bundle, run_dir, task, model_name, _make_plots)


# --------------------------------------------------------------------------- #
# 플롯 공통 처리
# --------------------------------------------------------------------------- #
def _make_plots(config, bundle, run_dir: Path, task, y_pred=None, trials=None,
                history=None) -> list[str]:
    requested = (config.get("evaluation", {}) or {}).get("plots", []) or []
    plot_dir = run_dir / "plots"
    paths: list[str] = []

    for name in requested:
        try:
            if name == "confusion_matrix" and y_pred is not None and task != "regression":
                paths.append(plot_confusion_matrix(bundle.y_test, y_pred, plot_dir, bundle.target_names))
            elif name == "residual_plot" and y_pred is not None:
                paths.append(plot_residuals(bundle.y_test, y_pred, plot_dir))
            elif name == "prediction_scatter" and y_pred is not None:
                paths.append(plot_prediction_scatter(bundle.y_test, y_pred, plot_dir))
            elif name == "loss_curve" and history:
                paths.append(plot_loss_curve(history, plot_dir))
            elif name == "accuracy_curve" and history:
                paths.append(plot_accuracy_curve(history, plot_dir))
        except Exception as e:  # 플롯 실패가 실험 전체를 막지 않도록
            print(f"[warn] 플롯 '{name}' 생성 실패: {e}")

    if trials:
        try:
            paths.append(plot_search_history(trials, plot_dir))
        except Exception as e:
            print(f"[warn] search_history 플롯 실패: {e}")

    return paths

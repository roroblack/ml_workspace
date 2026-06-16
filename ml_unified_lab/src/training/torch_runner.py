"""torch 학습 루프 (torch 설치 시 동작).

tabular(DNN/MLP/LSTM) 와 vision(CNN) 을 모두 처리한다.
탐색 방법:
  none     -> train.* 값으로 단일 학습
  random   -> train.params 조합을 무작위 샘플링
  bayesian -> Optuna(TPE)로 train.params 공간 탐색 (optuna 없으면 random 폴백)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.torch_models import build_torch_model
from ..reporting import classification_metrics, regression_metrics

# train.params 중 모델 생성자로 전달되는 키
_MODEL_PARAM_KEYS = ("hidden_units", "depth", "dropout", "num_layers")


def _make_tabular_loaders(bundle, batch_size, task):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    xtr = torch.tensor(bundle.X_train, dtype=torch.float32)
    xte = torch.tensor(bundle.X_test, dtype=torch.float32)
    if task == "classification":
        ytr = torch.tensor(bundle.y_train, dtype=torch.long)
        yte = torch.tensor(bundle.y_test, dtype=torch.long)
    else:
        ytr = torch.tensor(np.asarray(bundle.y_train, dtype=np.float32)).view(-1, 1)
        yte = torch.tensor(np.asarray(bundle.y_test, dtype=np.float32)).view(-1, 1)

    train_loader = DataLoader(TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(xte, yte), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def _vision_loaders(bundle, batch_size, max_train=None, max_test=None):
    import torch
    from torch.utils.data import DataLoader, Subset

    train_ds = bundle.train_dataset
    test_ds = bundle.test_dataset
    # CPU에서 탐색을 빨리 돌리기 위한 서브샘플링 (재현 가능하도록 고정 시드)
    if max_train and max_train < len(train_ds):
        g = torch.Generator().manual_seed(42)
        idx = torch.randperm(len(train_ds), generator=g)[:max_train].tolist()
        train_ds = Subset(train_ds, idx)
    if max_test and max_test < len(test_ds):
        test_ds = Subset(test_ds, list(range(max_test)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def _optimizer(name, params, lr):
    import torch

    name = (name or "adam").lower()
    table = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }
    return table.get(name, torch.optim.Adam)(params, lr=lr)


def _train_one(model, train_loader, test_loader, *, task, epochs, lr, optimizer_name, device):
    import torch
    import torch.nn as nn

    model = model.to(device)
    criterion = nn.CrossEntropyLoss() if task != "regression" else nn.MSELoss()
    optim = _optimizer(optimizer_name, model.parameters(), lr)
    history = {"train_loss": [], "valid_loss": [], "train_acc": [], "valid_acc": []}

    for _ in range(epochs):
        model.train()
        tl, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optim.step()
            tl += loss.item() * len(xb)
            if task != "regression":
                correct += (out.argmax(1) == yb).sum().item()
                total += len(xb)
        n_train = len(train_loader.dataset)
        history["train_loss"].append(tl / max(n_train, 1))
        history["train_acc"].append(correct / total if total else 0.0)

        model.eval()
        vl, vcorrect, vtotal = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                vl += criterion(out, yb).item() * len(xb)
                if task != "regression":
                    vcorrect += (out.argmax(1) == yb).sum().item()
                    vtotal += len(xb)
        history["valid_loss"].append(vl / max(len(test_loader.dataset), 1))
        history["valid_acc"].append(vcorrect / vtotal if vtotal else 0.0)

    return model, history


def _predict_and_truth(model, loader, task, device):
    import torch

    model.eval()
    preds, truth = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device))
            if task != "regression":
                preds.append(out.argmax(1).cpu().numpy())
            else:
                preds.append(out.cpu().numpy().ravel())
            truth.append(np.asarray(yb).ravel())
    return np.concatenate(preds), np.concatenate(truth)


def _sample_random_combos(params: dict, n_trials: int, seed: int):
    rng = np.random.default_rng(seed)
    for _ in range(n_trials):
        yield {k: (v[int(rng.integers(len(v)))] if isinstance(v, list) else v)
               for k, v in params.items()}


def train_torch(config, bundle, run_dir, task, model_name, make_plots) -> dict:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_cfg = config.get("train", {}) or {}
    search_cfg = config.get("search", {}) or {}
    method = search_cfg.get("method", "none")
    n_trials = search_cfg.get("trials", 10)
    search_params = search_cfg.get("params", {}) or {}
    seed = train_cfg.get("random_state", 42)

    base = dict(
        epochs=train_cfg.get("epochs", 10),
        lr=train_cfg.get("lr", 1e-3),
        batch_size=train_cfg.get("batch_size", 32),
        optimizer=train_cfg.get("optimizer", "adam"),
    )
    max_train = train_cfg.get("max_train_samples")
    max_test = train_cfg.get("max_test_samples")

    trials: list[dict] = []
    best = {"score": -np.inf}

    def evaluate(combo: dict) -> float:
        """한 파라미터 조합으로 학습/평가하고 best를 갱신, score 반환."""
        lr = combo.get("lr", base["lr"])
        bs = int(combo.get("batch_size", base["batch_size"]))
        opt_name = combo.get("optimizer", base["optimizer"])
        epochs = int(combo.get("epochs", base["epochs"]))
        # 모델 구조 파라미터: 탐색 조합 우선, 없으면 train.* 값 사용(단일 실행도 반영)
        model_params = {}
        for k in _MODEL_PARAM_KEYS:
            if k in combo:
                model_params[k] = combo[k]
            elif k in train_cfg:
                model_params[k] = train_cfg[k]

        if bundle.kind == "vision":
            train_loader, test_loader = _vision_loaders(bundle, bs, max_train, max_test)
        else:
            train_loader, test_loader = _make_tabular_loaders(bundle, bs, task)

        model = build_torch_model(model_name, bundle, model_params)
        model, history = _train_one(
            model, train_loader, test_loader, task=task, epochs=epochs,
            lr=lr, optimizer_name=opt_name, device=device,
        )
        preds, truth = _predict_and_truth(model, test_loader, task, device)
        if task == "regression":
            m = regression_metrics(truth, preds)
            score = -m["rmse"]
        else:
            m = classification_metrics(truth, preds)
            score = m["accuracy"]

        trials.append({"trial": len(trials) + 1, "params": combo, "score": float(score)})
        print(f"[torch] trial {len(trials)}: params={combo} score={score:.4f}")
        if score > best["score"]:
            best.update(score=score, model=model, history=history, params=combo,
                        preds=preds, truth=truth, metrics=m)
        return score

    # ----- 탐색 방법별 구동 -----
    if not search_params or method in ("none", None):
        evaluate({})
        effective_method = "single_run"
    elif method == "bayesian":
        effective_method = _run_optuna(evaluate, search_params, n_trials, seed)
    elif method == "random":
        for combo in _sample_random_combos(search_params, n_trials, seed):
            evaluate(combo)
        effective_method = "random"
    else:
        raise ValueError(f"알 수 없는 탐색 방법: '{method}'")

    # ----- 산출물 저장 -----
    torch.save(best["model"].state_dict(), run_dir / "model.pt")
    if len(trials) > 1:
        pd.DataFrame([
            {"trial": t["trial"], "score": t["score"],
             **{f"param_{k}": v for k, v in t["params"].items()}}
            for t in trials
        ]).to_csv(run_dir / "search_trials.csv", index=False)

    # 플롯 (vision은 best truth를 y_test로 사용)
    bundle.y_test = best["truth"]
    plots = make_plots(config, bundle, run_dir, task, y_pred=best["preds"],
                       trials=trials if len(trials) > 1 else None, history=best["history"])

    return {
        "experiment_name": config.get("experiment_name"),
        "framework": "torch",
        "task": task,
        "model": model_name,
        "dataset": config["dataset"],
        "search_method": effective_method,
        "best_params": best["params"],
        "metrics": best["metrics"],
        "plots": plots,
        "device": device,
        "n_trials": len(trials),
    }


def _run_optuna(evaluate, search_params: dict, n_trials: int, seed: int) -> str:
    """Optuna TPE 탐색. optuna 미설치 시 random으로 폴백."""
    try:
        import optuna
    except ImportError:
        for combo in _sample_random_combos(search_params, n_trials, seed):
            evaluate(combo)
        return "bayesian(random_fallback)"

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        combo = {}
        for key, values in search_params.items():
            combo[key] = trial.suggest_categorical(key, list(values)) \
                if isinstance(values, list) else values
        return evaluate(combo)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials)
    return "bayesian"

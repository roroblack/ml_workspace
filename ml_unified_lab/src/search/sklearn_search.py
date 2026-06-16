"""sklearn 모델 하이퍼파라미터 탐색.

method:
  none     -> 단일 학습 (탐색 없음)
  grid     -> GridSearchCV
  random   -> RandomizedSearchCV
  bayesian -> optuna (없으면 random으로 폴백)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from ..models.sklearn_models import default_param_grid


@dataclass
class SearchResult:
    best_estimator: Any
    best_params: dict
    best_score: float
    method: str
    trials: list[dict] = field(default_factory=list)


def _scoring_for(task: str) -> str:
    return "accuracy" if task == "classification" else "neg_root_mean_squared_error"


def _trials_from_cv(cv) -> list[dict]:
    res = cv.cv_results_
    trials = []
    for i, params in enumerate(res["params"]):
        trials.append({
            "trial": i + 1,
            "params": params,
            "score": float(res["mean_test_score"][i]),
            "std": float(res["std_test_score"][i]),
        })
    return trials


def run_sklearn_search(
    estimator,
    X_train,
    y_train,
    *,
    model_name: str,
    task: str,
    method: str = "none",
    param_grid: dict | None = None,
    n_trials: int = 20,
    cv: int = 5,
    random_state: int = 42,
) -> SearchResult:
    scoring = _scoring_for(task)
    grid = param_grid if param_grid else default_param_grid(model_name)

    # 탐색 없음 또는 그리드가 비었으면 단일 학습
    if method == "none" or not grid:
        est = clone(estimator)
        est.fit(X_train, y_train)
        return SearchResult(est, {}, float("nan"), "single_run", [])

    if method == "grid":
        cv_search = GridSearchCV(estimator, grid, scoring=scoring, cv=cv, n_jobs=-1)
        cv_search.fit(X_train, y_train)
        return SearchResult(
            cv_search.best_estimator_, cv_search.best_params_,
            float(cv_search.best_score_), "grid", _trials_from_cv(cv_search),
        )

    if method == "random":
        cv_search = RandomizedSearchCV(
            estimator, grid, n_iter=n_trials, scoring=scoring, cv=cv,
            n_jobs=-1, random_state=random_state,
        )
        cv_search.fit(X_train, y_train)
        return SearchResult(
            cv_search.best_estimator_, cv_search.best_params_,
            float(cv_search.best_score_), "random", _trials_from_cv(cv_search),
        )

    if method == "bayesian":
        return _bayesian_search(
            estimator, X_train, y_train, grid=grid, scoring=scoring,
            cv=cv, n_trials=n_trials, random_state=random_state,
        )

    raise ValueError(f"알 수 없는 탐색 방법: '{method}'")


def _bayesian_search(estimator, X_train, y_train, *, grid, scoring, cv,
                     n_trials, random_state) -> SearchResult:
    try:
        import optuna
        from sklearn.model_selection import cross_val_score
    except ImportError:
        # optuna 없으면 random search로 폴백
        cv_search = RandomizedSearchCV(
            estimator, grid, n_iter=n_trials, scoring=scoring, cv=cv,
            n_jobs=-1, random_state=random_state,
        )
        cv_search.fit(X_train, y_train)
        result = SearchResult(
            cv_search.best_estimator_, cv_search.best_params_,
            float(cv_search.best_score_), "bayesian(random_fallback)",
            _trials_from_cv(cv_search),
        )
        return result

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trials: list[dict] = []

    def objective(trial):
        params = {}
        for key, values in grid.items():
            params[key] = trial.suggest_categorical(key, list(values))
        est = clone(estimator).set_params(**params)
        score = cross_val_score(est, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1).mean()
        trials.append({"trial": trial.number + 1, "params": params, "score": float(score)})
        return score

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials)

    best = clone(estimator).set_params(**study.best_params)
    best.fit(X_train, y_train)
    return SearchResult(best, dict(study.best_params), float(study.best_value),
                        "bayesian", trials)

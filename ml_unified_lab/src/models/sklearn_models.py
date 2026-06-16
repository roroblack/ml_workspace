"""sklearn 모델 팩토리.

각 모델은 (classification, regression) 변형과 탐색용 기본 파라미터 그리드를
함께 정의한다. build_sklearn_model 은 task에 맞는 추정기를 생성한다.
"""
from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# name -> {task: (estimator_cls, default_kwargs)}, "grid": 기본 탐색 그리드
SKLEARN_MODELS: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "classification": (LogisticRegression, {"max_iter": 1000}),
        "grid": {"C": [0.1, 1.0, 10.0], "solver": ["lbfgs"]},
    },
    "linear_regression": {
        "regression": (LinearRegression, {}),
        "grid": {},
    },
    "knn": {
        "classification": (KNeighborsClassifier, {}),
        "regression": (KNeighborsRegressor, {}),
        "grid": {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]},
    },
    "svm": {
        "classification": (SVC, {"probability": True}),
        "regression": (SVR, {}),
        "grid": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]},
    },
    "naive_bayes": {
        "classification": (GaussianNB, {}),
        "grid": {"var_smoothing": [1e-9, 1e-8, 1e-7]},
    },
    "decision_tree": {
        "classification": (DecisionTreeClassifier, {"random_state": 42}),
        "regression": (DecisionTreeRegressor, {"random_state": 42}),
        "grid": {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]},
    },
    "random_forest": {
        "classification": (RandomForestClassifier, {"random_state": 42}),
        "regression": (RandomForestRegressor, {"random_state": 42}),
        "grid": {"n_estimators": [100, 200], "max_depth": [5, 10, None]},
    },
    "gradient_boosting": {
        "classification": (GradientBoostingClassifier, {"random_state": 42}),
        "regression": (GradientBoostingRegressor, {"random_state": 42}),
        "grid": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
    },
}


def build_sklearn_model(name: str, task: str, params: dict | None = None):
    """task에 맞는 추정기를 생성. params는 생성자에 그대로 전달."""
    if name not in SKLEARN_MODELS:
        raise ValueError(
            f"알 수 없는 sklearn 모델: '{name}'. 사용 가능: {sorted(SKLEARN_MODELS)}"
        )
    spec = SKLEARN_MODELS[name]
    if task not in spec:
        supported = [k for k in spec if k != "grid"]
        raise ValueError(
            f"모델 '{name}'은(는) task '{task}'를 지원하지 않습니다. 지원: {supported}"
        )
    est_cls, default_kwargs = spec[task]
    kwargs = {**default_kwargs, **(params or {})}
    return est_cls(**kwargs)


def default_param_grid(name: str) -> dict:
    return dict(SKLEARN_MODELS.get(name, {}).get("grid", {}))

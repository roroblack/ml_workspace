"""모델 레지스트리.

build_sklearn_model(name, task, params) -> (estimator, default_param_grid)
build_torch_model(...) 은 torch_models 에서 직접 사용한다.
"""
from __future__ import annotations

from .sklearn_models import SKLEARN_MODELS, build_sklearn_model, default_param_grid

__all__ = [
    "SKLEARN_MODELS",
    "build_sklearn_model",
    "default_param_grid",
]

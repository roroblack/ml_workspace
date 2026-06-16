"""데이터셋 레지스트리.

load_dataset(name, config) -> DatasetBundle
- tabular: 전처리/스케일/분할 완료된 numpy 배열 반환
- vision: torchvision Dataset 반환 (torch 필요)
"""
from __future__ import annotations

from .tabular import (
    load_breast_cancer_ds,
    load_concrete,
    load_csv_generic,
    load_heart,
    load_housing,
    load_iris_ds,
    load_subway,
    load_titanic,
    load_wine_ds,
)
from .transactions import load_groceries, load_online_retail
from .types import DatasetBundle
from .vision import load_vision

# 이름 -> 로더 함수
_TABULAR_LOADERS = {
    "iris": load_iris_ds,
    "wine": load_wine_ds,
    "breast_cancer": load_breast_cancer_ds,
    "titanic": load_titanic,
    "subway": load_subway,
    "concrete": load_concrete,
    "heart": load_heart,
    "housing": load_housing,
}

_VISION_DATASETS = {"mnist", "fashion_mnist", "cifar10"}

_TRANSACTION_LOADERS = {"groceries": load_groceries, "online_retail": load_online_retail}


def load_dataset(name: str, config: dict | None = None) -> DatasetBundle:
    config = config or {}
    train_cfg = config.get("train", {}) or {}
    test_size = train_cfg.get("test_size", 0.2)
    random_state = train_cfg.get("random_state", 42)

    if name in _TABULAR_LOADERS:
        return _TABULAR_LOADERS[name](test_size=test_size, random_state=random_state)
    if name in _VISION_DATASETS:
        return load_vision(name, config)
    if name in _TRANSACTION_LOADERS:
        return _TRANSACTION_LOADERS[name]()

    available = sorted(set(_TABULAR_LOADERS) | _VISION_DATASETS | set(_TRANSACTION_LOADERS))
    raise ValueError(f"알 수 없는 데이터셋: '{name}'. 사용 가능: {available}")


__all__ = ["load_dataset", "DatasetBundle", "load_csv_generic"]

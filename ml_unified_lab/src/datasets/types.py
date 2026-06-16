"""데이터셋 번들 공통 타입."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DatasetBundle:
    """tabular/vision 공통 데이터 컨테이너."""

    name: str
    task: str  # classification | regression | image_classification
    kind: str = "tabular"  # tabular | vision

    # tabular 용
    X_train: np.ndarray | None = None
    X_test: np.ndarray | None = None
    y_train: np.ndarray | None = None
    y_test: np.ndarray | None = None
    feature_names: list[str] = field(default_factory=list)
    target_names: list[str] = field(default_factory=list)

    # vision 용 (torch Dataset 객체)
    train_dataset: Any = None
    test_dataset: Any = None
    input_shape: tuple | None = None

    # association_rules 용 (장바구니 트랜잭션 목록: list[list[str]])
    transactions: list | None = None

    n_classes: int | None = None

    @property
    def is_classification(self) -> bool:
        return self.task in ("classification", "image_classification")

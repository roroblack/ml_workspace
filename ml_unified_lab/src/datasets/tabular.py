"""Tabular 데이터셋 로더.

sklearn 내장 데이터셋과 archive 안의 CSV 실습 데이터를 동일한
DatasetBundle 형태로 통일해서 반환한다. 모든 로더는 학습/평가 분할과
표준화(StandardScaler)를 적용한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets as sk_datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..config import resolve_path
from .types import DatasetBundle

# archive 안의 실습 데이터 경로 (datasets.yaml의 source와 동일)
TITANIC_DIR = "archive/raw_workspace/from_colab/0528_data/titanic"
SUBWAY_DIR = "archive/raw_workspace/from_colab/0528_data/subway"
CONCRETE_CSV = "archive/raw_workspace/from_colab/0602-s/concrete_stg.csv"
HEART_CSV = "archive/raw_workspace/from_colab_deep_learning/0608-s/heart.csv"
HOUSING_CSV = "archive/raw_workspace/from_colab_deep_learning/0608-s/housing.csv"
WISC_CSV = "archive/raw_workspace/from_colab/0529-s/data/wisc_data.csv"


def _finalize(
    X: np.ndarray,
    y: np.ndarray,
    *,
    name: str,
    task: str,
    feature_names: list[str],
    target_names: list[str] | None,
    test_size: float,
    random_state: int,
    scale: bool = True,
) -> DatasetBundle:
    """공통 분할 + 표준화 후 번들 생성."""
    stratify = y if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    n_classes = int(len(np.unique(y))) if task == "classification" else None
    return DatasetBundle(
        name=name,
        task=task,
        kind="tabular",
        X_train=X_train.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_train=y_train,
        y_test=y_test,
        feature_names=list(feature_names),
        target_names=list(target_names) if target_names else [],
        n_classes=n_classes,
    )


# --------------------------------------------------------------------------- #
# sklearn 내장
# --------------------------------------------------------------------------- #
def _from_sklearn(loader, name, *, test_size, random_state) -> DatasetBundle:
    data = loader()
    return _finalize(
        data.data,
        data.target,
        name=name,
        task="classification",
        feature_names=list(data.feature_names),
        target_names=list(getattr(data, "target_names", [])),
        test_size=test_size,
        random_state=random_state,
    )


def load_iris_ds(*, test_size=0.2, random_state=42) -> DatasetBundle:
    return _from_sklearn(sk_datasets.load_iris, "iris", test_size=test_size, random_state=random_state)


def load_wine_ds(*, test_size=0.2, random_state=42) -> DatasetBundle:
    return _from_sklearn(sk_datasets.load_wine, "wine", test_size=test_size, random_state=random_state)


def load_breast_cancer_ds(*, test_size=0.2, random_state=42) -> DatasetBundle:
    # archive에 wisc_data.csv가 있으면 그것을, 없으면 sklearn 내장 사용
    csv = resolve_path(WISC_CSV)
    if csv.exists():
        df = pd.read_csv(csv)
        df = df.drop(columns=[c for c in ["id"] if c in df.columns])
        y = (df["diagnosis"].map({"M": 1, "B": 0})).to_numpy()
        X = df.drop(columns=["diagnosis"]).to_numpy(dtype=float)
        feats = [c for c in df.columns if c != "diagnosis"]
        return _finalize(
            X, y, name="breast_cancer", task="classification",
            feature_names=feats, target_names=["benign", "malignant"],
            test_size=test_size, random_state=random_state,
        )
    return _from_sklearn(
        sk_datasets.load_breast_cancer, "breast_cancer",
        test_size=test_size, random_state=random_state,
    )


# --------------------------------------------------------------------------- #
# CSV 실습 데이터
# --------------------------------------------------------------------------- #
def load_csv_generic(
    path: str | Path,
    target: str,
    *,
    name: str,
    task: str,
    drop: list[str] | None = None,
    test_size=0.2,
    random_state=42,
) -> DatasetBundle:
    """범용 CSV 로더: 범주형은 one-hot, 결측은 중앙값/최빈값으로 보간."""
    df = pd.read_csv(resolve_path(path))
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    if drop:
        df = df.drop(columns=[c for c in drop if c in df.columns])

    y_raw = df[target]
    X = df.drop(columns=[target])

    # 결측 보간
    for col in X.columns:
        if X[col].dtype.kind in "biufc":
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "NA")

    X = pd.get_dummies(X, drop_first=True)
    feature_names = list(X.columns)

    if task == "classification":
        if y_raw.dtype.kind not in "biu":
            y = pd.factorize(y_raw)[0]
            target_names = list(pd.factorize(y_raw)[1].astype(str))
        else:
            y = y_raw.to_numpy()
            target_names = [str(v) for v in sorted(pd.unique(y))]
    else:
        y = y_raw.to_numpy(dtype=float)
        target_names = []

    return _finalize(
        X.to_numpy(dtype=float), np.asarray(y),
        name=name, task=task, feature_names=feature_names,
        target_names=target_names, test_size=test_size, random_state=random_state,
    )


def load_titanic(*, test_size=0.2, random_state=42) -> DatasetBundle:
    return load_csv_generic(
        Path(TITANIC_DIR) / "train.csv", target="Survived",
        name="titanic", task="classification",
        drop=["PassengerId", "Name", "Ticket", "Cabin"],
        test_size=test_size, random_state=random_state,
    )


def load_subway(*, test_size=0.2, random_state=42) -> DatasetBundle:
    return load_csv_generic(
        Path(SUBWAY_DIR) / "subway_train.csv", target="num_people",
        name="subway", task="regression", drop=["date"],
        test_size=test_size, random_state=random_state,
    )


def load_concrete(*, test_size=0.2, random_state=42) -> DatasetBundle:
    return load_csv_generic(
        CONCRETE_CSV, target="strength", name="concrete", task="regression",
        test_size=test_size, random_state=random_state,
    )


def load_heart(*, test_size=0.2, random_state=42) -> DatasetBundle:
    return load_csv_generic(
        HEART_CSV, target="target", name="heart", task="classification",
        test_size=test_size, random_state=random_state,
    )


def load_housing(*, test_size=0.2, random_state=42) -> DatasetBundle:
    return load_csv_generic(
        HOUSING_CSV, target="MEDV", name="housing", task="regression",
        test_size=test_size, random_state=random_state,
    )

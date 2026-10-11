"""Dataset loading and deterministic train/test splitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

DATASET_NAME = "Iris flower dataset"
DATASET_SOURCE = "scikit-learn bundled dataset via sklearn.datasets.load_iris"
EXPECTED_FEATURE_NAMES = (
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
)
EXPECTED_TARGET_NAMES = ("setosa", "versicolor", "virginica")


@dataclass(frozen=True)
class IrisDataset:
    """The bundled Iris features, integer targets, and their display schema."""

    features: pd.DataFrame
    target: pd.Series
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]


@dataclass(frozen=True)
class DatasetSplit:
    """A stratified split plus original row indices for auditability."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


def load_iris_dataset() -> IrisDataset:
    """Load Iris from scikit-learn and verify the expected feature/label schema."""

    iris = load_iris()
    feature_names = tuple(str(name) for name in iris.feature_names)
    target_names = tuple(str(name) for name in iris.target_names)
    if feature_names != EXPECTED_FEATURE_NAMES:
        raise ValueError(f"Unexpected Iris feature schema: {feature_names!r}")
    if target_names != EXPECTED_TARGET_NAMES:
        raise ValueError(f"Unexpected Iris target labels: {target_names!r}")

    features = pd.DataFrame(iris.data, columns=feature_names)
    target = pd.Series(iris.target.astype(int), name="target", index=features.index)
    return IrisDataset(
        features=features,
        target=target,
        feature_names=feature_names,
        target_names=target_names,
    )


def make_stratified_split(
    dataset: IrisDataset,
    *,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> DatasetSplit:
    """Create a reproducible split while preserving class proportions."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between 0 and 1")

    row_indices = np.arange(len(dataset.target))
    train_indices, test_indices = train_test_split(
        row_indices,
        test_size=test_size,
        random_state=random_seed,
        stratify=dataset.target,
    )
    return DatasetSplit(
        X_train=dataset.features.iloc[train_indices].copy(),
        X_test=dataset.features.iloc[test_indices].copy(),
        y_train=dataset.target.iloc[train_indices].copy(),
        y_test=dataset.target.iloc[test_indices].copy(),
        train_indices=tuple(int(index) for index in train_indices),
        test_indices=tuple(int(index) for index in test_indices),
    )


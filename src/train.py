"""Reproducible model selection, final holdout evaluation, and artifact writing.

Run from the repository root with ``python -m src.train``. The fixed test
partition is excluded from cross-validation, hyperparameter selection, and
feature-importance calculations; it is evaluated after model selection.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data import (
    DATASET_NAME,
    DATASET_SOURCE,
    DatasetSplit,
    IrisDataset,
    load_iris_dataset,
    make_stratified_split,
)
from src.predict import ARTIFACT_FORMAT_VERSION

RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
PERMUTATION_REPEATS = 10
MODEL_VERSION = "1.0.0"
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
SCORING = {
    "accuracy": "accuracy",
    "precision_macro": make_scorer(
        precision_score, average="macro", zero_division=0
    ),
    "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
    "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
}


@dataclass(frozen=True)
class CandidateSpec:
    """A named pipeline and the small grid searched using training data only."""

    pipeline: Pipeline
    parameter_grid: dict[str, list[object]] | list[dict[str, object]]
    uses_scaling: bool


def build_candidate_specs(random_seed: int = RANDOM_SEED) -> dict[str, CandidateSpec]:
    """Build comparable pipelines; scaling is included only where it is useful."""

    def scaled(classifier: object) -> Pipeline:
        return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])

    def unscaled(classifier: object) -> Pipeline:
        return Pipeline([("classifier", classifier)])

    return {
        "Dummy (most frequent)": CandidateSpec(
            pipeline=unscaled(DummyClassifier(strategy="most_frequent")),
            parameter_grid=[{}],
            uses_scaling=False,
        ),
        "Logistic Regression": CandidateSpec(
            pipeline=scaled(LogisticRegression(max_iter=2_000, random_state=random_seed)),
            parameter_grid={"classifier__C": [0.1, 1.0, 10.0]},
            uses_scaling=True,
        ),
        "Support Vector Classifier": CandidateSpec(
            pipeline=scaled(SVC(probability=True, random_state=random_seed)),
            parameter_grid={
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__kernel": ["linear", "rbf"],
            },
            uses_scaling=True,
        ),
        "K-Nearest Neighbors": CandidateSpec(
            pipeline=scaled(KNeighborsClassifier()),
            parameter_grid={
                "classifier__n_neighbors": [3, 5, 7],
                "classifier__weights": ["uniform", "distance"],
            },
            uses_scaling=True,
        ),
        "Decision Tree": CandidateSpec(
            pipeline=unscaled(DecisionTreeClassifier(random_state=random_seed)),
            parameter_grid={
                "classifier__max_depth": [None, 3, 5],
                "classifier__min_samples_leaf": [1, 2],
            },
            uses_scaling=False,
        ),
        "Random Forest": CandidateSpec(
            pipeline=unscaled(
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=random_seed,
                    n_jobs=1,
                )
            ),
            parameter_grid={
                "classifier__max_depth": [None, 3, 5],
                "classifier__min_samples_leaf": [1, 2],
            },
            uses_scaling=False,
        ),
        "Extra Trees": CandidateSpec(
            pipeline=unscaled(
                ExtraTreesClassifier(
                    n_estimators=200,
                    random_state=random_seed,
                    n_jobs=1,
                )
            ),
            parameter_grid={
                "classifier__max_depth": [None, 3, 5],
                "classifier__min_samples_leaf": [1, 2],
            },
            uses_scaling=False,
        ),
    }


def _summarize_search(
    name: str,
    spec: CandidateSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[GridSearchCV, dict[str, Any]]:
    search = GridSearchCV(
        estimator=spec.pipeline,
        param_grid=spec.parameter_grid,
        scoring=SCORING,
        refit="accuracy",
        cv=cv_splits,
        n_jobs=1,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(X_train, y_train)
    results = search.cv_results_
    best_index = int(search.best_index_)
    metrics: dict[str, Any] = {}
    for metric_name in SCORING:
        split_scores = [
            float(results[f"split{fold}_test_{metric_name}"][best_index])
            for fold in range(len(cv_splits))
        ]
        metrics[metric_name] = {
            "mean": float(results[f"mean_test_{metric_name}"][best_index]),
            "std": float(results[f"std_test_{metric_name}"][best_index]),
            "fold_scores": split_scores,
        }
    record: dict[str, Any] = {
        "model": name,
        "uses_scaling": spec.uses_scaling,
        "best_params": dict(search.best_params_),
        "cv": metrics,
    }
    return search, record


def _cross_validated_permutation_importance(
    estimator: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    random_seed: int,
) -> list[dict[str, float | str]]:
    """Aggregate validation-fold permutation importance using training rows only."""

    fold_importances: list[np.ndarray] = []
    for fold_number, (fit_indices, validation_indices) in enumerate(cv_splits):
        fold_model = clone(estimator)
        fold_model.fit(X_train.iloc[fit_indices], y_train.iloc[fit_indices])
        result = permutation_importance(
            fold_model,
            X_train.iloc[validation_indices],
            y_train.iloc[validation_indices],
            scoring="f1_macro",
            n_repeats=PERMUTATION_REPEATS,
            random_state=random_seed + fold_number,
            n_jobs=1,
        )
        fold_importances.append(result.importances_mean)

    matrix = np.vstack(fold_importances)
    return [
        {
            "feature": feature,
            "mean": float(np.mean(matrix[:, index])),
            "std_across_cv_folds": float(np.std(matrix[:, index], ddof=1)),
        }
        for index, feature in enumerate(X_train.columns)
    ]


def _dataset_metadata(dataset: IrisDataset) -> dict[str, Any]:
    counts = dataset.target.value_counts().sort_index()
    return {
        "name": DATASET_NAME,
        "source": DATASET_SOURCE,
        "source_url": None,
        "samples": int(len(dataset.target)),
        "feature_names": list(dataset.feature_names),
        "feature_units": "centimetres",
        "target_name": "species",
        "target_names": list(dataset.target_names),
        "class_distribution": {
            dataset.target_names[int(class_id)]: int(count)
            for class_id, count in counts.items()
        },
    }


def _evaluate_holdout(
    model: Pipeline,
    split: DatasetSplit,
    target_names: tuple[str, ...],
) -> dict[str, Any]:
    predictions = model.predict(split.X_test)
    report = classification_report(
        split.y_test,
        predictions,
        labels=list(range(len(target_names))),
        target_names=list(target_names),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        split.y_test,
        predictions,
        labels=list(range(len(target_names))),
    )
    return {
        "partition": "stratified held-out test set; not used for model selection",
        "samples": int(len(split.y_test)),
        "class_distribution": {
            target_names[class_id]: int((split.y_test == class_id).sum())
            for class_id in range(len(target_names))
        },
        "accuracy": float(accuracy_score(split.y_test, predictions)),
        "precision_macro": float(
            precision_score(split.y_test, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(split.y_test, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(split.y_test, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(split.y_test, predictions, average="weighted", zero_division=0)
        ),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in target_names
        },
        "confusion_matrix": matrix.astype(int).tolist(),
        "confusion_matrix_labels": list(target_names),
    }


def train_and_save(
    output_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    *,
    random_seed: int = RANDOM_SEED,
    test_size: float = TEST_SIZE,
) -> tuple[Path, Path, dict[str, Any]]:
    """Select, evaluate, and persist the fitted pipeline and audit metadata."""

    dataset = load_iris_dataset()
    split = make_stratified_split(dataset, test_size=test_size, random_seed=random_seed)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_seed)
    cv_splits = list(cv.split(split.X_train, split.y_train))

    specs = build_candidate_specs(random_seed)
    candidate_records: list[dict[str, Any]] = []
    searches: dict[str, GridSearchCV] = {}
    for name, spec in specs.items():
        search, record = _summarize_search(
            name,
            spec,
            split.X_train,
            split.y_train,
            cv_splits,
        )
        searches[name] = search
        candidate_records.append(record)

    eligible = [record for record in candidate_records if record["model"] != "Dummy (most frequent)"]
    selected_record = max(
        eligible,
        key=lambda record: (
            record["cv"]["accuracy"]["mean"],
            record["cv"]["f1_macro"]["mean"],
            -list(specs).index(record["model"]),
        ),
    )
    selected_name = str(selected_record["model"])
    final_model: Pipeline = searches[selected_name].best_estimator_

    # This importance analysis uses only training-partition validation folds.
    permutation_importances = _cross_validated_permutation_importance(
        final_model,
        split.X_train,
        split.y_train,
        cv_splits,
        random_seed=random_seed,
    )
    # The held-out test set is first used here, after all selection is complete.
    holdout = _evaluate_holdout(final_model, split, dataset.target_names)

    train_class_counts = {
        dataset.target_names[class_id]: int((split.y_train == class_id).sum())
        for class_id in range(len(dataset.target_names))
    }
    train_ranges = {
        feature: {
            "min": float(split.X_train[feature].min()),
            "max": float(split.X_train[feature].max()),
        }
        for feature in dataset.feature_names
    }
    metadata: dict[str, Any] = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "model_version": MODEL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": _dataset_metadata(dataset),
        "split": {
            "method": "train_test_split with stratify=target",
            "test_size_fraction": float(test_size),
            "random_seed": int(random_seed),
            "train_samples": int(len(split.y_train)),
            "test_samples": int(len(split.y_test)),
            "train_class_distribution": train_class_counts,
            "train_row_indices": list(split.train_indices),
            "test_row_indices": list(split.test_indices),
        },
        "cross_validation": {
            "method": "StratifiedKFold(shuffle=True)",
            "folds": CV_FOLDS,
            "random_seed": int(random_seed),
            "selection_metric": "mean training-fold accuracy; macro-F1 then candidate order break ties",
            "hyperparameter_selection": "GridSearchCV on the training partition only",
            "scoring": list(SCORING),
            "candidate_results": candidate_records,
            "selected_model": selected_name,
            "selected_parameters": selected_record["best_params"],
        },
        "training": {
            "selected_model": selected_name,
            "selected_parameters": selected_record["best_params"],
            "scaling_used": bool(specs[selected_name].uses_scaling),
            "feature_ranges": train_ranges,
        },
        "evaluation": holdout,
        "feature_importance": {
            "method": "permutation importance on each training CV validation fold",
            "scoring": "macro-F1",
            "repeats_per_fold": PERMUTATION_REPEATS,
            "limitations": (
                "Global, non-causal ranking; fold variability is reported. It does not explain an individual prediction."
            ),
            "features": permutation_importances,
        },
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "iris_model.joblib"
    metadata_path = destination / "model_metadata.json"
    joblib.dump(final_model, model_path, compress=3)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return model_path, metadata_path, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help=f"directory for model and metadata (default: {DEFAULT_ARTIFACT_DIR})",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="split and model random seed")
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="fraction of dataset reserved for the final holdout (default: 0.2)",
    )
    args = parser.parse_args()

    model_path, metadata_path, metadata = train_and_save(
        args.output_dir,
        random_seed=args.seed,
        test_size=args.test_size,
    )
    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Selected model: {metadata['training']['selected_model']}")
    print("Cross-validation comparison (training partition only):")
    for result in metadata["cross_validation"]["candidate_results"]:
        score = result["cv"]["accuracy"]
        print(f"  {result['model']}: {score['mean']:.3f} +/- {score['std']:.3f}")
    evaluation = metadata["evaluation"]
    print(
        "Held-out test metrics "
        f"(n={evaluation['samples']}): accuracy={evaluation['accuracy']:.3f}, "
        f"macro-F1={evaluation['f1_macro']:.3f}, "
        f"weighted-F1={evaluation['f1_weighted']:.3f}"
    )


if __name__ == "__main__":
    main()

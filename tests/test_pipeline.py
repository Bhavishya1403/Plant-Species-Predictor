"""Tests for the dataset, training/evaluation path, and serving helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest

from src.data import (
    EXPECTED_FEATURE_NAMES,
    EXPECTED_TARGET_NAMES,
    load_iris_dataset,
    make_stratified_split,
)
from src.predict import (
    ARTIFACT_FORMAT_VERSION,
    ArtifactError,
    FeatureValidationError,
    build_feature_frame,
    load_model_bundle,
    out_of_training_range_features,
    predict_species,
)
from src.train import build_candidate_specs, train_and_save


class DatasetTests(unittest.TestCase):
    def test_dataset_schema_and_distribution(self) -> None:
        dataset = load_iris_dataset()
        self.assertEqual(dataset.features.shape, (150, 4))
        self.assertEqual(tuple(dataset.features.columns), EXPECTED_FEATURE_NAMES)
        self.assertEqual(dataset.target_names, EXPECTED_TARGET_NAMES)
        self.assertEqual(dataset.target.value_counts().sort_index().tolist(), [50, 50, 50])

    def test_split_is_reproducible_stratified_and_disjoint(self) -> None:
        dataset = load_iris_dataset()
        first = make_stratified_split(dataset, test_size=0.2, random_seed=42)
        second = make_stratified_split(dataset, test_size=0.2, random_seed=42)
        self.assertEqual(first.train_indices, second.train_indices)
        self.assertEqual(first.test_indices, second.test_indices)
        self.assertTrue(set(first.train_indices).isdisjoint(first.test_indices))
        self.assertEqual(len(first.train_indices), 120)
        self.assertEqual(len(first.test_indices), 30)
        self.assertEqual(first.y_train.value_counts().sort_index().tolist(), [40, 40, 40])
        self.assertEqual(first.y_test.value_counts().sort_index().tolist(), [10, 10, 10])

    def test_split_rejects_invalid_fraction(self) -> None:
        with self.assertRaises(ValueError):
            make_stratified_split(load_iris_dataset(), test_size=1.0)


class FeatureValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.values = {
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2,
        }

    def test_mapping_is_ordered_to_canonical_feature_schema(self) -> None:
        reversed_values = dict(reversed(list(self.values.items())))
        frame = build_feature_frame(reversed_values)
        self.assertEqual(tuple(frame.columns), EXPECTED_FEATURE_NAMES)
        self.assertEqual(frame.iloc[0].tolist(), [5.1, 3.5, 1.4, 0.2])

    def test_missing_or_extra_names_are_rejected(self) -> None:
        with self.assertRaises(FeatureValidationError):
            build_feature_frame({key: value for key, value in self.values.items() if key != "petal width (cm)"})
        with self.assertRaises(FeatureValidationError):
            build_feature_frame({**self.values, "leaf length": 2.0})

    def test_non_numeric_boolean_and_non_finite_values_are_rejected(self) -> None:
        for invalid in ("5.1", True, float("nan"), float("inf")):
            bad_values = dict(self.values)
            bad_values["sepal length (cm)"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(FeatureValidationError):
                build_feature_frame(bad_values)


class TrainingArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="iris-project-tests-")
        cls.output_dir = Path(cls.temp_dir.name)
        cls.model_path, cls.metadata_path, cls.metadata = train_and_save(cls.output_dir)
        cls.model, _ = load_model_bundle(cls.model_path, cls.metadata_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_training_persists_pipeline_and_versioned_metadata(self) -> None:
        self.assertTrue(self.model_path.is_file())
        self.assertTrue(self.metadata_path.is_file())
        self.assertEqual(self.metadata["artifact_format_version"], ARTIFACT_FORMAT_VERSION)
        self.assertEqual(tuple(self.model.feature_names_in_), EXPECTED_FEATURE_NAMES)
        self.assertEqual(tuple(int(value) for value in self.model.classes_), (0, 1, 2))

    def test_cv_compares_baseline_and_six_candidates(self) -> None:
        records = self.metadata["cross_validation"]["candidate_results"]
        self.assertEqual(len(records), 7)
        self.assertEqual(records[0]["model"], "Dummy (most frequent)")
        self.assertIn(self.metadata["training"]["selected_model"], {row["model"] for row in records[1:]})
        for row in records:
            self.assertEqual(len(row["cv"]["accuracy"]["fold_scores"]), 5)

    def test_holdout_and_report_are_consistent(self) -> None:
        evaluation = self.metadata["evaluation"]
        matrix = np.asarray(evaluation["confusion_matrix"], dtype=int)
        self.assertEqual(matrix.shape, (3, 3))
        self.assertEqual(int(matrix.sum()), evaluation["samples"])
        self.assertAlmostEqual(float(np.trace(matrix) / matrix.sum()), evaluation["accuracy"])
        self.assertAlmostEqual(
            np.mean([evaluation["per_class"][name]["f1"] for name in EXPECTED_TARGET_NAMES]),
            evaluation["f1_macro"],
        )
        for index, name in enumerate(EXPECTED_TARGET_NAMES):
            support = int(matrix[index, :].sum())
            predicted = int(matrix[:, index].sum())
            expected_recall = float(matrix[index, index] / support) if support else 0.0
            expected_precision = float(matrix[index, index] / predicted) if predicted else 0.0
            self.assertAlmostEqual(evaluation["per_class"][name]["recall"], expected_recall)
            self.assertAlmostEqual(evaluation["per_class"][name]["precision"], expected_precision)

    def test_train_and_test_indices_are_separate(self) -> None:
        split = self.metadata["split"]
        self.assertTrue(set(split["train_row_indices"]).isdisjoint(split["test_row_indices"]))
        self.assertEqual(len(split["train_row_indices"]), 120)
        self.assertEqual(len(split["test_row_indices"]), 30)

    def test_prediction_shape_label_probability_and_feature_order(self) -> None:
        result = predict_species(
            self.model,
            {
                "petal width (cm)": 0.2,
                "petal length (cm)": 1.4,
                "sepal width (cm)": 3.5,
                "sepal length (cm)": 5.1,
            },
        )
        self.assertIn(result["class_name"], EXPECTED_TARGET_NAMES)
        self.assertIn(result["class_id"], range(3))
        probabilities = result["probabilities"]
        self.assertIsNotNone(probabilities)
        self.assertEqual(set(probabilities), set(EXPECTED_TARGET_NAMES))
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=6)

    def test_out_of_training_range_is_reported(self) -> None:
        values = {
            "sepal length (cm)": 99.0,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2,
        }
        outside = out_of_training_range_features(
            values,
            self.metadata["training"]["feature_ranges"],
        )
        self.assertEqual(outside, ["sepal length (cm)"])

    def test_fixed_seed_makes_training_reproducible(self) -> None:
        dataset = load_iris_dataset()
        split = make_stratified_split(dataset, random_seed=42)
        tree_spec = build_candidate_specs(42)["Decision Tree"]
        first = tree_spec.pipeline.fit(split.X_train, split.y_train)
        second = tree_spec.pipeline.fit(split.X_train, split.y_train)
        np.testing.assert_array_equal(first.predict(split.X_test), second.predict(split.X_test))

    def test_missing_artifact_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ArtifactError, "missing"):
            load_model_bundle(self.output_dir / "absent.joblib", self.metadata_path)

    def test_incompatible_model_schema_is_rejected(self) -> None:
        wrong_features = pd.DataFrame(
            np.asarray([[1.0, 2.0, 3.0, 4.0]] * 9),
            columns=["a", "b", "c", "d"],
        )
        wrong_target = pd.Series([0, 1, 2] * 3)
        wrong_model = Pipeline([("classifier", LogisticRegression())]).fit(
            wrong_features, wrong_target
        )
        wrong_model_path = self.output_dir / "wrong_schema.joblib"
        joblib.dump(wrong_model, wrong_model_path)
        with self.assertRaisesRegex(ArtifactError, "feature schema"):
            load_model_bundle(wrong_model_path, self.metadata_path)

    def test_unsupported_artifact_metadata_is_rejected(self) -> None:
        bad_metadata = dict(self.metadata)
        bad_metadata["artifact_format_version"] = 999
        bad_metadata_path = self.output_dir / "bad_metadata.json"
        bad_metadata_path.write_text(json.dumps(bad_metadata), encoding="utf-8")
        with self.assertRaisesRegex(ArtifactError, "Unsupported artifact format"):
            load_model_bundle(self.model_path, bad_metadata_path)

    def test_incomplete_metadata_is_rejected(self) -> None:
        bad_metadata = dict(self.metadata)
        bad_metadata.pop("evaluation")
        bad_metadata_path = self.output_dir / "incomplete_metadata.json"
        bad_metadata_path.write_text(json.dumps(bad_metadata), encoding="utf-8")
        with self.assertRaisesRegex(ArtifactError, "evaluation"):
            load_model_bundle(self.model_path, bad_metadata_path)


class StreamlitAppTests(unittest.TestCase):
    def test_app_renders_and_example_preset_updates_prediction(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=60).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual([item.value for item in app.title], ["Iris Species Predictor"])
        self.assertEqual([item.label for item in app.tabs], [
            "Predict a flower",
            "Model performance",
            "Explore the dataset",
        ])
        self.assertEqual(len(app.number_input), 4)
        app.selectbox[0].select("Typical virginica").run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("virginica", app.success[0].value)


if __name__ == "__main__":
    unittest.main()

"""Streamlit interface for the evaluated Iris species classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.data import load_iris_dataset
from src.predict import (
    ArtifactError,
    FeatureValidationError,
    load_model_bundle,
    out_of_training_range_features,
    predict_species,
)

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "iris_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"


st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_dataset() -> tuple[pd.DataFrame, pd.Series, tuple[str, ...], tuple[str, ...]]:
    dataset = load_iris_dataset()
    return dataset.features, dataset.target, dataset.feature_names, dataset.target_names


@st.cache_resource(show_spinner=False)
def get_model_bundle(
    model_path: str,
    metadata_path: str,
    model_signature: str,
    metadata_signature: str,
) -> tuple[Any, dict[str, Any]]:
    del model_signature, metadata_signature  # Included in Streamlit's cache key for artifact refreshes.
    return load_model_bundle(model_path, metadata_path)


def artifact_signature(path: Path) -> str:
    """Change the resource-cache key when an artifact is replaced in place."""

    try:
        stat = path.stat()
    except OSError:
        return "missing"
    return f"{stat.st_mtime_ns}:{stat.st_size}"


features, target, feature_names, target_names = get_dataset()
species_by_id = dict(enumerate(target_names))
species_labels = target.map(species_by_id)

st.title("Iris Species Predictor")
st.caption(
    "An educational multiclass classifier for the three Iris species in the "
    "scikit-learn Iris dataset. It is not a general-purpose plant identifier."
)

try:
    model, metadata = get_model_bundle(
        str(MODEL_PATH),
        str(METADATA_PATH),
        artifact_signature(MODEL_PATH),
        artifact_signature(METADATA_PATH),
    )
except ArtifactError as exc:
    st.error(f"The trained model could not be loaded: {exc}")
    st.info("Train the model from the repository root, then refresh this page:")
    st.code("python -m src.train", language="bash")
    st.stop()

evaluation = metadata["evaluation"]
dataset_metadata = metadata["dataset"]
training_metadata = metadata["training"]
cv_metadata = metadata["cross_validation"]

summary_columns = st.columns(4)
summary_columns[0].metric("Dataset samples", dataset_metadata["samples"])
summary_columns[1].metric("Species classes", len(target_names))
summary_columns[2].metric("Held-out accuracy", f"{evaluation['accuracy']:.1%}")
summary_columns[3].metric("Held-out samples", evaluation["samples"])
st.caption(
    "The displayed score is from the recorded stratified test partition, "
    "which was not used to choose or tune the model."
)

prediction_tab, performance_tab, explorer_tab = st.tabs(
    ["Predict a flower", "Model performance", "Explore the dataset"]
)

with prediction_tab:
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.subheader("Enter four measurements")
        st.write("Measurements are in centimetres. Values outside the training range are allowed but flagged.")

        preset_values: dict[str, dict[str, float]] = {}
        for class_id, species in enumerate(target_names):
            class_medians = features.loc[target == class_id, list(feature_names)].median()
            preset_values[f"Typical {species}"] = {
                name: float(class_medians[name]) for name in feature_names
            }

        def apply_preset() -> None:
            selected = st.session_state.get("example_preset", "Custom")
            if selected in preset_values:
                for index, feature_name in enumerate(feature_names):
                    st.session_state[f"measurement_{index}"] = preset_values[selected][feature_name]

        default_values = features.median()
        for index, feature_name in enumerate(feature_names):
            st.session_state.setdefault(f"measurement_{index}", float(default_values[feature_name]))

        st.selectbox(
            "Example input",
            ["Custom", *preset_values.keys()],
            key="example_preset",
            on_change=apply_preset,
            help="Example values are medians calculated from the loaded Iris dataset.",
        )

        entered: dict[str, float] = {}
        input_columns = st.columns(2)
        for index, feature_name in enumerate(feature_names):
            with input_columns[index % 2]:
                entered[feature_name] = st.number_input(
                    feature_name,
                    step=0.1,
                    format="%.2f",
                    key=f"measurement_{index}",
                    help="Input is checked for finite values and compared with the model's training range.",
                )

        try:
            out_of_range = out_of_training_range_features(
                entered,
                training_metadata["feature_ranges"],
            )
            result = predict_species(model, entered, target_names)
        except FeatureValidationError as exc:
            st.error(f"Please correct the measurements: {exc}")
            result = None
            out_of_range = []
        except Exception as exc:  # Keep unexpected inference failures visible to the user.
            st.error(f"Prediction failed: {exc}")
            result = None
            out_of_range = []

        if out_of_range:
            st.warning(
                "Outside the training range for: "
                + ", ".join(out_of_range)
                + ". This simple per-feature range check is only a warning; values inside all ranges can still be unusual in combination."
            )

    with right:
        st.subheader("Prediction")
        if result is not None:
            st.success(f"Predicted species: **{result['class_name']}**")
            probabilities = result["probabilities"]
            if probabilities is not None:
                probability_frame = pd.DataFrame(
                    [
                        {"Species": name, "Estimated probability": value}
                        for name, value in probabilities.items()
                    ]
                ).sort_values("Estimated probability", ascending=False)
                st.bar_chart(
                    probability_frame,
                    x="Species",
                    y="Estimated probability",
                    y_label="Model probability",
                )
                st.caption(
                    "These are model-estimated probabilities, not guaranteed or calibrated real-world certainty."
                )
            else:
                st.info("This estimator does not expose class probabilities.")

            st.markdown("**Input row sent to the model**")
            st.dataframe(
                pd.DataFrame([entered], columns=feature_names),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("A prediction will appear when all four inputs are valid.")

with performance_tab:
    st.subheader("Final held-out evaluation")
    st.write(evaluation["partition"])
    metric_columns = st.columns(5)
    metric_columns[0].metric("Accuracy", f"{evaluation['accuracy']:.1%}")
    metric_columns[1].metric("Macro precision", f"{evaluation['precision_macro']:.3f}")
    metric_columns[2].metric("Macro recall", f"{evaluation['recall_macro']:.3f}")
    metric_columns[3].metric("Macro F1", f"{evaluation['f1_macro']:.3f}")
    metric_columns[4].metric("Weighted F1", f"{evaluation['f1_weighted']:.3f}")

    confusion = pd.DataFrame(
        evaluation["confusion_matrix"],
        index=[f"Actual {name}" for name in target_names],
        columns=[f"Predicted {name}" for name in target_names],
    )
    st.markdown("**Confusion matrix — rows are actual classes; columns are predictions**")
    st.dataframe(confusion, width="stretch")

    per_class = pd.DataFrame.from_dict(evaluation["per_class"], orient="index")
    per_class.index.name = "Species"
    st.markdown("**Per-class test metrics**")
    st.dataframe(per_class, width="stretch")

    st.subheader("Cross-validation model comparison")
    st.write(
        f"Five-fold stratified CV on the {metadata['split']['train_samples']} training rows. "
        "Each value is mean ± standard deviation across folds for the best grid setting of that candidate. "
        "The held-out test partition is excluded from this comparison."
    )
    comparison_rows = []
    for candidate in cv_metadata["candidate_results"]:
        accuracy_cv = candidate["cv"]["accuracy"]
        f1_cv = candidate["cv"]["f1_macro"]
        comparison_rows.append(
            {
                "Model": candidate["model"],
                "CV accuracy mean": accuracy_cv["mean"],
                "CV accuracy std": accuracy_cv["std"],
                "CV macro-F1 mean": f1_cv["mean"],
                "CV macro-F1 std": f1_cv["std"],
                "Best parameters": str(candidate["best_params"]),
            }
        )
    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["CV accuracy mean", "CV macro-F1 mean"], ascending=False
    )
    st.dataframe(comparison, hide_index=True, width="stretch")
    st.caption(f"Selected using training-fold CV: {training_metadata['selected_model']}.")

    st.subheader("Global feature importance")
    importance = pd.DataFrame(metadata["feature_importance"]["features"])
    importance = importance.sort_values("mean", ascending=False)
    importance = importance.rename(
        columns={"feature": "Feature", "mean": "Mean importance", "std_across_cv_folds": "CV fold std"}
    )
    st.bar_chart(importance, x="Feature", y="Mean importance")
    st.dataframe(importance, hide_index=True, width="stretch")
    st.caption(
        "Permutation importance is global and non-causal. It was computed on training-partition "
        "CV validation folds, and it does not explain an individual prediction."
    )

with explorer_tab:
    st.subheader("Dataset overview")
    st.write(
        f"{dataset_metadata['samples']} samples, {len(feature_names)} numeric features, "
        f"and {len(target_names)} species labels from {dataset_metadata['source']}."
    )
    distribution = pd.DataFrame(
        {
            "Species": list(dataset_metadata["class_distribution"]),
            "Samples": list(dataset_metadata["class_distribution"].values()),
        }
    )
    st.bar_chart(distribution, x="Species", y="Samples")

    selector_left, selector_right = st.columns(2)
    with selector_left:
        x_feature = st.selectbox(
            "Horizontal axis",
            feature_names,
            index=2,
            key="explorer_x",
        )
    with selector_right:
        y_feature = st.selectbox(
            "Vertical axis",
            feature_names,
            index=3,
            key="explorer_y",
        )
    scatter_data = features.copy()
    scatter_data["species"] = species_labels
    st.scatter_chart(
        scatter_data,
        x=x_feature,
        y=y_feature,
        color="species",
        x_label=x_feature,
        y_label=y_feature,
    )

    selected_distribution_feature = st.selectbox(
        "Feature distribution",
        feature_names,
        key="distribution_feature",
    )
    binned = pd.DataFrame(
        {
            "measurement bin": pd.cut(features[selected_distribution_feature], bins=8).astype(str),
            "species": species_labels,
        }
    )
    distribution_by_bin = (
        binned.groupby(["measurement bin", "species"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    st.bar_chart(distribution_by_bin)
    st.caption(f"Distribution of {selected_distribution_feature}; measurement units are centimetres.")

st.divider()
st.caption(
    f"Model version {metadata['model_version']} · scikit-learn {metadata['runtime']['scikit_learn']} · "
    "Predictions apply only to the Iris dataset's three labels and four measurements."
)

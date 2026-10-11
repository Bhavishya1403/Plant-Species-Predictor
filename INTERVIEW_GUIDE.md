# Interview guide: Iris Species Predictor

## 60-second explanation

I built a small multiclass classification demo around scikit-learn's bundled Iris dataset. It has 150 samples, four measurements in centimetres, and three species labels. The training module makes a reproducible stratified 80/20 split, compares a dummy baseline and six classifiers with five-fold stratified cross-validation on the training partition, and keeps scaling inside pipelines for models that need it. The best candidate by mean CV accuracy was a scaled linear SVC. On the untouched 30-row holdout it scored 93.3% accuracy and 0.933 macro-F1. The fitted pipeline and an evaluation record are saved, and Streamlit loads and caches them for predictions, metrics, and dataset exploration. The main limitation is that this is a small Iris benchmark, not a general plant classifier or evidence of performance on field data.

## Two-minute explanation

The application is intentionally limited to the three Iris species in scikit-learn's bundled dataset. `src/data.py` checks the expected feature order and target labels, and `src/train.py` records the dataset and class distribution. The data is split once with stratification, a 20% test fraction, and seed 42. That leaves 120 training examples, 40 per class, and a fixed test set of 30, 10 per class.

Model selection happens only on the training partition. A most-frequent dummy baseline is compared with Logistic Regression, SVC, KNN, Decision Tree, Random Forest, and Extra Trees. Each candidate uses five-fold `StratifiedKFold`; a small `GridSearchCV` searches reasonable settings. Standard scaling is part of the pipelines for Logistic Regression, SVC, and KNN, so each CV fold learns its own scaler from its fold's training rows. Trees do not need scaling. Mean CV accuracy is the selection metric, with macro-F1 and then a fixed candidate order breaking ties. The best candidate was an SVC with a linear kernel and `C=0.1` inside a `StandardScaler` pipeline.

The final selected pipeline is refit on all 120 training rows and evaluated once on the reserved 30 test rows. It achieved 28/30 correct (93.3% accuracy); the confusion matrix shows one versicolor predicted as virginica and one virginica predicted as versicolor. The model and metadata are written to `artifacts/`; metadata includes CV variation, per-class test metrics, the confusion matrix, split indices, versions, and training-fold permutation importance. The Streamlit app loads and caches those files, validates and orders inputs by feature name, and warns when a value is outside the training feature range. That warning is only a simple range heuristic, and SVC probabilities are estimates rather than proven real-world certainty. The test set is small, and the result says nothing about other plant species.

## Exact execution flow

1. `src.data.load_iris_dataset()` calls `sklearn.datasets.load_iris()`, checks the four expected feature names and three target names, and returns a feature DataFrame and integer target Series.
2. `make_stratified_split()` separates row indices using `train_test_split(..., stratify=target, test_size=0.2, random_state=42)`. It returns the train/test frames and the original row indices for metadata.
3. `src.train.build_candidate_specs()` constructs seven candidates. Scaled estimators use `Pipeline(StandardScaler, classifier)`; tree and dummy estimators use a classifier-only pipeline.
4. Each candidate is evaluated and tuned using five training-only stratified folds. GridSearchCV selects parameters using accuracy; CV fold means and standard deviations are recorded for accuracy, macro precision/recall/F1, and weighted F1.
5. The non-dummy candidate with the highest mean CV accuracy is selected. Macro-F1 and then candidate order break exact ties. GridSearchCV refits its selected pipeline on the whole training partition.
6. Training-fold permutation importance is computed using validation folds from the training partition, not the held-out test set.
7. The selected pipeline predicts the untouched test partition once. Accuracy, macro precision/recall/F1, weighted F1, per-class precision/recall/F1, support, and confusion matrix are recorded.
8. `joblib` saves the fitted pipeline to `artifacts/iris_model.joblib`; JSON metadata is written to `artifacts/model_metadata.json`.
9. `app.py` loads the actual dataset and cached artifact, then renders three tabs: prediction, measured model performance, and a dataset explorer.
10. Named numeric inputs are checked for exact feature names, canonicalized into the training order, and checked for finite values. `predict_proba()` is displayed when supported. Per-feature values outside training min/max produce a caution, not a rejection or calibrated OOD finding.

## Core concepts with this project

- **Classification:** Predicting a discrete label. Here the label is a species. **Regression** would predict a continuous number, for example petal length in centimetres.
- **Multiclass classification:** Selecting among more than two labels. This project has setosa, versicolor, and virginica.
- **Stratified split:** A split that preserves approximate class proportions. Here each partition has equal numbers from all three species.
- **Cross-validation:** Repeated training/validation partitions within the training data. It helps compare candidates and estimate selection variability; it is not the final held-out score.
- **Overfitting:** Matching training data too closely, potentially at the cost of new examples. A high training score alone would not establish performance on unseen flowers.
- **Generalization:** How a model behaves on data not used to fit or select it. This project's 30-row holdout is one limited estimate; external Iris-like samples would provide a stronger check.
- **Accuracy:** Correct predictions divided by all predictions. It is 28/30 = 0.933 on this recorded holdout.
- **Precision:** For a class, the share of predictions for that class that are correct. Versicolor precision is 0.90 here.
- **Recall:** For a class, the share of actual examples of that class that are found. Virginica recall is 0.90 here.
- **F1:** The harmonic mean of precision and recall. Macro-F1 gives every class equal weight; weighted-F1 weights by class support. They happen to match here because the holdout is balanced and class metrics are symmetric.
- **Confusion matrix:** Rows are actual labels, columns are predicted labels. Here the diagonal has 10, 9, 9; there is one versicolor/virginica confusion in each direction.
- **Class imbalance:** Unequal label counts. This Iris dataset has 50 per class, so no observed dataset imbalance; a deployment population could still have different proportions.

## Evaluation facts to state accurately

The fixed split contains 120 training and 30 test rows. Five-fold CV runs only inside the training partition. The chosen candidate had training-CV accuracy `0.975 ± 0.020`, while its final test accuracy was `0.933`; those numbers answer different questions. The test confusion matrix is:

| Actual / predicted | setosa | versicolor | virginica |
| --- | ---: | ---: | ---: |
| setosa | 10 | 0 | 0 |
| versicolor | 0 | 9 | 1 |
| virginica | 0 | 1 | 9 |

The prior 96% held-out claim was not present in the original implementation or artifacts. The new run does not reproduce it; report the actual recorded 93.3% for this split. Do not generalize that result beyond the setup described above.

## Why the implementation choices matter

- **Preprocessing inside CV:** If scaling were fitted once before cross-validation, validation-fold measurements would influence the learned scale. A `Pipeline` fits the scaler separately inside each training fold and applies it to that fold's validation data.
- **Scaling:** SVC, logistic regression, and distance-based KNN compare numeric feature magnitudes, so large-scale features can dominate without scaling. Decision trees split features independently of their units; scaling is generally unnecessary for these tree models.
- **Holdout vs training score:** The fitted model has already seen the training rows, so scoring it on them measures fit, not unseen-data behavior. The test partition is reserved until selection finishes.
- **Why a simple model can be enough:** The Iris dataset has four clean numeric features and three known classes. A small pipeline can be clear and effective here; complexity or this score does not imply broad botanical capability.
- **Probability vs confidence:** SVC probability output is an estimated class distribution from the fitted estimator. The project has not performed a calibration study, and it should not be described as a guarantee of certainty.

## Likely interview questions and code-grounded answers

1. **What problem does it solve?** It maps four Iris measurements to one of three labels. The app and README explicitly limit the scope to the bundled Iris dataset.
2. **Where does the data come from?** `load_iris()` in `src/data.py`; the repo does not download a CSV or document a separate original-source URL.
3. **How large is the dataset?** 150 rows, four numeric columns, three labels, and 50 rows per label.
4. **What are the exact feature names and order?** Sepal length, sepal width, petal length, petal width, all in cm; the order is checked in `src/data.py` and used when building input frames.
5. **What is the target representation?** Integer labels 0, 1, and 2 are mapped to setosa, versicolor, and virginica.
6. **What is the train/test methodology?** A fixed 80/20 stratified split with seed 42, giving 120 training and 30 test samples.
7. **Why stratify?** It keeps each species represented proportionally in both partitions; each test class has ten rows here.
8. **What does cross-validation do here?** Five stratified folds within the 120 training rows compare candidates and grid settings. The 30 test rows are not part of that search.
9. **Why include a DummyClassifier?** It provides a low-effort baseline. Its mean CV accuracy is 0.333, which is a useful reference for the multiclass candidates.
10. **How was the final estimator selected?** Highest mean CV accuracy; macro-F1 and then fixed candidate order break exact ties. SVC won with a linear kernel and `C=0.1`.
11. **Why use SVC?** With scaled measurements, it can form useful decision boundaries in this compact numeric feature space. CV selected it among the tested candidates; it was not assumed best in advance.
12. **Why scale only some models?** SVC, logistic regression, and KNN use distance or optimization behavior affected by feature scale. Trees split per feature and do not need scaling.
13. **Why is the scaler inside a Pipeline?** Each CV training fold learns scaling only from its own fitting rows, reducing leakage into validation folds.
14. **What is the final result?** On the fixed test split, accuracy and macro-F1 are both 0.933; 28 of 30 predictions are correct.
15. **How does that differ from the CV result?** Selected SVC mean CV accuracy is 0.975 with fold std 0.020, measured on training-fold validation partitions. Test accuracy 0.933 is measured on the separate reserved test partition.
16. **What errors did the model make?** One versicolor was predicted as virginica and one virginica as versicolor. All ten setosa examples in this split were correct.
17. **What do precision and recall mean in this example?** Versicolor precision 0.90 means 9 of 10 predictions labeled versicolor were correct; its recall 0.90 means 9 of 10 actual versicolor rows were found.
18. **Why report macro-F1 and weighted-F1?** Macro-F1 gives each species equal influence; weighted-F1 incorporates support. The dataset is balanced, but both are useful when class supports differ.
19. **Is the earlier 96% held-out claim supported?** Not by the original repo. Its app had no split or metric. The implemented, reproducible fixed split measured 93.3%; I would report that result and its methodology.
20. **How is the model persisted?** The fitted scikit-learn pipeline is saved as `iris_model.joblib`; JSON metadata includes schema, versions, split, CV results, and test report.
21. **Does Streamlit retrain on widget changes?** No. `st.cache_resource` caches the loaded model bundle, and the model is produced by the separate training command.
22. **What input safeguards exist?** The prediction helper checks exact names, converts values into canonical feature order, rejects booleans/non-numeric/non-finite values, and reports features outside training minima/maxima.
23. **What are probabilities and are they calibrated?** The selected SVC was created with `probability=True`; the UI displays its estimates. The project does not run a calibration evaluation, so they are not claimed as calibrated certainty.
24. **What does permutation importance tell you?** It measures the global change in macro-F1 after feature shuffling on training CV validation folds. It is not causal and does not explain a particular prediction.
25. **What is the biggest limitation?** A small 30-row single holdout on a curated dataset cannot establish performance on arbitrary plant species or field data.
26. **How would you improve confidence in generalization?** Collect representative independent observations, define a deployment population, and evaluate on external or temporally/geographically separated data. Repeated CV can help describe variability on Iris but cannot replace external data.

## Five debugging scenarios

1. **App says the model file is missing.** Run `python -m src.train` from the repository root; confirm both artifact paths exist; restart/refresh Streamlit. Do not make the UI silently train a model.
2. **Artifact schema is rejected.** Compare `metadata.dataset.feature_names`, the fitted pipeline's `feature_names_in_`, `EXPECTED_FEATURE_NAMES`, and app input keys. Retrain after schema/code changes.
3. **Prediction changes after an environment upgrade.** Check Python and package versions in metadata and requirements. Recreate the pinned environment, retrain, then compare the same split and results before replacing the artifact.
4. **A metric differs from the README.** Confirm seed, test fraction, candidate grids, scikit-learn version, and that `python -m src.train` completed. Read the generated `model_metadata.json`; update docs only from that run.
5. **Predictions are odd for user measurements.** Verify centimetres, feature order, and numeric validity. Check the range warning and remember that within-range coordinates can still be an unusual combination; the app only supports Iris labels.

## Five model-improvement trade-offs

1. **Tune a larger grid vs. keep the experiment interpretable:** more search can improve selection but increases variance and encourages overfitting to CV on only 120 samples. Add complexity only with a clear hypothesis.
2. **Use repeated/nested CV vs. one test split:** repeated/nested CV gives a richer estimate of selection variability, but costs more and can be confusing to present. Keep an independent final test set for a final claim.
3. **Optimize accuracy vs. class-specific errors:** accuracy fits the balanced current dataset, but macro-F1 or a class-specific cost may be better if errors have unequal importance or the class mix changes.
4. **Show probability scores vs. calibrated decisions:** scores help users see relative model outputs, but calibration requires additional data and evaluation. Calibrating on the same final test would compromise its independence.
5. **Warn on feature ranges vs. estimate joint distribution shift:** min/max checks are explainable and cheap but miss unusual combinations. A multivariate detector could catch more patterns, but adds thresholds and false alarms and still needs validation.

## Five details to personally verify before the interview

1. Re-run `python -m src.train` and be ready to explain why the seed/test split is fixed and why new seeds may produce different metrics.
2. Open `artifacts/model_metadata.json` and confirm the actual selected parameters, test confusion matrix, and package versions in your current workspace.
3. Explain which partition feeds CV, which partition feeds the final test metrics, and why permutation importance uses training-fold validation rows.
4. Try the three demo measurement tuples in `DEMO_SCRIPT.md` and make sure the displayed labels/probabilities match the current generated model artifact.
5. Open the app and explain the range warning and probability caveat in your own words; neither is a guarantee that an arbitrary flower identification is reliable.

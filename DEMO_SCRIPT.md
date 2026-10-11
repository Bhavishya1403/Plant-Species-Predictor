# 4-minute demo script

## Before presenting

From the repository root, make sure the current artifacts are generated and launch the app:

```bash
python -m src.train
streamlit run app.py
```

Use a browser window with the three tabs visible. The examples below were checked against the current generated artifact. All values are in centimetres.

## 0:00–0:30 — Set scope honestly

Say: “This demo classifies measurements into one of the three species in scikit-learn's small Iris dataset. It is not an identifier for arbitrary plants. The performance dashboard uses a fixed held-out partition, separate from the training cross-validation shown below.”

Point out the four features and the top-level held-out summary. Clarify that the displayed test result is 28 correct predictions out of 30 for the current seed and split.

## 0:30–1:35 — Make predictions

Open **Predict a flower**. Enter each tuple in the displayed feature order: sepal length, sepal width, petal length, petal width.

1. Setosa example: `5.1, 3.5, 1.4, 0.2` → expected label: **setosa**.
2. Versicolor example: `6.4, 3.2, 4.5, 1.5` → expected label: **versicolor**.
3. Virginica example: `6.3, 3.3, 6.0, 2.5` → expected label: **virginica**.

Show the probability bars, then explain: “These are the SVC's estimated class probabilities. I have not run a calibration study, so I don't present them as guaranteed real-world certainty.” If time is short, use the example preset for one species and explain that it fills class-wise medians from the loaded dataset.

## 1:35–2:35 — Explain validation and performance

Enter `99` for a feature to show the training-range warning, then restore a normal value. Say: “This is a transparent per-feature range check. It flags extrapolation but does not detect every unusual combination, and it is not a calibrated out-of-distribution detector.”

Open **Model performance**. Show accuracy, macro precision/recall/F1, per-class metrics, and the confusion matrix. Explain the two errors: one versicolor was predicted as virginica and one virginica as versicolor. All ten setosa test samples were classified correctly in this split.

State the method: “The split is stratified, 80/20, with seed 42. The 30-row test set is kept out of CV and model selection. On this run, accuracy was 0.933; with only 30 test samples, one prediction is 3.3 percentage points.”

## 2:35–3:25 — Show model selection

Show the CV comparison table. Point out the dummy baseline at 0.333 and selected SVC at `0.975 ± 0.020` five-fold training CV accuracy. Say: “That CV estimate is not the test score. It was used to compare models on the 120-row training partition; the 30-row holdout was evaluated after selection.”

Show the feature-importance chart. Say: “This is global permutation importance from training validation folds, not a causal explanation and not an explanation for one flower.”

## 3:25–4:00 — Explore and close

Open **Explore the dataset**, change the feature axes, and show class counts and binned distributions. Close with: “The workflow is reproducible and artifacts are saved, but the conclusion is limited to this small Iris experiment. I would need representative external plant data before making broader claims.”


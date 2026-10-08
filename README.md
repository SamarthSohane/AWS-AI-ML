# Fraud/Anomaly Detection with Drift Monitoring

A reproducible, time-aware benchmark for highly imbalanced card fraud detection. It trains a class-weighted supervised baseline and boosted-tree model, fits an Isolation Forest without using fraud labels in its objective, selects operating thresholds on validation data with an explicit business cost, evaluates once on a held-out later window, and monitors feature and score drift. A separate synthetic concept-drift stress test demonstrates delayed-label performance alerting and a prototype retraining loop.

> **Research/education artifact, not a production payment control.** The data are anonymized and span roughly two days; the cost inputs are illustrative. Do not use the fitted threshold or model to block real transactions without institution-specific validation, calibration, privacy/security review, and human oversight.

## Problem Statement

A fraud model trained once can quietly degrade when transaction distributions or fraud tactics change. The benchmark asks whether two supervised classifiers and a normal-only anomaly detector can rank fraud transactions, how the operating threshold trades false blocks against missed fraud, and whether monitoring can flag change before model quality becomes unacceptable.

## Objectives

- Audit the data, including schema, missingness, duplicates, class balance, amount, and time.
- Respect severe class imbalance without applying naive oversampling.
- Compare a class-weighted Logistic Regression baseline, class-weighted HistGradientBoosting, and Isolation Forest using Average Precision (PR-AUC) prominently.
- Select each decision threshold on a validation window only and evaluate on a later, untouched test window.
- Quantify false-positive/false-negative costs under a transparent, configurable assumption.
- Monitor feature PSI and BH-adjusted KS tests, model-score PSI/KS, and trigger an alert.
- Stress-test pure concept drift (changed labels conditional on features) and prototype retraining with a separate calibration slice.

## Dataset

The benchmark is the Kaggle **Credit Card Fraud Detection** dataset (ULB European cardholder transactions, September 2013): 284,807 rows, 30 numeric input features (`Time`, `V1`–`V28`, `Amount`) and the binary `Class` target. The raw file contains 492 frauds (0.1727%) and 1,081 exact duplicate rows. The loader removes exact duplicates before splitting, leaving 283,726 rows and 473 frauds (0.1667%). The audit records both raw and post-cleaning figures. No columns have missing values in this copy; the modeling pipelines still median-impute using training data only.

Download from the [Kaggle dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), accepting the current terms, and place `creditcard.csv` at `data/raw/creditcard.csv`. See [`data/README.md`](data/README.md). The dataset is not committed; the pipeline results in this repository were generated from a public mirror of the Kaggle benchmark. Follow the source dataset's license and redistribution conditions.

## Project Architecture

```text
fraud-anomaly-drift-monitoring/
├── README.md
├── requirements.txt
├── pyproject.toml
├── data/README.md
├── notebooks/fraud_detection_and_drift_monitoring.ipynb
├── src/
│   ├── config.py                 # central run and drift settings
│   ├── data_preprocessing.py     # audit, deduplication, chronological split
│   ├── supervised_model.py       # class-weighted LR and HistGradientBoosting
│   ├── anomaly_detection.py      # legitimate-reference Isolation Forest
│   ├── evaluation.py             # PR/ROC, thresholds, cost model
│   ├── drift_detection.py        # PSI, KS/FDR, score drift, alert rules
│   ├── simulation.py             # synthetic concept drift and performance alert
│   ├── visualization.py          # EDA/evaluation/monitoring figures
│   └── pipeline.py               # end-to-end experiment
├── scripts/run_pipeline.py
├── models/README.md
├── reports/
│   ├── figures/                  # generated, committed experiment figures
│   ├── results/                  # CSV/JSON/text results from actual runs
│   └── technical_report.md
└── tests/test_pipeline.py
```

## Methodology

### Data Preprocessing and Split

- The loader audits the raw file, rejects a missing/non-binary target and nonnumeric feature columns, and removes exact duplicate rows (the number and policy are recorded). Feature missingness, if present, is median-imputed inside each model pipeline.
- The data are sorted by `Time`; 60% of the earlier transactions form training, 20% the validation window, and the final 20% the held-out test window. Equal timestamps stay together at split boundaries. The measured partitions are 170,235 / 56,745 / 56,746 rows, containing 342 / 57 / 74 frauds. This preserves the temporal direction and makes the later window a drift simulation; it is not a random i.i.d. split.
- Logistic Regression and Isolation Forest scale with `StandardScaler`, fit only on their training cohort; HistGradientBoosting uses imputation but no scaling. Test rows never fit transformations, models, thresholds, or alert rules.

### Imbalance Handling

The cleaned training window has 342 frauds among 170,235 transactions. The positive class is rare enough that accuracy would reward predicting every row legitimate. ROC-AUC remains useful as a ranking diagnostic, but a very high ROC-AUC can coexist with poor precision at an operational alert budget. Average Precision (the step-wise PR-area summary returned by scikit-learn) directly exposes precision/recall performance on the rare positive class and is the primary ranking metric.

Both supervised candidates use `class_weight="balanced"`; there is no resampling of validation or test data. This avoids synthetic-neighbor assumptions for anonymized PCA features. Threshold selection is an additional, validation-only way to choose the alert/review burden.

### Supervised and Unsupervised Learning

- **Baseline:** Logistic Regression with balanced class weights, median imputation, and standardization.
- **Nonlinear candidate:** HistGradientBoostingClassifier with balanced weights, regularization, shallow leaf structure, fixed seed, and no internal random validation split. The final supervised candidate is selected by validation Average Precision (not test performance); in this run HistGradientBoosting wins the near-tie.
- **Anomaly model:** Isolation Forest is fit only on legitimate rows from the training window. The target is used only to identify a presumed-normal reference cohort; it is not passed to the Isolation Forest objective. Its output is an anomaly score, and validation labels are used only for threshold calibration after fitting. Test labels are used only for final evaluation.

### Evaluation and Threshold Selection

Precision, recall, F1, Average Precision/PR-AUC, ROC-AUC, confusion counts, false-positive rate, and false-negative rate are reported. The PR and ROC curves and confusion matrices are saved in `reports/figures/`. The test comparison uses thresholds selected from validation expected cost. The threshold sweep includes 0.10–0.90 for probability scores; the Isolation Forest uses validation anomaly-score quantiles because its scores are not probabilities.

The validation-selected HistGradientBoosting threshold is **0.881039**. Its validation precision/recall are 0.935/0.754 at 3 false positives and 14 false negatives. At 0.80 it has 9/14 FP/FN and higher expected cost; at 0.90 recall falls and cost rises. The chosen value is not a calibrated 88% fraud probability: class weighting changes score calibration, and the threshold is an empirical operating point for this validation window.

### Cost Analysis

The configurable objective is `Total Cost = FP × C_FP + FN × C_FN`.

- `C_FP = $5` USD-equivalent: illustrative 3-minute manual review at $60/hour ($3) plus $2 customer-friction allowance.
- `C_FN = $90.37` by default: 80% of the mean fraud transaction amount in the **training** window ($112.96), treating the unrecovered principal as expected loss. `--fp-cost` and `--fn-cost` can override these values. The actual missed-fraud amount is also reported separately for context.

These are assumptions, not measured issuer costs. Validation cost minimization selects the threshold; the final test evaluates that frozen threshold. Under this cost model, the selected HistGradientBoosting threshold gives test precision 0.859, recall 0.743, 9 FP, 19 FN, and estimated total cost $1,762.01. A different review capacity, recovery rate, chargeback cost, or customer friction profile can change the selected point.

### Drift Detection and Alerting

- **Features:** Compare the training reference with the later test window using decile-based Population Stability Index (PSI) and two-sample KS tests. KS p-values are adjusted across features using Benjamini–Hochberg. The monotonically increasing `Time` counter is shown in the time EDA and split ranges but excluded from actionable feature alerts: its distribution must change between non-overlapping temporal windows by construction.
- **Practical flags:** PSI <0.10 is treated as small, 0.10–0.25 as moderate, and ≥0.25 as significant heuristics (not universal laws). A moderate PSI is flagged when BH-adjusted KS q≤0.05; significant PSI is flagged directly. Alert if at least 20% of monitored features are flagged, mean feature PSI ≥0.10, or the maximum score PSI ≥0.10. Thresholds are centralized in `src/config.py`.
- **Outputs:** For the model-score comparison, the out-of-sample validation window is the reference and the final test window is current. This avoids using optimistic in-sample training predictions as the score baseline. Monitor both predicted fraud probability and anomaly score; class proportions at each validation-selected threshold are included.
- The run reports **ALERT**: 29 features monitored, 7 with PSI ≥0.25, 9 in the 0.10–0.25 band, 55.2% flagged, mean PSI 0.233, and maximum PSI 1.463 (V1). The max model-score PSI is 0.026, so this alert was driven by feature-level shift, not score PSI. KS p-values can be tiny at this sample size, which is why PSI/effect size and FDR-adjusted q-values are used together.

### Concept Drift Simulation and Retraining Prototype

The bonus stress test leaves feature values unchanged, selects a low-scoring V4 tail from training quantiles and validation scores (no validation/test labels), then synthetically labels 10% of otherwise legitimate records in that region as fraud. This is a controlled change in `P(Y|X)`, not real fraud or observed concept drift. It adds 362 synthetic frauds to the 74 test frauds, raising the stress-test prevalence to 0.768%.

Because covariates and model scores are unchanged, a feature-PSI monitor is not expected to detect this *pure* concept shift. Once feedback labels arrive, the performance monitor does: test AP falls from 0.799 to 0.142 and recall at the frozen threshold from 0.743 to 0.126, crossing the configured alert thresholds. The prototype retrains on historical data plus the first 80% of the shifted validation window, calibrates its operating threshold on the remaining 20%, and evaluates on the separate shifted test window. Test AP improves to 0.255 and recall to 0.798; precision falls to 0.119 and estimated cost falls from $34,475 to $20,852 while false positives rise from 9 to 2,580. This illustrates adaptation/review trade-offs, not production-ready automatic retraining.

## Results

Generated from the fixed-seed end-to-end run (seed 42). Metrics below are on the **held-out chronological test window**; threshold selection was performed on validation.

| Model | Type | Precision | Recall | F1 | PR-AUC / AP | ROC-AUC | FP | FN | Threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression (balanced) | Supervised baseline | 0.724 | 0.743 | 0.733 | 0.747 | 0.982 | 21 | 19 | 0.999630 |
| HistGradientBoosting (balanced) | Supervised final | **0.859** | 0.743 | **0.797** | **0.799** | 0.982 | **9** | 19 | **0.881039** |
| Isolation Forest (normal-only) | Unsupervised | 0.000 | 0.000 | 0.000 | 0.036 | 0.950 | 0 | 74 | 0.273661 |

The Isolation Forest has a relatively high ROC-AUC (0.950), but poor AP (0.036) and the validation-cost operating point flags no test transactions. That contrast is precisely why ROC-AUC is not sufficient under 0.13% test prevalence. Unsupervised detection can still help surface novel patterns for analyst triage when labels are delayed or unavailable; this Isolation Forest configuration is not competitive as a standalone blocker on this benchmark.

### Selected supervised threshold trade-off (validation only)

| Threshold | Precision | Recall | FP | FN | Estimated cost |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.266 | 0.807 | 127 | 11 | $1,629.06 |
| 0.50 | 0.632 | 0.754 | 25 | 14 | $1,390.16 |
| 0.80 | 0.827 | 0.754 | 9 | 14 | $1,310.16 |
| **0.881039 (selected)** | **0.935** | **0.754** | **3** | **14** | **$1,280.16** |
| 0.90 | 0.932 | 0.719 | 3 | 16 | $1,460.90 |

`reports/results/` contains the full audit, split summary, model comparison, threshold sweep, assumptions, feature/model drift, alert report, and synthetic retraining comparison. All reported metrics are generated by the pipeline; none are hard-coded into the outputs.

## Key Findings

1. Class weighting plus validation threshold tuning handles the benchmark better than naive accuracy optimization; no oversampling was applied.
2. HistGradientBoosting slightly wins the validation AP selection and is the stronger held-out model by AP and precision, though test recall ties the logistic baseline.
3. The PR-AUC difference is modest in validation, and only 74 frauds occur in test; do not over-interpret small metric differences.
4. Isolation Forest's test ROC-AUC looks strong, but AP and its cost-selected operating point are weak.
5. Temporal feature distributions differ materially; the configurable drift policy raises a retraining/review alert. Model score PSI alone remains below its threshold.
6. Pure concept drift requires labels/performance feedback to detect; PSI alone cannot guarantee concept-drift detection.

## Limitations

- Only two days of transactions are available; the time split is a benchmark simulation, not long-term seasonality monitoring.
- `V1`–`V28` are anonymized PCA components; interpretability, merchant/customer entities, and causal explanations are unavailable.
- The later test partition contains 74 frauds. Its precision/recall and threshold outcomes have uncertainty; confidence intervals and repeated time windows are future work.
- Exact duplicate rows are removed; in a true payments stream identical feature rows might represent separate events.
- Class-weighted scores are not calibrated probabilities. A production system should calibrate on a representative, out-of-time sample and account for analyst capacity, declines/reversals, recovery, fraud amount, and customer harm.
- PSI cutoffs and alert logic are heuristics. A real monitoring service needs stable reference windows, volume-aware limits, alert ownership, delayed-label monitoring, and a validated rollback/retraining procedure.
- The synthetic concept injection intentionally creates a new low-score fraud regime to stress the system; it is not evidence that the dataset contains this pattern.

## Future Improvements

- Calibrate probabilities and choose policy thresholds with real issuer loss and review-capacity estimates.
- Repeat temporal backtests across multiple windows; add bootstrap confidence intervals and per-amount/segment cost analysis.
- Add a data-contract/schema monitor, robust/windowed PSI, multivariate drift tests, and alert persistence/hysteresis.
- Evaluate LightGBM/XGBoost or calibrated ensembles with a constrained dependency/runtime budget.
- Add delayed-label monitoring, champion/challenger comparison, approval-gated retraining, and safe model registry/version rollback.
- Replace static thresholds with capacity-constrained top-k review policies and monitor segment fairness where lawful and data permit.

## Installation

Tested on Python 3.13; Python 3.10+ is required by type syntax and supported library releases.

```bash
git clone <your-public-repository-url>
cd fraud-anomaly-drift-monitoring
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download the dataset as described in [`data/README.md`](data/README.md).

## Usage

Run the complete pipeline from the repository root:

```bash
python scripts/run_pipeline.py --data data/raw/creditcard.csv
```

Optional controls:

```bash
python scripts/run_pipeline.py \
  --data data/raw/creditcard.csv \
  --output-dir . \
  --seed 42 \
  --fp-cost 5 \
  --fn-cost 90.37 \
  --skip-retraining-demo
```

If `--fn-cost` is omitted, it defaults to 80% of the mean fraud amount in the training window. Results are written under `reports/figures/` and `reports/results/`; fitted models are written locally under `models/` and ignored by git.

### Running the Notebook

Open `notebooks/fraud_detection_and_drift_monitoring.ipynb` in Jupyter and run all cells. The notebook calls the same end-to-end pipeline, then displays the generated tables and figures. It requires the dataset at `data/raw/creditcard.csv` or an edited `PipelineConfig` path.

### Running the Pipeline

The CLI runs data audit, EDA, split, both supervised candidates, Isolation Forest, validation-only cost thresholds, held-out test comparison, feature/model drift monitoring, concept-shift stress test, and the optional retraining prototype.

### Running Tests

```bash
python -m pytest -q
# or, after installing this project's test configuration:
pytest -q
```

Tests use generated toy data and do not require the large Kaggle CSV.

## Reproducibility

Random seeds are fixed (default 42) for estimators and simulations. Splits are time ordered with timestamp ties kept together. Preprocessing is embedded in fitted pipelines and fit on training/adaptation data only. Thresholds use validation data and the final test window is not used to select models, parameters, or thresholds. The complete generated run metadata is in `reports/results/run_summary.json`.

## Technical Write-up

See [`reports/technical_report.md`](reports/technical_report.md) for the concise academic-style report.

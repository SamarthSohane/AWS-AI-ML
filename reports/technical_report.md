# Technical Report: Fraud/Anomaly Detection with Drift Monitoring

## 1. Problem and Data

This study evaluates a binary transaction-fraud detector under extreme class imbalance and temporal distribution shift. A frozen detector may become ineffective as legitimate behavior or fraud tactics evolve; consequently, both ranking performance and operational monitoring are required. Experiments use the Kaggle Credit Card Fraud Detection benchmark, containing 284,807 European card transactions with 30 numerical inputs (`Time`, `V1`–`V28`, `Amount`) and the target `Class`. In the raw CSV, 492 records are fraudulent (0.1727%), with no missing values and 1,081 exact duplicate rows. Exact duplicates are removed before splitting, yielding 283,726 transactions and 473 frauds (0.1667%).

The data are ordered by elapsed time. A chronological 60/20/20 split assigns 170,235 earlier records to training (342 fraud), the next 56,745 to validation (57 fraud), and the latest 56,746 to test (74 fraud). Timestamp ties remain within one partition. This design reduces future-to-past leakage and intentionally exposes later-window shift, but the dataset covers only about two days and therefore is not a long-horizon deployment simulation. Median imputation, where needed, is fitted in each training pipeline only; Logistic Regression and Isolation Forest also standardize using the training data. No validation/test resampling is performed.

## 2. Modeling and Imbalance Strategy

A class-weighted Logistic Regression provides a linear baseline. HistGradientBoostingClassifier provides a regularized nonlinear candidate. Both use balanced class weights; the second model is chosen as the final supervised candidate by validation Average Precision (AP), not test results. Validation AP is 0.7729 for HistGradientBoosting versus 0.7727 for Logistic Regression, a near tie. The unsupervised detector is an Isolation Forest fit only on known-legitimate training features. Training labels are used to select the presumed-normal cohort, but are not passed to the detector objective; labels are used only for subsequent validation threshold calibration and held-out evaluation.

Accuracy is uninformative because an all-legitimate classifier would exceed 99.8% accuracy while detecting no fraud. ROC-AUC is reported as a complementary ranking measure but can appear high because the vast negative class dominates pairwise comparisons. AP/PR-AUC emphasizes precision as recall increases for the rare positive class and is therefore the primary comparison metric. No oversampling is used: class weighting preserves observed covariates, and the anonymized PCA features make blind SMOTE interpolation difficult to justify.

## 3. Threshold and Cost Method

Each model receives an operating threshold chosen by minimizing validation expected cost over its distinct validation scores. The threshold is frozen before testing. The explicit objective is `FP × C_FP + FN × C_FN`. The false-positive assumption is $5 per legitimate transaction blocked/reviewed: three minutes of analyst time at $60/hour plus a $2 customer-friction allowance. The default false-negative assumption is $90.37, computed as 80% unrecovered loss times the $112.96 mean fraud transaction amount in training. Users can override both costs. This constant-loss model is intentionally transparent but omits transaction-specific recovery and operational capacity; the report also records actual missed test fraud amount.

The selected HistGradientBoosting operating threshold is 0.881039. On validation it produces 3 false positives and 14 false negatives (precision 0.935; recall 0.754; estimated cost $1,280.16). Class weighting changes score calibration, so the numerical threshold should not be interpreted as a calibrated 88.1% fraud probability. The threshold is an empirical policy point for the validation window and stated cost assumptions.

## 4. Results

On the untouched later test window, the final HistGradientBoosting model achieves precision 0.859, recall 0.743, F1 0.797, AP 0.799, and ROC-AUC 0.982, with 9 false positives and 19 false negatives. Its expected cost is $1,762.01 under the assumed values; $2,886.13 of fraudulent transaction principal is present among missed test cases. The class-weighted Logistic Regression baseline reaches precision 0.724, recall 0.743, F1 0.733, AP 0.747, and ROC-AUC 0.982, with 21 FP and 19 FN. Isolation Forest reaches AP 0.036 and ROC-AUC 0.950. Its validation-cost threshold predicts no test fraud, yielding zero recall. This contrast illustrates why the high ROC-AUC alone does not establish operational usefulness in a rare-event problem.

## 5. Drift Monitoring

Feature drift compares the training reference to the final chronological test window using decile-bin PSI and two-sample KS statistics. KS p-values are corrected across 29 modeled features with Benjamini–Hochberg. The elapsed `Time` counter is omitted from actionable drift flags because its non-overlapping distributions are guaranteed by chronological splitting; it remains visible in the time EDA and split ranges. PSI below 0.10, 0.10–0.25, and at least 0.25 are treated as practical low, moderate, and significant bands. A feature in the moderate band is flagged only when adjusted KS q≤0.05. These values are monitoring heuristics, not universal statistical limits.

The run flags 16 of 29 features (55.2%), including 7 with PSI≥0.25 and 9 with moderate PSI; average PSI is 0.233 and maximum PSI is 1.463. The configured alert triggers on feature shift; validation-to-test model-score PSI is at most 0.026 and does not independently trigger. This separation is useful: feature drift may precede measured quality degradation, while model-output monitoring captures changes after feature processing. Score reference uses out-of-sample validation predictions rather than optimistic in-sample training predictions.

## 6. Synthetic Concept Drift and Retraining

A separate stress test preserves all feature values and synthetically changes labels within a low-scoring V4 tail (V4≤−2.0728). This region is identified from training feature quantiles and validation model scores, not labels. Ten percent of otherwise legitimate examples inside that region are relabeled as fraud; 362 synthetic positives are added to the 74 original test frauds. This is an intentionally constructed shift in `P(Y|X)`, not observed fraud. Because features and predictions are unchanged, PSI alone cannot detect it. A delayed-label performance monitor does: AP falls from 0.799 to 0.142 and recall at the frozen threshold falls from 0.743 to 0.126, exceeding the configured alert limits (AP drop 0.05 or recall drop 0.10).

A prototype retraining loop fits the selected model on historical training data plus the first 80% of the synthetic shifted validation window, selects a new threshold on the remaining calibration slice, and evaluates on the separate concept-shifted test window. AP rises from 0.142 to 0.255 and recall from 0.126 to 0.798. Precision decreases from 0.859 to 0.119; false positives increase from 9 to 2,580 and false negatives fall from 381 to 88. The illustrative total cost decreases from $34,475 to $20,852, but review volume rises sharply. This is an adaptation demonstration, not evidence that automated retraining is safe or generally beneficial.

## 7. Limitations and Future Work

The short observation period, anonymized PCA features, and 74 natural test frauds limit statistical precision and interpretability. Duplicate removal assumes identical full records are duplicate observations. Cost assumptions are not institution-specific, and weighted model scores are not calibrated. PSI/KS alerts require backtesting and monitoring ownership; statistical significance at large sample sizes does not establish practical impact. Synthetic concept drift is a deliberately adverse scenario. Further work should include multiple time-window backtests, uncertainty intervals, calibration, amount/segment-aware expected loss, analyst-capacity constraints, stable reference windows, delayed-label monitoring, approval-gated retraining, rollback, and a model registry.

## Reproduction

From the repository root, install `requirements.txt`, place the Kaggle CSV at `data/raw/creditcard.csv`, then run:

```bash
python scripts/run_pipeline.py --data data/raw/creditcard.csv
python -m pytest -q
```

Results are written to `reports/results/` and figures to `reports/figures/`; `reports/results/run_summary.json` records the split, operating points, costs, drift alerts, and comparison metrics.

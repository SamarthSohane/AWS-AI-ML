"""Fast unit tests for data preparation, fitting, evaluation, and monitoring."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DriftConfig
from src.data_preprocessing import load_transactions, split_transactions
from src.drift_detection import calculate_feature_drift, evaluate_drift_alert
from src.evaluation import CostModel, evaluate_predictions, select_threshold_by_cost
from src.supervised_model import train_supervised_models, positive_class_scores


def make_toy_transactions(n: int = 300, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    time = np.arange(n, dtype=float)
    fraud = (rng.random(n) < 0.12).astype(int)
    signal = rng.normal(loc=fraud * 2.0, scale=1.0, size=n)
    return pd.DataFrame(
        {
            "Time": time,
            "V1": signal,
            "Amount": np.maximum(0, rng.lognormal(2.0, 0.8, size=n)),
            "Class": fraud,
        }
    )


def test_loading_duplicate_handling_target_and_time_split(tmp_path):
    data = make_toy_transactions()
    data.loc[4, "Amount"] = np.nan  # feature missingness is imputed in a train-fitted pipeline
    data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    path = tmp_path / "toy.csv"
    data.to_csv(path, index=False)

    cleaned, audit = load_transactions(path, target_column="Class", deduplicate=True)
    assert "Class" in cleaned.columns
    assert audit["exact_duplicate_rows"] == 1
    assert len(cleaned) == len(data) - 1
    assert audit["missing_cells"] == 1

    splits = split_transactions(cleaned, target_column="Class", time_column="Time")
    assert len(splits.X_train) + len(splits.X_validation) + len(splits.X_test) == len(cleaned)
    assert splits.y_train.isin([0, 1]).all()
    assert splits.X_train["Time"].max() < splits.X_validation["Time"].min()
    assert splits.X_validation["Time"].max() < splits.X_test["Time"].min()
    assert splits.feature_columns == ["Time", "V1", "Amount"]


def test_supervised_fit_prediction_shape_and_metrics():
    data = make_toy_transactions(n=400)
    train, valid, test = data.iloc[:240], data.iloc[240:320], data.iloc[320:]
    models = train_supervised_models(train.drop(columns="Class"), train["Class"], random_seed=9)
    assert set(models) == {"logistic_regression", "hist_gradient_boosting"}
    for model in models.values():
        scores = positive_class_scores(model, test.drop(columns="Class"))
        assert scores.shape == (len(test),)
        assert np.isfinite(scores).all()
        metrics = evaluate_predictions(test["Class"], scores, threshold=0.5)
        assert {"precision", "recall", "f1", "pr_auc", "roc_auc", "false_positives", "false_negatives"}.issubset(metrics)
        assert 0 <= metrics["pr_auc"] <= 1


def test_threshold_selection_uses_cost_and_validation_scores():
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.40, 0.60, 0.30, 0.20])
    cost = CostModel(false_positive_cost=5.0, false_negative_cost=20.0)
    threshold, metrics = select_threshold_by_cost(y, scores, cost)
    assert np.isclose(threshold, 0.40)
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["estimated_total_cost"] == 0


def test_drift_metrics_and_alerting():
    rng = np.random.default_rng(7)
    reference = pd.DataFrame({"stable": rng.normal(0, 1, 2000), "shifted": rng.normal(0, 1, 2000)})
    current = pd.DataFrame({"stable": rng.normal(0, 1, 2000), "shifted": rng.normal(1.5, 1, 2000)})
    config = DriftConfig(
        feature_alert_fraction=0.20,
        average_psi_alert=0.10,
        score_psi_alert=10.0,
    )
    report = calculate_feature_drift(reference, current, config=config)
    assert set(report["feature"]) == {"stable", "shifted"}
    assert report.loc[report.feature == "shifted", "psi"].iloc[0] > report.loc[report.feature == "stable", "psi"].iloc[0]
    assert report["ks_q_value"].notna().all()
    alert = evaluate_drift_alert(report, pd.DataFrame(), config=config)
    assert alert["alert"] is True
    assert alert["status"] == "ALERT"

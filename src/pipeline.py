"""End-to-end reproducible experiment orchestration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .anomaly_detection import anomaly_scores, train_isolation_forest
from .config import PipelineConfig
from .data_preprocessing import load_transactions, split_transactions
from .drift_detection import (
    add_score_q_values,
    calculate_feature_drift,
    compare_score_distributions,
    evaluate_drift_alert,
)
from .evaluation import (
    CostModel,
    derive_cost_model,
    evaluate_predictions,
    select_threshold_by_cost,
    threshold_sweep,
)
from .simulation import (
    choose_low_risk_feature_region,
    inject_concept_drift,
    performance_drift_alert,
)
from .supervised_model import build_supervised_models, positive_class_scores, train_supervised_models
from .visualization import (
    plot_confusion_matrices,
    plot_eda,
    plot_feature_drift,
    plot_model_curves,
    plot_model_score_distributions,
    plot_performance_comparison,
    plot_threshold_analysis,
)

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression (class-weighted baseline)",
    "hist_gradient_boosting": "HistGradientBoosting (class-weighted)",
    "isolation_forest": "Isolation Forest (normal-only, unsupervised)",
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10g")


def _simple_metric_row(model: str, model_type: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {"model": model, "type": model_type, **metrics}


def _monitoring_text(alert: dict[str, Any], score_report: pd.DataFrame) -> str:
    score_text = "\n".join(
        f"  {row['model']}: PSI={row['psi']:.3f}; KS={row['ks_statistic']:.3f}; "
        f"predicted-positive rate {row.get('reference_predicted_positive_rate', float('nan')):.3%} → "
        f"{row.get('current_predicted_positive_rate', float('nan')):.3%}"
        for row in score_report.to_dict(orient="records")
    )
    return (
        "DRIFT MONITORING REPORT\n"
        "-----------------------\n"
        f"Features monitored: {alert['features_monitored']}\n"
        f"Features with significant drift (PSI ≥ 0.25): {alert['features_significant_psi']}\n"
        f"Features with moderate PSI (0.10–0.25): {alert['features_moderate_psi']}\n"
        f"Features flagged (fraction): {alert['features_flagged_fraction']:.1%}\n"
        f"Average PSI: {alert['average_psi']:.3f}\n"
        f"Maximum PSI: {alert['maximum_psi']:.3f}\n"
        f"Maximum model-score PSI: {alert['maximum_model_score_psi']:.3f}\n"
        "Model-output comparisons:\n"
        f"{score_text}\n"
        f"Drift status: {alert['status']}\n"
        f"Triggered by: {alert['triggered_by']}\n"
        f"Recommended action: {alert['recommended_action']}\n"
        "Note: PSI and KS thresholds are heuristics and require calibration to a production baseline.\n"
    )


def run_experiment(config: PipelineConfig | None = None) -> dict[str, Any]:
    """Run EDA, model training, validation threshold selection, test evaluation,
    drift monitoring, and the optional concept-drift/retraining stress test.

    The held-out test labels are used only for final reporting and a clearly
    isolated synthetic concept-drift stress test; they never choose model
    hyperparameters or operating thresholds.
    """
    cfg = config or PipelineConfig()
    output_root = Path(cfg.output_dir).expanduser().resolve()
    data_path = Path(cfg.data_path).expanduser()
    if not data_path.is_absolute():
        data_path = (Path.cwd() / data_path).resolve()
    results_dir = output_root / "reports" / "results"
    figures_dir = output_root / "reports" / "figures"
    models_dir = output_root / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    frame, audit = load_transactions(data_path, target_column=cfg.target_column, deduplicate=True)
    try:
        audit["source_path"] = str(data_path.relative_to(output_root))
    except ValueError:
        # Avoid writing a machine-specific absolute path into a shareable report.
        audit["source_path"] = data_path.name
    _write_json(results_dir / "dataset_audit.json", audit)
    plot_eda(frame, cfg.target_column, figures_dir)
    splits = split_transactions(
        frame,
        target_column=cfg.target_column,
        time_column=cfg.time_column,
        train_fraction=cfg.train_fraction,
        validation_fraction=cfg.validation_fraction,
    )
    _write_json(results_dir / "split_summary.json", splits.split_metadata)
    class_rows = []
    for name, labels in [
        ("train", splits.y_train),
        ("validation", splits.y_validation),
        ("test", splits.y_test),
    ]:
        class_rows.append(
            {
                "window": name,
                "rows": int(len(labels)),
                "fraud": int(labels.sum()),
                "legitimate": int((labels == 0).sum()),
                "fraud_fraction": float(labels.mean()),
            }
        )
    _save_csv(pd.DataFrame(class_rows), results_dir / "class_distribution_by_split.csv")

    # Train preprocessing and model parameters only on the historical training window.
    supervised_models = train_supervised_models(splits.X_train, splits.y_train, random_seed=cfg.random_seed)
    isolation_model = train_isolation_forest(
        splits.X_train,
        splits.y_train,
        random_seed=cfg.random_seed,
        n_estimators=cfg.isolation_estimators,
    )
    cost_model: CostModel = derive_cost_model(
        splits.X_train,
        splits.y_train,
        false_positive_cost=cfg.false_positive_cost,
        false_negative_cost_override=cfg.false_negative_cost_override,
        unrecovered_fraud_fraction=cfg.unrecovered_fraud_fraction,
        amount_column=cfg.amount_column,
    )
    _write_json(
        results_dir / "cost_assumptions.json",
        {
            "false_positive_cost": cost_model.false_positive_cost,
            "false_negative_cost": cost_model.false_negative_cost,
            "formula": "FP × C_FP + FN × C_FN",
            "derivation": cost_model.description,
            "unrecovered_fraud_fraction": cfg.unrecovered_fraud_fraction,
            "currency_note": "USD-equivalent illustrative assumptions; replace with institution-specific loss and review costs.",
        },
    )

    validation_scores: dict[str, np.ndarray] = {
        key: np.asarray(positive_class_scores(model, splits.X_validation), dtype=float)
        for key, model in supervised_models.items()
    }
    test_scores: dict[str, np.ndarray] = {
        key: np.asarray(positive_class_scores(model, splits.X_test), dtype=float)
        for key, model in supervised_models.items()
    }
    validation_scores["isolation_forest"] = np.asarray(anomaly_scores(isolation_model, splits.X_validation), dtype=float)
    test_scores["isolation_forest"] = np.asarray(anomaly_scores(isolation_model, splits.X_test), dtype=float)

    thresholds: dict[str, float] = {}
    threshold_selection: dict[str, dict[str, Any]] = {}
    for key, scores in validation_scores.items():
        threshold, validation_operating_metrics = select_threshold_by_cost(
            splits.y_validation, scores, cost_model
        )
        thresholds[key] = threshold
        threshold_selection[key] = {
            "threshold": threshold,
            "validation_metrics_at_selected_threshold": validation_operating_metrics,
            "validation_average_precision": float(average_precision_score(splits.y_validation, scores)),
        }
    # Select the supervised candidate by validation average precision. Cost is
    # used for its operating point, not to tune the held-out test set.
    supervised_keys = list(supervised_models)
    final_key = max(
        supervised_keys,
        key=lambda key: (
            threshold_selection[key]["validation_average_precision"],
            -threshold_selection[key]["validation_metrics_at_selected_threshold"]["estimated_total_cost"],
        ),
    )
    final_model = supervised_models[final_key]
    _write_json(
        results_dir / "selected_thresholds.json",
        {
            "thresholds": thresholds,
            "supervised_model_selected_by_validation_pr_auc": final_key,
            "selection_rule": "Highest validation Average Precision among supervised candidates; thresholds minimize validation expected cost.",
            "validation_selection_details": threshold_selection,
        },
    )

    # Threshold analysis is strictly validation-window analysis. The probability
    # grid includes the requested 0.10 ... 0.90 operating points and the selected
    # cost threshold. Isolation Forest uses score quantiles (not probabilities).
    threshold_frames: list[pd.DataFrame] = []
    probability_grid = list(np.round(np.arange(0.10, 0.91, 0.10), 10))
    for key in supervised_models:
        grid = sorted(set(probability_grid + [thresholds[key]]))
        table = threshold_sweep(
            splits.y_validation,
            validation_scores[key],
            grid,
            DISPLAY_NAMES[key],
            cost_model=cost_model,
            amounts=splits.X_validation[cfg.amount_column].to_numpy() if cfg.amount_column in splits.X_validation else None,
        )
        table["score_type"] = "probability"
        table["selected_cost_threshold"] = np.isclose(table["threshold"], thresholds[key], rtol=1e-10, atol=1e-12)
        table["threshold_quantile"] = np.nan
        threshold_frames.append(table)
    if_key = "isolation_forest"
    if_quantiles = [0.90, 0.95, 0.975, 0.99, 0.995, 0.999]
    if_grid = sorted(set(float(np.quantile(validation_scores[if_key], q)) for q in if_quantiles) | {thresholds[if_key]})
    if_table = threshold_sweep(
        splits.y_validation,
        validation_scores[if_key],
        if_grid,
        DISPLAY_NAMES[if_key],
        cost_model=cost_model,
        amounts=splits.X_validation[cfg.amount_column].to_numpy() if cfg.amount_column in splits.X_validation else None,
    )
    if_table["score_type"] = "anomaly_score"
    if_table["selected_cost_threshold"] = np.isclose(if_table["threshold"], thresholds[if_key], rtol=1e-10, atol=1e-12)
    if_table["threshold_quantile"] = [float(np.mean(validation_scores[if_key] <= value)) for value in if_table["threshold"]]
    threshold_frames.append(if_table)
    threshold_table = pd.concat(threshold_frames, ignore_index=True, sort=False)
    _save_csv(threshold_table, results_dir / "threshold_analysis.csv")

    # All held-out test metrics use validation-selected thresholds.
    test_metric_rows: list[dict[str, Any]] = []
    validation_metric_rows: list[dict[str, Any]] = []
    test_metrics_by_name: dict[str, dict[str, Any]] = {}
    score_by_display: dict[str, np.ndarray] = {}
    for key in ["logistic_regression", "hist_gradient_boosting", "isolation_forest"]:
        display_name = DISPLAY_NAMES[key]
        test_metrics = evaluate_predictions(
            splits.y_test,
            test_scores[key],
            thresholds[key],
            cost_model=cost_model,
            amounts=splits.X_test[cfg.amount_column].to_numpy() if cfg.amount_column in splits.X_test else None,
        )
        validation_metrics = evaluate_predictions(
            splits.y_validation,
            validation_scores[key],
            thresholds[key],
            cost_model=cost_model,
            amounts=splits.X_validation[cfg.amount_column].to_numpy() if cfg.amount_column in splits.X_validation else None,
        )
        kind = "Unsupervised anomaly detector" if key == "isolation_forest" else "Supervised classifier"
        test_metric_rows.append(_simple_metric_row(display_name, kind, test_metrics))
        validation_metric_rows.append(_simple_metric_row(display_name, kind, validation_metrics))
        test_metrics_by_name[display_name] = test_metrics
        score_by_display[display_name] = test_scores[key]
    comparison = pd.DataFrame(test_metric_rows)
    validation_comparison = pd.DataFrame(validation_metric_rows)
    _save_csv(comparison, results_dir / "model_comparison.csv")
    _save_csv(validation_comparison, results_dir / "validation_model_comparison.csv")
    _save_csv(
        pd.DataFrame(
            [
                {
                    "model": DISPLAY_NAMES[key],
                    "model_key": key,
                    "validation_pr_auc": threshold_selection[key]["validation_average_precision"],
                    "validation_selected_threshold": thresholds[key],
                    "validation_expected_cost": threshold_selection[key]["validation_metrics_at_selected_threshold"]["estimated_total_cost"],
                    "selected_as_final_supervised": bool(key == final_key),
                }
                for key in ["logistic_regression", "hist_gradient_boosting", "isolation_forest"]
            ]
        ),
        results_dir / "cost_threshold_summary.csv",
    )

    plot_model_curves(splits.y_test, score_by_display, figures_dir)
    plot_confusion_matrices(splits.y_test, test_metrics_by_name, figures_dir)
    plot_threshold_analysis(threshold_table, DISPLAY_NAMES[final_key], figures_dir)

    # Feature drift: train is the historical reference; the chronological test
    # window is current. Multiple KS tests use Benjamini-Hochberg correction.
    # The raw Time field is an elapsed-time index. Its distribution must move
    # between non-overlapping chronological windows, so it is reported in EDA
    # and split metadata but excluded from actionable feature-drift alerts.
    drift_features = [column for column in splits.feature_columns if column != cfg.time_column]
    feature_report = calculate_feature_drift(
        splits.X_train,
        splits.X_test,
        features=drift_features,
        config=cfg.drift,
    )
    _save_csv(feature_report, results_dir / "drift_report.csv")
    # Use validation predictions as the score-reference cohort: these are
    # out-of-sample scores from the frozen training model, unlike in-sample
    # training scores, which would make a biased confidence baseline.
    reference_model_scores: dict[str, np.ndarray] = {
        key: np.asarray(positive_class_scores(model, splits.X_validation), dtype=float)
        for key, model in supervised_models.items()
    }
    reference_model_scores["isolation_forest"] = np.asarray(anomaly_scores(isolation_model, splits.X_validation), dtype=float)
    score_rows = []
    score_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key in ["logistic_regression", "hist_gradient_boosting", "isolation_forest"]:
        row = compare_score_distributions(
            reference_model_scores[key],
            test_scores[key],
            DISPLAY_NAMES[key],
            decision_threshold=thresholds[key],
            bins=cfg.drift.psi_bins,
        )
        score_rows.append(row)
        score_arrays[DISPLAY_NAMES[key]] = (reference_model_scores[key], test_scores[key])
    score_report = add_score_q_values(pd.DataFrame(score_rows))
    _save_csv(score_report, results_dir / "model_score_drift.csv")
    drift_alert = evaluate_drift_alert(feature_report, score_report, cfg.drift)
    _write_json(results_dir / "drift_alert.json", drift_alert)
    (results_dir / "drift_monitoring_report.txt").write_text(
        _monitoring_text(drift_alert, score_report), encoding="utf-8"
    )
    plot_feature_drift(feature_report, splits.X_train, splits.X_test, figures_dir)
    plot_model_score_distributions(score_arrays, figures_dir)

    # Bonus stress test: feature values are untouched, but the fraud mechanism is
    # altered within a previously low-score feature tail. This is synthetic and
    # intentionally separate from primary test metrics.
    concept_region = choose_low_risk_feature_region(
        splits.X_train,
        splits.X_validation,
        validation_scores[final_key],
        tail_fraction=cfg.concept_region_tail_fraction,
    )
    y_validation_concept, validation_injection = inject_concept_drift(
        splits.X_validation,
        splits.y_validation,
        concept_region,
        injection_rate_within_region=cfg.concept_injection_rate,
        random_seed=cfg.random_seed + 101,
    )
    y_test_concept, test_injection = inject_concept_drift(
        splits.X_test,
        splits.y_test,
        concept_region,
        injection_rate_within_region=cfg.concept_injection_rate,
        random_seed=cfg.random_seed + 202,
    )
    _write_json(
        results_dir / "concept_drift_injection.json",
        {
            "description": (
                "Synthetic concept-drift stress test: preserve every feature value, then increase the fraud-label "
                "incidence in a feature-defined tail that had low mean score on validation. The chosen region used "
                "training quantiles and validation scores only, not labels. This is not observed fraud data."
            ),
            "region_rule": concept_region,
            "validation_injection": validation_injection,
            "test_injection": test_injection,
        },
    )
    baseline_test_metrics = test_metrics_by_name[DISPLAY_NAMES[final_key]]
    concept_test_metrics = evaluate_predictions(
        y_test_concept,
        test_scores[final_key],
        thresholds[final_key],
        cost_model=cost_model,
        amounts=splits.X_test[cfg.amount_column].to_numpy() if cfg.amount_column in splits.X_test else None,
    )
    performance_alert = performance_drift_alert(
        baseline_test_metrics,
        concept_test_metrics,
        ap_drop_threshold=cfg.drift.performance_ap_drop_alert,
        recall_drop_threshold=cfg.drift.performance_recall_drop_alert,
    )
    _write_json(results_dir / "concept_drift_performance_alert.json", performance_alert)
    concept_comparison = pd.DataFrame(
        [
            {"scenario": "Observed held-out test labels", **baseline_test_metrics},
            {"scenario": "Synthetic concept drift; frozen model", **concept_test_metrics},
        ]
    )
    _save_csv(concept_comparison, results_dir / "concept_drift_performance.csv")
    plot_performance_comparison(
        concept_comparison,
        figures_dir / "concept_drift_performance.png",
        "Performance monitor under synthetic concept drift",
        ["scenario"],
    )

    retraining_comparison = pd.DataFrame()
    retraining_summary: dict[str, Any] | None = None
    if cfg.run_retraining_demo:
        # Use the first 80% of the synthetic later validation window as labeled
        # adaptation data and its remaining 20% only to calibrate a new threshold.
        # The final synthetic test window remains isolated for evaluation.
        adaptation_end = int(len(splits.X_validation) * cfg.concept_adaptation_fraction)
        adaptation_end = min(max(adaptation_end, 1), len(splits.X_validation) - 1)
        X_adapt = splits.X_validation.iloc[:adaptation_end]
        y_adapt = y_validation_concept[:adaptation_end]
        X_calibrate = splits.X_validation.iloc[adaptation_end:]
        y_calibrate = y_validation_concept[adaptation_end:]
        X_retrain = pd.concat([splits.X_train, X_adapt], ignore_index=True)
        y_retrain = np.concatenate([splits.y_train.to_numpy(), y_adapt])
        retraining_model = build_supervised_models(random_seed=cfg.random_seed)[final_key]
        retraining_model.fit(X_retrain, y_retrain)
        calibration_scores = np.asarray(positive_class_scores(retraining_model, X_calibrate), dtype=float)
        retraining_threshold, retraining_calibration_metrics = select_threshold_by_cost(
            y_calibrate, calibration_scores, cost_model
        )
        retrained_test_scores = np.asarray(positive_class_scores(retraining_model, splits.X_test), dtype=float)
        retrained_metrics = evaluate_predictions(
            y_test_concept,
            retrained_test_scores,
            retraining_threshold,
            cost_model=cost_model,
            amounts=splits.X_test[cfg.amount_column].to_numpy() if cfg.amount_column in splits.X_test else None,
        )
        frozen_row = {"model_state": "Frozen model (threshold from original validation)", **concept_test_metrics}
        retrained_row = {
            "model_state": "Retrained on historical + labeled concept window; threshold from separate concept calibration slice",
            **retrained_metrics,
        }
        retraining_comparison = pd.DataFrame([frozen_row, retrained_row])
        _save_csv(retraining_comparison, results_dir / "retraining_comparison.csv")
        retraining_summary = {
            "adaptation_rows": int(len(X_adapt)),
            "calibration_rows": int(len(X_calibrate)),
            "test_rows": int(len(splits.X_test)),
            "retrained_threshold": float(retraining_threshold),
            "calibration_metrics": retraining_calibration_metrics,
            "result_note": "Prototype only; the synthetic concept labels are not production ground truth.",
        }
        _write_json(results_dir / "retraining_summary.json", retraining_summary)
        plot_performance_comparison(
            retraining_comparison,
            figures_dir / "retraining_performance_comparison.png",
            "Retraining prototype on a synthetic concept-shifted window",
            ["model_state"],
        )
    else:
        _save_csv(retraining_comparison, results_dir / "retraining_comparison.csv")

    if cfg.save_models:
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(supervised_models, models_dir / "supervised_candidates.joblib", compress=3)
        joblib.dump(isolation_model, models_dir / "isolation_forest.joblib", compress=3)
        joblib.dump(final_model, models_dir / "final_supervised_model.joblib", compress=3)
        _write_json(
            models_dir / "model_metadata.json",
            {
                "selected_supervised_key": final_key,
                "selected_supervised_display_name": DISPLAY_NAMES[final_key],
                "selected_threshold": thresholds[final_key],
                "isolation_threshold": thresholds["isolation_forest"],
                "random_seed": cfg.random_seed,
                "training_rows": int(len(splits.X_train)),
                "feature_columns": splits.feature_columns,
                "warning": "joblib/pickle model files must only be loaded from trusted sources.",
            },
        )

    run_summary = {
        "dataset": {
            "raw_rows": audit["rows"],
            "raw_columns": audit["columns"],
            "raw_fraud_count": audit["fraud_count"],
            "raw_fraud_fraction": audit["fraud_fraction"],
            "duplicates_removed": audit["duplicates_removed"],
            "rows_after_cleaning": audit["rows_after_duplicate_handling"],
        },
        "split": splits.split_metadata,
        "final_supervised_model": final_key,
        "final_supervised_threshold": thresholds[final_key],
        "isolation_forest_threshold": thresholds["isolation_forest"],
        "cost_model": {
            "false_positive_cost": cost_model.false_positive_cost,
            "false_negative_cost": cost_model.false_negative_cost,
            "description": cost_model.description,
        },
        "test_comparison": comparison.to_dict(orient="records"),
        "drift_alert": drift_alert,
        "concept_drift_performance_alert": performance_alert,
        "retraining_demo": retraining_summary,
        "reproducibility": {"random_seed": cfg.random_seed},
    }
    _write_json(results_dir / "run_summary.json", run_summary)

    print("\n=== Experiment complete ===")
    print(f"Rows after exact deduplication: {len(frame):,}; raw fraud rate: {audit['fraud_fraction']:.4%}")
    print(f"Final supervised candidate (validation PR-AUC): {final_key}")
    print(f"Operating threshold (validation cost minimum): {thresholds[final_key]:.8g}")
    print(f"Expected-cost assumptions: FP=${cost_model.false_positive_cost:.2f}, FN=${cost_model.false_negative_cost:.2f}")
    print("Held-out test comparison:")
    print(comparison[["model", "precision", "recall", "f1", "pr_auc", "roc_auc", "false_positives", "false_negatives"]].to_string(index=False))
    print(f"Feature/model drift status: {drift_alert['status']} (avg PSI={drift_alert['average_psi']:.3f})")
    print(f"Synthetic concept-drift performance status: {performance_alert['status']}")
    print("Outputs: reports/figures/ and reports/results/")
    return {
        "config": cfg,
        "audit": audit,
        "splits": splits,
        "models": supervised_models,
        "isolation_model": isolation_model,
        "cost_model": cost_model,
        "thresholds": thresholds,
        "selected_model_key": final_key,
        "comparison": comparison,
        "validation_comparison": validation_comparison,
        "threshold_analysis": threshold_table,
        "drift_report": feature_report,
        "score_drift_report": score_report,
        "drift_alert": drift_alert,
        "concept_drift_performance_alert": performance_alert,
        "retraining_comparison": retraining_comparison,
        "run_summary": run_summary,
    }

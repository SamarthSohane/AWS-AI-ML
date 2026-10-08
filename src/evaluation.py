"""Imbalance-aware metrics, validation-only threshold selection, and cost analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score


@dataclass(frozen=True)
class CostModel:
    false_positive_cost: float
    false_negative_cost: float
    description: str = ""

    def total(self, false_positives: int, false_negatives: int) -> float:
        return float(false_positives * self.false_positive_cost + false_negatives * self.false_negative_cost)


def derive_cost_model(
    X_train,
    y_train,
    false_positive_cost: float = 5.0,
    false_negative_cost_override: float | None = None,
    unrecovered_fraud_fraction: float = 0.80,
    amount_column: str = "Amount",
) -> CostModel:
    """Derive a simple transparent expected-cost model from the training window.

    FP cost is an operational assumption (3 minutes of review at $60/hour plus
    $2 estimated customer friction = $5). FN cost defaults to 80% of the mean
    *training-window fraud amount*, as a proxy for the unrecovered principal.
    Both inputs can be overridden at run time; no validation/test outcomes are
    used to define the monetary assumptions.
    """
    if false_negative_cost_override is not None:
        fn_cost = float(false_negative_cost_override)
        description = "FN cost explicitly overridden by the run configuration"
    elif amount_column in X_train.columns and bool((y_train == 1).any()):
        mean_fraud_amount = float(pd.to_numeric(X_train.loc[y_train == 1, amount_column], errors="coerce").mean())
        if not np.isfinite(mean_fraud_amount):
            mean_fraud_amount = 125.0
        fn_cost = mean_fraud_amount * float(unrecovered_fraud_fraction)
        description = (
            f"FN cost = {unrecovered_fraud_fraction:.0%} × mean fraud transaction amount "
            "in the training window"
        )
    else:
        fn_cost = 100.0 * float(unrecovered_fraud_fraction)
        description = "Fallback FN cost assumption; no training fraud amount was available"
    return CostModel(
        false_positive_cost=float(false_positive_cost),
        false_negative_cost=float(fn_cost),
        description=description,
    )


def evaluate_predictions(
    y_true,
    scores,
    threshold: float,
    cost_model: CostModel | None = None,
    amounts=None,
) -> dict[str, float | int]:
    """Calculate threshold and ranking metrics for a binary fraud task."""
    y = np.asarray(y_true, dtype=int).reshape(-1)
    score = np.asarray(scores, dtype=float).reshape(-1)
    if len(y) != len(score):
        raise ValueError("y_true and scores must have equal length")
    if not np.isfinite(score).all():
        raise ValueError("Scores contain NaN or infinite values")
    pred = (score >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = float(precision_score(y, pred, zero_division=0))
    recall = float(recall_score(y, pred, zero_division=0))
    f1 = float(f1_score(y, pred, zero_division=0))
    result: dict[str, float | int] = {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else float("nan"),
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan"),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "predicted_positive_rate": float(pred.mean()) if len(pred) else 0.0,
        "actual_positive_rate": float(y.mean()) if len(y) else 0.0,
    }
    if amounts is not None:
        amount_array = np.asarray(amounts, dtype=float).reshape(-1)
        if len(amount_array) != len(y):
            raise ValueError("amounts and y_true must have equal length")
        result["missed_fraud_amount"] = float(np.nansum(amount_array[(y == 1) & (pred == 0)]))
    if cost_model is not None:
        result["false_positive_cost"] = float(cost_model.false_positive_cost)
        result["false_negative_cost"] = float(cost_model.false_negative_cost)
        result["estimated_total_cost"] = cost_model.total(int(fp), int(fn))
    return result


def select_threshold_by_cost(
    y_true,
    scores,
    cost_model: CostModel,
) -> tuple[float, dict[str, float | int]]:
    """Find the exact empirical validation threshold with minimum expected cost.

    Every distinct validation score is a decision boundary. The function never
    receives or inspects test data; callers must pass validation scores only.
    Ties are resolved in favor of the higher-recall operating point.
    """
    y = np.asarray(y_true, dtype=int).reshape(-1)
    score = np.asarray(scores, dtype=float).reshape(-1)
    if len(y) != len(score) or len(y) == 0:
        raise ValueError("y_true and scores must be non-empty arrays of equal length")
    if not np.isfinite(score).all():
        raise ValueError("Scores contain NaN or infinite values")
    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_y = y[order]
    cumulative_tp = np.cumsum(sorted_y)
    cumulative_fp = np.cumsum(1 - sorted_y)
    group_ends = np.flatnonzero(np.r_[sorted_score[1:] != sorted_score[:-1], True])
    thresholds = sorted_score[group_ends]
    tp = cumulative_tp[group_ends]
    fp = cumulative_fp[group_ends]
    fn = int(y.sum()) - tp
    costs = fp * cost_model.false_positive_cost + fn * cost_model.false_negative_cost
    best_cost = float(np.min(costs))
    best = np.flatnonzero(np.isclose(costs, best_cost, rtol=1e-12, atol=1e-9))
    recalls = tp[best] / max(int(y.sum()), 1)
    best_index = int(best[np.argmax(recalls)])
    threshold = float(thresholds[best_index])
    details = evaluate_predictions(y, score, threshold, cost_model=cost_model)
    return threshold, details


def threshold_sweep(
    y_true,
    scores,
    thresholds: Iterable[float],
    model_name: str,
    cost_model: CostModel | None = None,
    amounts=None,
) -> pd.DataFrame:
    """Evaluate an explicit threshold grid; intended for validation analysis."""
    rows = []
    for threshold in thresholds:
        row = evaluate_predictions(y_true, scores, float(threshold), cost_model, amounts)
        row["model"] = model_name
        rows.append(row)
    return pd.DataFrame(rows)

"""Documented, deterministic stress tests for covariate/concept drift."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def choose_low_risk_feature_region(
    X_reference: pd.DataFrame,
    X_validation: pd.DataFrame,
    validation_scores,
    tail_fraction: float = 0.05,
) -> dict[str, Any]:
    """Choose a feature-defined low-score tail using features and scores only.

    The rule is learned from the training feature quantiles and validation
    predictions, never from validation/test labels. It is used only to define a
    deliberately artificial new-fraud regime for a concept-drift stress test.
    """
    scores = np.asarray(validation_scores, dtype=float).reshape(-1)
    if len(scores) != len(X_validation):
        raise ValueError("validation_scores and X_validation must have equal length")
    if not 0 < tail_fraction < 0.25:
        raise ValueError("tail_fraction must be in (0, 0.25)")
    candidates = [c for c in X_reference.columns if c in X_validation.columns and c not in {"Time", "Amount"}]
    if not candidates:
        candidates = [c for c in X_reference.columns if c in X_validation.columns]
    best: dict[str, Any] | None = None
    for feature in candidates:
        ref = pd.to_numeric(X_reference[feature], errors="coerce").dropna()
        val = pd.to_numeric(X_validation[feature], errors="coerce").to_numpy(dtype=float)
        if not len(ref):
            continue
        low_cut = float(ref.quantile(tail_fraction))
        high_cut = float(ref.quantile(1 - tail_fraction))
        for direction, cut in (("low", low_cut), ("high", high_cut)):
            mask = val <= cut if direction == "low" else val >= cut
            count = int(mask.sum())
            # Avoid a degenerate region caused by ties or a constant feature.
            if count < max(20, int(0.01 * len(val))) or count > int(0.25 * len(val)):
                continue
            mean_score = float(np.mean(scores[mask]))
            candidate = {
                "feature": str(feature),
                "direction": direction,
                "cut": cut,
                "training_quantile": float(tail_fraction if direction == "low" else 1 - tail_fraction),
                "validation_region_rows": count,
                "validation_region_fraction": float(count / len(val)),
                "validation_mean_score": mean_score,
                "validation_score_median": float(np.median(scores)),
            }
            if best is None or candidate["validation_mean_score"] < best["validation_mean_score"]:
                best = candidate
    if best is None:
        raise ValueError("Could not find a non-degenerate feature tail for concept-drift simulation")
    return best


def region_mask(X: pd.DataFrame, region: dict[str, Any]) -> np.ndarray:
    feature = region["feature"]
    if feature not in X:
        raise ValueError(f"Concept-drift feature {feature!r} not present in the window")
    values = pd.to_numeric(X[feature], errors="coerce").to_numpy(dtype=float)
    return values <= float(region["cut"]) if region["direction"] == "low" else values >= float(region["cut"])


def inject_concept_drift(
    X: pd.DataFrame,
    y,
    region: dict[str, Any],
    injection_rate_within_region: float = 0.10,
    random_seed: int = 42,
) -> tuple[np.ndarray, dict[str, int | float | str]]:
    """Relabel a sampled fraction of known-legitimate records inside a regime.

    Features remain unchanged. This changes P(Y|X) by creating a previously
    low-risk feature segment with an elevated synthetic fraud incidence. This is
    explicitly a stress-test label intervention, not a claim about real fraud.
    """
    if not 0 < injection_rate_within_region < 1:
        raise ValueError("injection_rate_within_region must be in (0, 1)")
    y_original = np.asarray(y, dtype=int).reshape(-1)
    if len(y_original) != len(X):
        raise ValueError("X and y must have equal length")
    eligible_mask = region_mask(X, region) & (y_original == 0)
    eligible = np.flatnonzero(eligible_mask)
    n_injected = int(round(len(eligible) * injection_rate_within_region))
    rng = np.random.default_rng(random_seed)
    injected = rng.choice(eligible, size=n_injected, replace=False) if n_injected else np.array([], dtype=int)
    y_shifted = y_original.copy()
    y_shifted[injected] = 1
    report: dict[str, int | float | str] = {
        "feature": str(region["feature"]),
        "direction": str(region["direction"]),
        "cut": float(region["cut"]),
        "region_rows": int(region_mask(X, region).sum()),
        "eligible_legitimate_rows": int(len(eligible)),
        "synthetic_frauds_added": int(n_injected),
        "injection_rate_within_eligible_region": float(injection_rate_within_region),
        "base_fraud_count": int(y_original.sum()),
        "shifted_fraud_count": int(y_shifted.sum()),
        "base_fraud_fraction": float(y_original.mean()),
        "shifted_fraud_fraction": float(y_shifted.mean()),
    }
    return y_shifted, report


def performance_drift_alert(
    reference_metrics: dict[str, float | int],
    current_metrics: dict[str, float | int],
    ap_drop_threshold: float = 0.05,
    recall_drop_threshold: float = 0.10,
) -> dict[str, float | bool | str]:
    """Check delayed-label performance degradation for a concept-drift signal."""
    ap_drop = float(reference_metrics["pr_auc"] - current_metrics["pr_auc"])
    recall_drop = float(reference_metrics["recall"] - current_metrics["recall"])
    alert = bool(ap_drop >= ap_drop_threshold or recall_drop >= recall_drop_threshold)
    return {
        "reference_pr_auc": float(reference_metrics["pr_auc"]),
        "current_pr_auc": float(current_metrics["pr_auc"]),
        "pr_auc_drop": ap_drop,
        "reference_recall": float(reference_metrics["recall"]),
        "current_recall": float(current_metrics["recall"]),
        "recall_drop": recall_drop,
        "pr_auc_drop_threshold": float(ap_drop_threshold),
        "recall_drop_threshold": float(recall_drop_threshold),
        "performance_alert": alert,
        "status": "ALERT" if alert else "NORMAL",
        "recommended_action": "Investigate concept drift and retrain after labels are validated." if alert else "Continue monitoring labeled outcomes.",
    }

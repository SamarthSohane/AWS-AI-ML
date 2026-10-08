"""Feature and model-output drift monitoring using PSI, KS, and BH correction."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .config import DriftConfig


def _finite_values(values) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def population_stability_index(reference, current, bins: int = 10, epsilon: float = 1e-6) -> float:
    """Compute PSI using quantile cut points learned from the reference sample."""
    ref = _finite_values(reference)
    cur = _finite_values(current)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    interior = np.unique(np.quantile(ref, quantiles))
    edges = np.r_[-np.inf, interior, np.inf]
    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float)
    ref_pct = (ref_counts + epsilon) / (ref_counts.sum() + epsilon * len(ref_counts))
    cur_pct = (cur_counts + epsilon) / (cur_counts.sum() + epsilon * len(cur_counts))
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate adjusted p-values."""
    p = np.asarray(list(p_values), dtype=float)
    result = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    pv = p[valid]
    if not len(pv):
        return result
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adjusted_sorted = ranked * m / np.arange(1, m + 1)
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    result[valid] = adjusted
    return result


def calculate_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: Iterable[str] | None = None,
    config: DriftConfig | None = None,
) -> pd.DataFrame:
    """Compare numeric distributions; KS p-values are BH-adjusted across features.

    PSI is treated as an effect-size heuristic. KS testing is sensitive to large
    sample sizes, so practical flags require a non-trivial PSI as well as an
    adjusted p-value for the moderate-drift band.
    """
    cfg = config or DriftConfig()
    if features is None:
        features = [
            c for c in reference.columns
            if c in current.columns
            and pd.api.types.is_numeric_dtype(reference[c])
            and pd.api.types.is_numeric_dtype(current[c])
        ]
    feature_list = list(features)
    rows = []
    for feature in feature_list:
        ref = _finite_values(reference[feature])
        cur = _finite_values(current[feature])
        psi = population_stability_index(ref, cur, bins=cfg.psi_bins)
        if len(ref) and len(cur):
            ks = ks_2samp(ref, cur, alternative="two-sided", method="asymp")
            ks_statistic, p_value = float(ks.statistic), float(ks.pvalue)
        else:
            ks_statistic, p_value = float("nan"), float("nan")
        rows.append(
            {
                "feature": str(feature),
                "reference_mean": float(np.mean(ref)) if len(ref) else float("nan"),
                "current_mean": float(np.mean(cur)) if len(cur) else float("nan"),
                "reference_std": float(np.std(ref)) if len(ref) else float("nan"),
                "current_std": float(np.std(cur)) if len(cur) else float("nan"),
                "psi": psi,
                "ks_statistic": ks_statistic,
                "ks_p_value": p_value,
                "reference_n": int(len(ref)),
                "current_n": int(len(cur)),
            }
        )
    report = pd.DataFrame(rows)
    if report.empty:
        return report
    report["ks_q_value"] = benjamini_hochberg(report["ks_p_value"].to_numpy())
    report["drift_level"] = "none"
    report.loc[report["psi"] >= cfg.moderate_psi, "drift_level"] = "moderate"
    report.loc[report["psi"] >= cfg.significant_psi, "drift_level"] = "significant"
    moderate_supported = (report["psi"] >= cfg.moderate_psi) & (report["ks_q_value"] <= cfg.alpha)
    report["drift_flag"] = (report["psi"] >= cfg.significant_psi) | moderate_supported
    # If the effect is significant by PSI, the flag is independent of a p-value;
    # if moderate, it additionally needs FDR-adjusted statistical support.
    report = report.sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)
    return report


def compare_score_distributions(
    reference_scores,
    current_scores,
    model_name: str,
    decision_threshold: float | None = None,
    bins: int = 10,
) -> dict[str, float | str]:
    """Monitor prediction/anomaly-score distributions and operating class rates."""
    ref = _finite_values(reference_scores)
    cur = _finite_values(current_scores)
    if not len(ref) or not len(cur):
        raise ValueError("Both reference and current scores must contain finite values")
    ks = ks_2samp(ref, cur, alternative="two-sided", method="asymp")
    row: dict[str, float | str] = {
        "model": model_name,
        "reference_mean": float(np.mean(ref)),
        "current_mean": float(np.mean(cur)),
        "reference_std": float(np.std(ref)),
        "current_std": float(np.std(cur)),
        "psi": population_stability_index(ref, cur, bins=bins),
        "ks_statistic": float(ks.statistic),
        "ks_p_value": float(ks.pvalue),
        "reference_n": int(len(ref)),
        "current_n": int(len(cur)),
    }
    if decision_threshold is not None:
        row["reference_predicted_positive_rate"] = float(np.mean(ref >= decision_threshold))
        row["current_predicted_positive_rate"] = float(np.mean(cur >= decision_threshold))
    return row


def add_score_q_values(score_report: pd.DataFrame) -> pd.DataFrame:
    """Adjust score-distribution KS p-values across monitored outputs."""
    result = score_report.copy()
    if len(result):
        result["ks_q_value"] = benjamini_hochberg(result["ks_p_value"].to_numpy())
    return result


def evaluate_drift_alert(
    feature_report: pd.DataFrame,
    score_report: pd.DataFrame,
    config: DriftConfig | None = None,
) -> dict[str, float | int | bool | str]:
    """Apply configurable heuristic alert rules and provide an action string."""
    cfg = config or DriftConfig()
    if feature_report.empty:
        average_psi = maximum_psi = feature_fraction = 0.0
        feature_count = significant_count = moderate_count = 0
    else:
        psi_values = pd.to_numeric(feature_report["psi"], errors="coerce").dropna()
        average_psi = float(psi_values.mean()) if len(psi_values) else 0.0
        maximum_psi = float(psi_values.max()) if len(psi_values) else 0.0
        feature_count = int(len(feature_report))
        significant_count = int((feature_report["psi"] >= cfg.significant_psi).sum())
        moderate_count = int(((feature_report["psi"] >= cfg.moderate_psi) & (feature_report["psi"] < cfg.significant_psi)).sum())
        feature_fraction = float(feature_report["drift_flag"].astype(bool).mean()) if feature_count else 0.0
    max_score_psi = (
        float(pd.to_numeric(score_report["psi"], errors="coerce").max())
        if len(score_report) and score_report["psi"].notna().any()
        else 0.0
    )
    fraction_trigger = feature_fraction >= cfg.feature_alert_fraction
    average_trigger = average_psi >= cfg.average_psi_alert
    score_trigger = max_score_psi >= cfg.score_psi_alert
    alert = bool(fraction_trigger or average_trigger or score_trigger)
    return {
        "features_monitored": feature_count,
        "features_significant_psi": significant_count,
        "features_moderate_psi": moderate_count,
        "features_flagged_fraction": feature_fraction,
        "average_psi": average_psi,
        "maximum_psi": maximum_psi,
        "maximum_model_score_psi": max_score_psi,
        "feature_fraction_threshold": float(cfg.feature_alert_fraction),
        "average_psi_threshold": float(cfg.average_psi_alert),
        "model_score_psi_threshold": float(cfg.score_psi_alert),
        "alert": alert,
        "status": "ALERT" if alert else "NORMAL",
        "recommended_action": "Model retraining/review recommended." if alert else "Continue monitoring; no immediate action.",
        "triggered_by": ", ".join(
            name for name, fired in [
                ("flagged feature fraction", fraction_trigger),
                ("average feature PSI", average_trigger),
                ("model-output PSI", score_trigger),
            ] if fired
        ) or "none",
    }

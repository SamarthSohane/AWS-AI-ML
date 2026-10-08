"""Central configuration for the fraud and drift monitoring experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DriftConfig:
    """Practical monitoring heuristics; these are not universal constants."""

    psi_bins: int = 10
    alpha: float = 0.05
    moderate_psi: float = 0.10
    significant_psi: float = 0.25
    feature_alert_fraction: float = 0.20
    average_psi_alert: float = 0.10
    score_psi_alert: float = 0.10
    performance_ap_drop_alert: float = 0.05
    performance_recall_drop_alert: float = 0.10


@dataclass(frozen=True)
class PipelineConfig:
    """Run-level settings. Paths are relative to the repository root by default."""

    data_path: Path = Path("data/raw/creditcard.csv")
    output_dir: Path = Path(".")
    target_column: str = "Class"
    time_column: str = "Time"
    amount_column: str = "Amount"
    random_seed: int = 42
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    false_positive_cost: float = 5.0
    false_negative_cost_override: float | None = None
    unrecovered_fraud_fraction: float = 0.80
    isolation_estimators: int = 200
    concept_region_tail_fraction: float = 0.05
    concept_injection_rate: float = 0.10
    concept_adaptation_fraction: float = 0.80
    save_models: bool = True
    run_retraining_demo: bool = True
    drift: DriftConfig = DriftConfig()

    def __post_init__(self) -> None:
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train_fraction + validation_fraction must be < 1")
        if self.false_positive_cost < 0:
            raise ValueError("false_positive_cost must be non-negative")
        if self.false_negative_cost_override is not None and self.false_negative_cost_override < 0:
            raise ValueError("false_negative_cost_override must be non-negative")
        if not 0 <= self.unrecovered_fraud_fraction <= 1:
            raise ValueError("unrecovered_fraud_fraction must be in [0, 1]")
        if not 0 < self.concept_injection_rate < 1:
            raise ValueError("concept_injection_rate must be in (0, 1)")
        if not 0 < self.concept_adaptation_fraction < 1:
            raise ValueError("concept_adaptation_fraction must be in (0, 1)")

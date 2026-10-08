"""Data loading, auditing, duplicate handling, and leakage-safe time splitting."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DataSplits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_columns: list[str]
    split_metadata: dict[str, Any]


def audit_dataset(frame: pd.DataFrame, target_column: str = "Class") -> dict[str, Any]:
    """Summarize raw schema, quality, class balance, and the main numeric fields."""
    if target_column not in frame.columns:
        raise ValueError(f"Target column {target_column!r} is missing; columns={list(frame.columns)}")
    target = frame[target_column]
    class_counts = target.value_counts(dropna=False).sort_index()
    class_counts_json = {str(k): int(v) for k, v in class_counts.items()}
    total = int(len(frame))
    fraud_count = int((target == 1).sum())
    numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
    amount_summary: dict[str, float] = {}
    if "Amount" in frame.columns and pd.api.types.is_numeric_dtype(frame["Amount"]):
        amount_summary = {
            key: float(value)
            for key, value in frame["Amount"].describe(percentiles=[0.25, 0.5, 0.75, 0.9]).items()
        }
    time_summary: dict[str, float] = {}
    if "Time" in frame.columns and pd.api.types.is_numeric_dtype(frame["Time"]):
        time_summary = {
            "min": float(frame["Time"].min()),
            "max": float(frame["Time"].max()),
            "unique_values": int(frame["Time"].nunique(dropna=True)),
            "is_monotonic_in_input": bool(frame["Time"].is_monotonic_increasing),
        }
    return {
        "rows": total,
        "columns": int(frame.shape[1]),
        "column_names": [str(c) for c in frame.columns],
        "dtypes": {str(c): str(dtype) for c, dtype in frame.dtypes.items()},
        "feature_types": {
            "numeric": [str(c) for c in numeric_columns if c != target_column],
            "non_numeric": [str(c) for c in frame.columns if c not in numeric_columns and c != target_column],
        },
        "missing_cells": int(frame.isna().sum().sum()),
        "missing_by_column": {str(k): int(v) for k, v in frame.isna().sum().items() if int(v) > 0},
        "exact_duplicate_rows": int(frame.duplicated(keep="first").sum()),
        "class_counts": class_counts_json,
        "fraud_count": fraud_count,
        "fraud_fraction": float(fraud_count / total) if total else 0.0,
        "amount_summary": amount_summary,
        "time_summary": time_summary,
    }


def load_transactions(
    csv_path: str | Path,
    target_column: str = "Class",
    deduplicate: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read and validate the CSV. Exact duplicates are removed by default.

    Duplicate detection is label-agnostic: the complete row fingerprint is used.
    The dataset audit records counts before and after cleaning. Feature missingness
    is left for a train-fitted median imputer; missing targets are rejected.
    """
    path = Path(csv_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the Kaggle Credit Card Fraud "
            "Detection CSV and place it at data/raw/creditcard.csv, or pass --data PATH. "
            "See data/README.md for exact instructions."
        )
    frame = pd.read_csv(path, low_memory=False)
    if target_column not in frame.columns:
        raise ValueError(f"Expected target column {target_column!r}; found {list(frame.columns)}")
    if frame[target_column].isna().any():
        raise ValueError("The target column contains missing values; resolve these before training.")
    observed_targets = set(pd.unique(frame[target_column]))
    if not observed_targets.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"Expected a binary 0/1 target; observed values include {observed_targets}")
    non_numeric_features = [
        c for c in frame.columns if c != target_column and not pd.api.types.is_numeric_dtype(frame[c])
    ]
    if non_numeric_features:
        raise ValueError(
            "This project expects numeric transaction features. Non-numeric columns found: "
            f"{non_numeric_features}"
        )

    audit = audit_dataset(frame, target_column)
    audit["source_path"] = str(path)
    audit["duplicate_policy"] = "drop exact duplicate rows before splitting" if deduplicate else "retain all rows"
    if deduplicate:
        frame = frame.drop_duplicates(keep="first")
    frame = frame.reset_index(drop=True)
    audit["rows_after_duplicate_handling"] = int(len(frame))
    audit["duplicates_removed"] = int(audit["exact_duplicate_rows"] if deduplicate else 0)
    audit["class_counts_after_duplicate_handling"] = {
        str(k): int(v) for k, v in frame[target_column].value_counts().sort_index().items()
    }
    audit["fraud_fraction_after_duplicate_handling"] = float(frame[target_column].mean())
    return frame, audit


def split_transactions(
    frame: pd.DataFrame,
    target_column: str = "Class",
    time_column: str = "Time",
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> DataSplits:
    """Create earlier/train, middle/validation, and later/test partitions.

    If a numeric time feature is available, rows are stably sorted by time and
    identical timestamp groups are kept together at split boundaries. Otherwise,
    a deterministic contiguous split is used (the input order must be meaningful).
    No preprocessing is fitted here.
    """
    if target_column not in frame:
        raise ValueError(f"Target column {target_column!r} not found")
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1):
        raise ValueError("Split fractions must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1")
    if len(frame) < 10:
        raise ValueError("At least 10 rows are required for train/validation/test splitting")

    ordered = frame.copy()
    used_time_split = time_column in ordered.columns and pd.api.types.is_numeric_dtype(ordered[time_column])
    if used_time_split:
        if ordered[time_column].isna().any():
            raise ValueError(f"Time column {time_column!r} contains missing values")
        ordered = ordered.sort_values(time_column, kind="mergesort").reset_index(drop=True)
        train_cut_index = min(max(int(len(ordered) * train_fraction), 1), len(ordered) - 2)
        validation_cut_index = min(
            max(int(len(ordered) * (train_fraction + validation_fraction)), train_cut_index + 1),
            len(ordered) - 1,
        )
        train_cut_time = ordered.loc[train_cut_index, time_column]
        validation_cut_time = ordered.loc[validation_cut_index, time_column]
        train = ordered.loc[ordered[time_column] < train_cut_time]
        validation = ordered.loc[
            (ordered[time_column] >= train_cut_time) & (ordered[time_column] < validation_cut_time)
        ]
        test = ordered.loc[ordered[time_column] >= validation_cut_time]
        # Degenerate time values (e.g. a constant time column) cannot define windows.
        if min(len(train), len(validation), len(test)) == 0:
            used_time_split = False
    if not used_time_split:
        ordered = frame.reset_index(drop=True)
        train_end = int(len(ordered) * train_fraction)
        validation_end = int(len(ordered) * (train_fraction + validation_fraction))
        train = ordered.iloc[:train_end]
        validation = ordered.iloc[train_end:validation_end]
        test = ordered.iloc[validation_end:]

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("A split is empty; check the data and split fractions")
    feature_columns = [str(c) for c in frame.columns if c != target_column]

    def unpack(part: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        X = part.loc[:, feature_columns].copy().reset_index(drop=True)
        y = part[target_column].astype(int).copy().reset_index(drop=True)
        return X, y

    X_train, y_train = unpack(train)
    X_validation, y_validation = unpack(validation)
    X_test, y_test = unpack(test)
    metadata: dict[str, Any] = {
        "method": "chronological by Time; ties kept together" if used_time_split else "contiguous input order",
        "requested_fractions": {
            "train": float(train_fraction),
            "validation": float(validation_fraction),
            "test": float(1 - train_fraction - validation_fraction),
        },
        "actual_rows": {"train": len(X_train), "validation": len(X_validation), "test": len(X_test)},
        "fraud_counts": {
            "train": int(y_train.sum()),
            "validation": int(y_validation.sum()),
            "test": int(y_test.sum()),
        },
    }
    if used_time_split:
        metadata["time_ranges"] = {
            "train": [float(train[time_column].min()), float(train[time_column].max())],
            "validation": [float(validation[time_column].min()), float(validation[time_column].max())],
            "test": [float(test[time_column].min()), float(test[time_column].max())],
        }
    return DataSplits(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test,
        feature_columns=feature_columns,
        split_metadata=metadata,
    )

"""Isolation Forest fit on a presumed-normal cohort, without supervised loss."""
from __future__ import annotations

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_isolation_forest(random_seed: int = 42, n_estimators: int = 200) -> Pipeline:
    """Create a preprocessing + Isolation Forest pipeline.

    Scaling is not required by the tree ensemble, but keeps the preprocessing
    contract explicit and makes score diagnostics easier to compare. The scaler
    and imputer are fit only on the training cohort passed to ``fit``.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "isolation_forest",
                IsolationForest(
                    n_estimators=n_estimators,
                    max_samples="auto",
                    contamination="auto",
                    max_features=1.0,
                    bootstrap=False,
                    n_jobs=-1,
                    random_state=random_seed,
                ),
            ),
        ]
    )


def train_isolation_forest(
    X_train,
    y_train,
    random_seed: int = 42,
    n_estimators: int = 200,
) -> Pipeline:
    """Fit only on training rows labeled legitimate.

    Labels are used only to define a clean reference cohort; they are not passed
    to Isolation Forest and do not enter its objective. Validation/test labels
    are not accessed during fitting or score generation.
    """
    normal_mask = (y_train.to_numpy() == 0) if hasattr(y_train, "to_numpy") else (y_train == 0)
    X_normal = X_train.loc[normal_mask] if hasattr(X_train, "loc") else X_train[normal_mask]
    if len(X_normal) < 2:
        raise ValueError("Isolation Forest requires at least two legitimate training transactions")
    model = build_isolation_forest(random_seed=random_seed, n_estimators=n_estimators)
    model.fit(X_normal)
    return model


def anomaly_scores(model, X):
    """Return anomaly scores where larger values mean more anomalous."""
    return -model.decision_function(X)

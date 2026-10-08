"""Supervised model candidates with preprocessing learned on training data only."""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_supervised_models(random_seed: int = 42) -> dict[str, Pipeline]:
    """Return an interpretable weighted baseline and a nonlinear boosted-tree model."""
    logistic = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    solver="lbfgs",
                    C=1.0,
                    max_iter=1000,
                    random_state=random_seed,
                ),
            ),
        ]
    )
    histogram_boosting = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.06,
                    max_iter=160,
                    max_leaf_nodes=15,
                    min_samples_leaf=30,
                    l2_regularization=2.0,
                    class_weight="balanced",
                    early_stopping=False,
                    random_state=random_seed,
                ),
            ),
        ]
    )
    return {"logistic_regression": logistic, "hist_gradient_boosting": histogram_boosting}


def train_supervised_models(X_train, y_train, random_seed: int = 42) -> dict[str, Pipeline]:
    """Fit all candidates on the supplied training partition."""
    models = build_supervised_models(random_seed=random_seed)
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def positive_class_scores(model, X) -> "object":
    """Return P(Class=1); tolerate estimators without predict_proba if needed."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        classes = list(model.classes_)
        return probabilities[:, classes.index(1)]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    raise TypeError("Model must implement predict_proba or decision_function")

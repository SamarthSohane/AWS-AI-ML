"""Publication-quality figures for EDA, model evaluation, thresholds, and drift."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score, confusion_matrix

sns.set_theme(style="whitegrid", context="notebook", palette="deep")


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_eda(frame: pd.DataFrame, target: str, figures_dir: str | Path) -> None:
    out = Path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = frame[target].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(["Legitimate (0)", "Fraud (1)"], counts.values, color=["#2563eb", "#f97316"])
    ax.set_yscale("log")
    ax.set_ylabel("Transactions (log scale)")
    ax.set_title("Target class distribution — extreme imbalance")
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.04, f"{count:,}\n{count / len(frame):.3%}", ha="center", va="bottom")
    _save(fig, out / "class_distribution.png")

    if "Amount" in frame:
        amount = pd.to_numeric(frame["Amount"], errors="coerce").clip(lower=0)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.hist(np.log1p(amount.dropna()), bins=70, color="#2563eb", alpha=0.88)
        ax.set_xlabel("log(1 + transaction amount)")
        ax.set_ylabel("Transactions")
        ax.set_title("Transaction amount distribution (log1p scale)")
        _save(fig, out / "transaction_amount_distribution.png")

        plot_data = frame[[target, "Amount"]].copy()
        plot_data["log_amount"] = np.log1p(pd.to_numeric(plot_data["Amount"], errors="coerce").clip(lower=0))
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.violinplot(data=plot_data, x=target, y="log_amount", hue=target, inner="quartile", cut=0, ax=ax, palette={0: "#2563eb", 1: "#f97316"}, legend=False)
        ax.set_xticks([0, 1], ["Legitimate", "Fraud"])
        ax.set_xlabel("")
        ax.set_ylabel("log(1 + amount)")
        ax.set_title("Amount distribution by transaction class")
        _save(fig, out / "fraud_vs_legitimate_amount.png")

    if "Time" in frame:
        hours = pd.to_numeric(frame["Time"], errors="coerce") / 3600.0
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.hist(hours.dropna(), bins=48, color="#0f766e", alpha=0.9)
        ax.set_xlabel("Elapsed hours from first recorded transaction")
        ax.set_ylabel("Transactions")
        ax.set_title("Transaction-time distribution (dataset covers roughly two days)")
        _save(fig, out / "transaction_time_distribution.png")

    numeric = frame.select_dtypes(include=[np.number]).drop(columns=[target], errors="ignore")
    if numeric.shape[1] <= 40 and numeric.shape[1] >= 2:
        corr = numeric.corr(method="spearman")
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, square=True, linewidths=0.05, ax=ax, cbar_kws={"shrink": 0.75})
        ax.set_title("Spearman correlation of numeric transaction features")
        _save(fig, out / "feature_correlation_heatmap.png")


def plot_model_curves(y_true, scores_by_model: dict[str, np.ndarray], figures_dir: str | Path) -> None:
    out = Path(figures_dir)
    y = np.asarray(y_true, dtype=int)
    prevalence = float(y.mean())
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, scores in scores_by_model.items():
        precision, recall, _ = precision_recall_curve(y, scores)
        ap = average_precision_score(y, scores)
        ax.plot(recall, precision, linewidth=2, label=f"{name} (AP={ap:.3f})")
    ax.axhline(prevalence, color="black", linestyle="--", linewidth=1, label=f"No-skill prevalence={prevalence:.4f}")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision", title="Precision–Recall curves (held-out test window)")
    ax.legend(loc="best", fontsize=8)
    _save(fig, out / "precision_recall_curves.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, scores in scores_by_model.items():
        fpr, tpr, _ = roc_curve(y, scores)
        auc = roc_auc_score(y, scores)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (ROC-AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curves (secondary diagnostic)")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, out / "roc_curves.png")


def plot_confusion_matrices(y_true, model_results: dict[str, dict], figures_dir: str | Path) -> None:
    out = Path(figures_dir)
    names = list(model_results)
    ncols = min(3, len(names))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.0 * nrows), squeeze=False)
    for ax, name in zip(axes.flat, names):
        result = model_results[name]
        cm = np.array([[result["true_negatives"], result["false_positives"]], [result["false_negatives"], result["true_positives"]]])
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax, xticklabels=["Predicted legit", "Predicted fraud"], yticklabels=["Actual legit", "Actual fraud"])
        ax.set_title(f"{name}\nthreshold={result['threshold']:.5g}")
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    fig.suptitle("Held-out test confusion matrices at validation-selected thresholds", y=1.02)
    _save(fig, out / "confusion_matrices.png")


def plot_threshold_analysis(threshold_table: pd.DataFrame, model_name: str, figures_dir: str | Path) -> None:
    out = Path(figures_dir)
    view = threshold_table.loc[threshold_table["model"] == model_name].sort_values("threshold")
    if view.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(view["threshold"], view["precision"], marker="o", label="Precision")
    ax.plot(view["threshold"], view["recall"], marker="o", label="Recall")
    if "selected_cost_threshold" in view and view["selected_cost_threshold"].any():
        selected = float(view.loc[view["selected_cost_threshold"], "threshold"].iloc[0])
        ax.axvline(selected, color="black", linestyle="--", label=f"Cost-selected={selected:.4g}")
    ax.set(xlabel="Decision threshold", ylabel="Metric", ylim=(0, 1.02), title=f"Validation precision/recall vs threshold — {model_name}")
    ax.legend()
    _save(fig, out / "threshold_precision_recall.png")

    if "estimated_total_cost" in view:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(view["threshold"], view["estimated_total_cost"], color="#b91c1c", marker="o")
        if "selected_cost_threshold" in view and view["selected_cost_threshold"].any():
            selected = float(view.loc[view["selected_cost_threshold"], "threshold"].iloc[0])
            ax.axvline(selected, color="black", linestyle="--", label=f"Cost-selected={selected:.4g}")
            ax.legend()
        ax.set(xlabel="Decision threshold", ylabel="Estimated validation cost (USD-equivalent)", title="Validation business cost vs threshold")
        _save(fig, out / "threshold_expected_cost.png")


def plot_feature_drift(drift_report: pd.DataFrame, reference: pd.DataFrame, current: pd.DataFrame, figures_dir: str | Path, top_n: int = 4) -> None:
    out = Path(figures_dir)
    if drift_report.empty:
        return
    view = drift_report.sort_values("psi", ascending=False).head(top_n).sort_values("psi")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(view))))
    colors = ["#dc2626" if p >= 0.25 else ("#f59e0b" if p >= 0.1 else "#2563eb") for p in view["psi"]]
    ax.barh(view["feature"], view["psi"], color=colors)
    ax.axvline(0.10, color="#f59e0b", linestyle="--", label="PSI 0.10 heuristic")
    ax.axvline(0.25, color="#dc2626", linestyle="--", label="PSI 0.25 heuristic")
    ax.set_xlabel("Population Stability Index")
    ax.set_title("Largest feature-distribution shifts")
    ax.legend(fontsize=8)
    _save(fig, out / "feature_psi.png")

    features = list(view["feature"])
    fig, axes = plt.subplots(len(features), 1, figsize=(8, 3.0 * len(features)), squeeze=False)
    for ax, feature in zip(axes.flat, features):
        ref = pd.to_numeric(reference[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        cur = pd.to_numeric(current[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        lower, upper = np.quantile(ref, [0.005, 0.995]) if len(ref) > 10 else (ref.min(), ref.max())
        ref = ref.clip(lower, upper)
        cur = cur.clip(lower, upper)
        ax.hist(ref, bins=40, density=True, alpha=0.55, label="Reference/train", color="#2563eb")
        ax.hist(cur, bins=40, density=True, alpha=0.50, label="Current/later", color="#f97316")
        psi = float(view.loc[view["feature"] == feature, "psi"].iloc[0])
        ax.set_title(f"{feature}: PSI={psi:.3f}")
        ax.legend(fontsize=8)
    fig.suptitle("Reference vs current feature distributions (tails clipped for display only)", y=1.01)
    _save(fig, out / "feature_drift_distributions.png")


def plot_model_score_distributions(score_arrays: dict[str, tuple[np.ndarray, np.ndarray]], figures_dir: str | Path) -> None:
    out = Path(figures_dir)
    n = len(score_arrays)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3.8 * n), squeeze=False)
    for ax, (name, (reference, current)) in zip(axes.flat, score_arrays.items()):
        ref = np.asarray(reference, dtype=float)
        cur = np.asarray(current, dtype=float)
        low = float(np.nanquantile(np.r_[ref, cur], 0.001))
        high = float(np.nanquantile(np.r_[ref, cur], 0.999))
        ax.hist(ref, bins=50, range=(low, high), density=True, alpha=0.55, label="Reference window (validation)", color="#2563eb")
        ax.hist(cur, bins=50, range=(low, high), density=True, alpha=0.50, label="Current window (test)", color="#f97316")
        ax.set_title(name)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    axes[-1, 0].set_xlabel("Model score")
    fig.suptitle("Model-output distribution monitoring", y=1.01)
    _save(fig, out / "model_score_distributions.png")


def plot_performance_comparison(table: pd.DataFrame, path: str | Path, title: str, categories: list[str]) -> None:
    metrics = [m for m in ["pr_auc", "precision", "recall", "f1"] if m in table.columns]
    if not metrics or table.empty:
        return
    melted = table.melt(id_vars=categories, value_vars=metrics, var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=melted, x="metric", y="value", hue=categories[0], ax=ax)
    ax.set(ylim=(0, 1.02), ylabel="Score", xlabel="", title=title)
    _save(fig, Path(path))

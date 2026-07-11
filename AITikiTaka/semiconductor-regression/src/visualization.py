"""
Visualization utilities for the semiconductor process regression project.
All plots are saved as PNG files to the configured plots directory.
"""

import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(save_dir: str) -> None:
    """Create save_dir if it does not exist."""
    os.makedirs(save_dir, exist_ok=True)


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: List[str],
    model_name: str,
    save_dir: str,
) -> None:
    """
    Plot actual vs. predicted values for each target column and save as PNG.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth target values, shape (n_samples, n_targets).
    y_pred : np.ndarray
        Predicted target values, shape (n_samples, n_targets).
    target_cols : list of str
        Target column names.
    model_name : str
        Model name used in the plot title and filename.
    save_dir : str
        Directory where the PNG file will be saved.
    """
    _ensure_dir(save_dir)
    n_targets = len(target_cols)
    fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 5))
    if n_targets == 1:
        axes = [axes]

    for i, (col, ax) in enumerate(zip(target_cols, axes)):
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]

        min_val = min(col_true.min(), col_pred.min())
        max_val = max(col_true.max(), col_pred.max())

        ax.scatter(col_true, col_pred, alpha=0.6, edgecolors="k", linewidths=0.3)
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect fit")
        ax.set_xlabel(f"Actual {col}", fontsize=12)
        ax.set_ylabel(f"Predicted {col}", fontsize=12)
        ax.set_title(f"{model_name}\nActual vs Predicted — {col}", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    safe_name = model_name.replace(" ", "_")
    filename = os.path.join(save_dir, f"{safe_name}_actual_vs_predicted.png")
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualization] Saved: {filename}")


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: List[str],
    model_name: str,
    save_dir: str,
) -> None:
    """
    Plot residuals (actual − predicted) vs. predicted values for each target column.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth target values, shape (n_samples, n_targets).
    y_pred : np.ndarray
        Predicted target values, shape (n_samples, n_targets).
    target_cols : list of str
        Target column names.
    model_name : str
        Model name used in the plot title and filename.
    save_dir : str
        Directory where the PNG file will be saved.
    """
    _ensure_dir(save_dir)
    n_targets = len(target_cols)
    fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 5))
    if n_targets == 1:
        axes = [axes]

    for i, (col, ax) in enumerate(zip(target_cols, axes)):
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]
        residuals = col_true - col_pred

        ax.scatter(col_pred, residuals, alpha=0.6, edgecolors="k", linewidths=0.3)
        ax.axhline(0, color="r", linestyle="--", linewidth=1.5)
        ax.set_xlabel(f"Predicted {col}", fontsize=12)
        ax.set_ylabel("Residual (Actual − Predicted)", fontsize=12)
        ax.set_title(f"{model_name}\nResiduals — {col}", fontsize=13)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    safe_name = model_name.replace(" ", "_")
    filename = os.path.join(save_dir, f"{safe_name}_residuals.png")
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualization] Saved: {filename}")


def plot_feature_importance(
    model,
    feature_cols: List[str],
    model_name: str,
    save_dir: str,
) -> None:
    """
    Plot feature importances for tree-based models (RandomForest, XGBoost).

    For MultiOutputRegressor wrappers the importances are averaged across all
    contained estimators. Silently skips if the model does not expose
    ``feature_importances_`` (e.g. LinearRegression).

    Parameters
    ----------
    model : estimator
        Trained scikit-learn model or pipeline.
    feature_cols : list of str
        Feature column names.
    model_name : str
        Model name used in the plot title and filename.
    save_dir : str
        Directory where the PNG file will be saved.
    """
    _ensure_dir(save_dir)

    importances: Optional[np.ndarray] = None

    if hasattr(model, "feature_importances_"):
        # Natively multi-output (e.g. RandomForest)
        importances = model.feature_importances_
    elif hasattr(model, "estimators_"):
        # MultiOutputRegressor: average across per-output estimators
        sub_importances = []
        for est in model.estimators_:
            inner = est
            # Unwrap Pipeline if necessary
            if hasattr(inner, "named_steps"):
                for step_name, step in inner.named_steps.items():
                    if hasattr(step, "feature_importances_"):
                        inner = step
                        break
            if hasattr(inner, "feature_importances_"):
                sub_importances.append(inner.feature_importances_)
        if sub_importances:
            importances = np.mean(sub_importances, axis=0)

    if importances is None:
        print(
            f"[visualization] '{model_name}' does not expose feature importances; "
            "skipping importance plot."
        )
        return

    sorted_idx = np.argsort(importances)
    sorted_features = [feature_cols[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(sorted_features, sorted_importances, color="steelblue", edgecolor="k", linewidth=0.4)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"{model_name} — Feature Importances", fontsize=13)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    # Label bars
    for bar, val in zip(bars, sorted_importances):
        ax.text(
            bar.get_width() + max(sorted_importances) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    safe_name = model_name.replace(" ", "_")
    filename = os.path.join(save_dir, f"{safe_name}_feature_importance.png")
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualization] Saved: {filename}")


def plot_correlation_heatmap(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    save_dir: str,
) -> None:
    """
    Plot a heatmap of the full correlation matrix (features + targets).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing both feature and target columns.
    feature_cols : list of str
        Feature column names.
    target_cols : list of str
        Target column names.
    save_dir : str
        Directory where the PNG file will be saved.
    """
    _ensure_dir(save_dir)

    all_cols = feature_cols + target_cols
    corr = df[all_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax)

    labels = list(corr.columns)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="left", fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j, i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center",
                fontsize=9,
                color="black" if abs(corr.iloc[i, j]) < 0.7 else "white",
            )

    ax.set_title("Correlation Heatmap", fontsize=14, pad=20)
    plt.tight_layout()

    filename = os.path.join(save_dir, "correlation_heatmap.png")
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualization] Saved: {filename}")

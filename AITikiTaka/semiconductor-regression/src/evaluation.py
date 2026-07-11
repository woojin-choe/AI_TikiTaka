"""
Evaluation utilities for the semiconductor process regression project.
Computes per-target and averaged MAE, RMSE, and R² metrics.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: list,
    model_name: str,
) -> Dict[str, float]:
    """
    Compute and print MAE, RMSE, and R² per target column plus their averages.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth target values, shape (n_samples, n_targets).
    y_pred : np.ndarray
        Predicted target values, shape (n_samples, n_targets).
    target_cols : list
        Names of target columns (used for labelling output).
    model_name : str
        Human-readable model name (used for labelling output).

    Returns
    -------
    dict
        Flat dictionary with keys like "Thickness_MAE", "Thickness_RMSE",
        "Thickness_R2", "Uniformity_MAE", ..., "avg_MAE", "avg_RMSE", "avg_R2".
    """
    results: Dict[str, float] = {}
    mae_list, rmse_list, r2_list = [], [], []

    print(f"\n{'=' * 55}")
    print(f"  Model: {model_name}")
    print(f"{'=' * 55}")
    print(f"  {'Target':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print(f"  {'-' * 46}")

    for i, col in enumerate(target_cols):
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]

        mae = mean_absolute_error(col_true, col_pred)
        rmse = float(np.sqrt(mean_squared_error(col_true, col_pred)))
        r2 = r2_score(col_true, col_pred)

        results[f"{col}_MAE"] = mae
        results[f"{col}_RMSE"] = rmse
        results[f"{col}_R2"] = r2

        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)

        print(f"  {col:<20} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f}")

    avg_mae = float(np.mean(mae_list))
    avg_rmse = float(np.mean(rmse_list))
    avg_r2 = float(np.mean(r2_list))

    results["avg_MAE"] = avg_mae
    results["avg_RMSE"] = avg_rmse
    results["avg_R2"] = avg_r2

    print(f"  {'-' * 46}")
    print(f"  {'Average':<20} {avg_mae:>8.4f} {avg_rmse:>8.4f} {avg_r2:>8.4f}")
    print(f"{'=' * 55}\n")

    return results


def select_best_model(results: dict) -> str:
    """
    Select the best model based on the lowest average validation MAE.

    Parameters
    ----------
    results : dict
        Nested dictionary keyed by model name, where each value is the metrics
        dict returned by ``evaluate_model``. Expected key: "avg_MAE".

    Returns
    -------
    str
        Name of the model with the lowest average MAE.

    Raises
    ------
    ValueError
        If results dict is empty or "avg_MAE" is missing for any model.
    """
    if not results:
        raise ValueError("results dict is empty; cannot select best model.")

    best_name = None
    best_mae = float("inf")

    for model_name, metrics in results.items():
        if "avg_MAE" not in metrics:
            raise ValueError(
                f"'avg_MAE' key missing from metrics for model '{model_name}'."
            )
        if metrics["avg_MAE"] < best_mae:
            best_mae = metrics["avg_MAE"]
            best_name = model_name

    print(f"\n[evaluation] Best model selected: '{best_name}' "
          f"(avg validation MAE = {best_mae:.4f})")
    return best_name

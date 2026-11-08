"""
train.py — End-to-end training pipeline for semiconductor process regression.

Usage:
    python train.py --data data/process_data.csv
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

# Allow running from the project root without installing the package
sys.path.insert(0, os.path.dirname(__file__))

from src.config import (
    ARTIFACTS_DIR,
    FEATURE_COLUMNS,
    PLOTS_DIR,
    RANDOM_STATE,
    TARGET_COLUMNS,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.data_loader import load_data
from src.evaluation import evaluate_model, select_best_model
from src.models import get_models
from src.preprocessing import preprocess
from src.visualization import (
    plot_actual_vs_predicted,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_residuals,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# 그룹 기반 분할에 사용할 컬럼 후보 (우선순위 순서)
GROUP_COLUMNS = ["Lot_ID", "Wafer_ID", "Run_ID", "Recipe_ID", "Timestamp"]


def _split_dataframe(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> tuple:
    """
    Split a dataframe into train / validation / test subsets.

    Lot_ID / Wafer_ID 등 그룹 컬럼이 존재하면 그룹 기반 분할을 사용합니다.
    같은 Lot의 데이터가 train/val/test에 동시에 들어가지 않습니다.
    그룹 컬럼이 없으면 랜덤 분할로 폴백합니다.

    Parameters
    ----------
    df : pd.DataFrame
    train_ratio, val_ratio, test_ratio : float
        Must sum to 1.0.
    random_state : int

    Returns
    -------
    tuple of (df_train, df_val, df_test)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1."

    # 그룹 컬럼 탐지
    group_col = next(
        (col for col in GROUP_COLUMNS if col in df.columns), None
    )

    if group_col is not None:
        print(f"[train] Group column detected: '{group_col}' → using group-based split.")
        groups = df[group_col].unique()
        rng = np.random.default_rng(random_state)
        groups = rng.permutation(groups)

        n = len(groups)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        train_groups = set(groups[:n_train])
        val_groups   = set(groups[n_train: n_train + n_val])
        test_groups  = set(groups[n_train + n_val:])

        df_train = df[df[group_col].isin(train_groups)].copy()
        df_val   = df[df[group_col].isin(val_groups)].copy()
        df_test  = df[df[group_col].isin(test_groups)].copy()

        print(
            f"[train] Groups — train: {len(train_groups)}, "
            f"val: {len(val_groups)}, test: {len(test_groups)}"
        )
    else:
        print("[train] No group column found → using random split.")
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        n = len(df)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        df_train = df.iloc[:n_train]
        df_val   = df.iloc[n_train: n_train + n_val]
        df_test  = df.iloc[n_train + n_val:]

    return df_train, df_val, df_test


def _save_json(obj: dict, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[train] Saved: {filepath}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_training(data_path: str) -> None:
    """
    Execute the full training pipeline:
      1. Load data
      2. Preprocess (train set only for scaler fitting)
      3. Split 70 / 15 / 15
      4. Train all 3 models
      5. Evaluate on validation set → select best model
      6. Evaluate best model on test set
      7. Save artifacts
      8. Generate all plots

    Parameters
    ----------
    data_path : str
        Path to the raw CSV data file.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("  STEP 1: Loading data")
    print("=" * 60)
    df_raw = load_data(data_path)

    # ------------------------------------------------------------------ #
    # 2. Light cleaning before split (keep original df for heatmap)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("  STEP 2: Splitting data (70 / 15 / 15)")
    print("=" * 60)

    # Drop completely blank rows to avoid contamination
    group_columns = [c for c in GROUP_COLUMNS if c in df_raw.columns]
    df_clean = df_raw[FEATURE_COLUMNS + TARGET_COLUMNS + group_columns].copy()
    df_train_raw, df_val_raw, df_test_raw = _split_dataframe(
        df_clean, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_STATE
    )
    print(
        f"[train] Split sizes — train: {len(df_train_raw)}, "
        f"val: {len(df_val_raw)}, test: {len(df_test_raw)}"
    )

    # ------------------------------------------------------------------ #
    # 3. Preprocess — fit scaler on TRAIN only
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("  STEP 3: Preprocessing (train set)")
    print("=" * 60)
    X_train, y_train, scaler = preprocess(
        df_train_raw, FEATURE_COLUMNS, TARGET_COLUMNS, fit=True
    )

    print("\n" + "=" * 60)
    print("  STEP 3b: Preprocessing (validation set)")
    print("=" * 60)
    X_val, y_val, _ = preprocess(
        df_val_raw, FEATURE_COLUMNS, TARGET_COLUMNS, fit=False, scaler=scaler
    )

    print("\n" + "=" * 60)
    print("  STEP 3c: Preprocessing (test set)")
    print("=" * 60)
    X_test, y_test, _ = preprocess(
        df_test_raw, FEATURE_COLUMNS, TARGET_COLUMNS, fit=False, scaler=scaler
    )

    # ------------------------------------------------------------------ #
    # 4. Train all models
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("  STEP 4: Training models")
    print("=" * 60)
    models = get_models()
    trained_models: dict = {}
    for name, model in models.items():
        print(f"[train] Training '{name}' ...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"[train] '{name}' training complete.")

    # ------------------------------------------------------------------ #
    # 5. Evaluate on validation set
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("  STEP 5: Validation set evaluation")
    print("=" * 60)
    val_results: dict = {}
    for name, model in trained_models.items():
        y_val_pred = model.predict(X_val)
        val_results[name] = evaluate_model(y_val, y_val_pred, TARGET_COLUMNS, name)

    # ------------------------------------------------------------------ #
    # 6. Select best model
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("  STEP 6: Model selection")
    print("=" * 60)
    best_name = select_best_model(val_results)
    best_model = trained_models[best_name]

    model_selection_info = {
        "best_model": best_name,
        "selection_criterion": "lowest average validation MAE",
        "validation_avg_MAE": {
            name: metrics["avg_MAE"] for name, metrics in val_results.items()
        },
        "rationale": (
            f"'{best_name}' was chosen because it achieved the lowest average MAE "
            f"({val_results[best_name]['avg_MAE']:.4f}) across all target columns "
            f"on the held-out validation set, without any exposure to test data."
        ),
    }

    # ------------------------------------------------------------------ #
    # 7. Evaluate best model on test set
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("  STEP 7: Test set evaluation (best model only)")
    print("=" * 60)
    y_test_pred = best_model.predict(X_test)
    test_metrics = evaluate_model(y_test, y_test_pred, TARGET_COLUMNS, f"{best_name} [TEST]")

    # ------------------------------------------------------------------ #
    # 8. Save artifacts
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("  STEP 8: Saving artifacts")
    print("=" * 60)

    # Model
    model_path = os.path.join(ARTIFACTS_DIR, "model.joblib")
    joblib.dump({"model": best_model, "scaler": scaler}, model_path)
    print(f"[train] Saved: {model_path}")

    # Feature / target column lists
    _save_json(FEATURE_COLUMNS, os.path.join(ARTIFACTS_DIR, "feature_columns.json"))
    _save_json(TARGET_COLUMNS, os.path.join(ARTIFACTS_DIR, "target_columns.json"))

    # Full metrics: validation for all models + test for best
    all_metrics = {
        "validation": {name: {k: round(v, 6) for k, v in m.items()}
                       for name, m in val_results.items()},
        "test_best_model": {k: round(v, 6) for k, v in test_metrics.items()},
    }
    _save_json(all_metrics, os.path.join(ARTIFACTS_DIR, "metrics.json"))

    # Model selection rationale
    _save_json(model_selection_info, os.path.join(ARTIFACTS_DIR, "model_selection.json"))

    # ------------------------------------------------------------------ #
    # 9. Plots
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("  STEP 9: Generating plots")
    print("=" * 60)

    # Correlation heatmap (use full cleaned dataset)
    df_for_heatmap = df_clean.dropna()
    plot_correlation_heatmap(df_for_heatmap, FEATURE_COLUMNS, TARGET_COLUMNS, PLOTS_DIR)

    # Plots for every model on the validation set
    for name, model in trained_models.items():
        y_pred = model.predict(X_val)
        plot_actual_vs_predicted(y_val, y_pred, TARGET_COLUMNS, name, PLOTS_DIR)
        plot_residuals(y_val, y_pred, TARGET_COLUMNS, name, PLOTS_DIR)
        plot_feature_importance(model, FEATURE_COLUMNS, name, PLOTS_DIR)

    # Extra: best model on test set
    plot_actual_vs_predicted(
        y_test, y_test_pred, TARGET_COLUMNS, f"{best_name}_TEST", PLOTS_DIR
    )
    plot_residuals(
        y_test, y_test_pred, TARGET_COLUMNS, f"{best_name}_TEST", PLOTS_DIR
    )

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best model : {best_name}")
    print(f"  Test avg MAE  : {test_metrics['avg_MAE']:.4f}")
    print(f"  Test avg RMSE : {test_metrics['avg_RMSE']:.4f}")
    print(f"  Test avg R²   : {test_metrics['avg_R2']:.4f}")
    print(f"\n  Artifacts saved to: {ARTIFACTS_DIR}/")
    print(f"  Plots    saved to:  {PLOTS_DIR}/")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train semiconductor process regression models."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/process_data.csv",
        help="Path to the process data CSV file (default: data/process_data.csv).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args.data)

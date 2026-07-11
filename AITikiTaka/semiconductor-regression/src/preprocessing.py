"""
Preprocessing utilities for the semiconductor process regression project.
Handles missing values, duplicates, scaling, and exploratory statistics.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import MISSING_THRESHOLD


def preprocess(
    df: pd.DataFrame,
    feature_cols: list,
    target_cols: list,
    fit: bool = True,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Full preprocessing pipeline: validates columns, handles non-numeric values,
    missing data, duplicates, prints descriptive stats and correlation matrix,
    then scales features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe.
    feature_cols : list
        Names of input feature columns.
    target_cols : list
        Names of target output columns.
    fit : bool
        If True, fit a new StandardScaler on the data (use for training set).
        If False, transform using the provided scaler (use for val/test sets).
    scaler : StandardScaler or None
        Pre-fitted scaler to use when fit=False. Ignored when fit=True.

    Returns
    -------
    X : np.ndarray
        Scaled feature matrix.
    y : np.ndarray
        Target matrix (unscaled).
    scaler : StandardScaler
        The fitted scaler (same object returned whether newly fitted or passed in).

    Raises
    ------
    ValueError
        If required columns are missing or if fit=False but no scaler is provided.
    """
    all_required = feature_cols + target_cols

    # --- Validate required columns ---
    missing_cols = [c for c in all_required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataframe: {missing_cols}")

    df = df[all_required].copy()

    # --- Coerce non-numeric values ---
    for col in all_required:
        original_dtype = df[col].dtype
        if original_dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            n_coerced = df[col].isna().sum()
            print(f"[preprocessing] Column '{col}' contained non-numeric values; "
                  f"{n_coerced} value(s) coerced to NaN.")

    # --- Print missing value counts ---
    print("\n[preprocessing] Missing value counts per column:")
    for col in all_required:
        n_missing = df[col].isna().sum()
        ratio = n_missing / len(df)
        print(f"  {col:20s}: {n_missing:4d} missing ({ratio:.1%})")

    # --- Handle missing values: drop or median impute per column ---
    for col in all_required:
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue
        ratio = n_missing / len(df)
        if ratio > MISSING_THRESHOLD:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(
                f"[preprocessing] Column '{col}' missing ratio {ratio:.1%} > "
                f"{MISSING_THRESHOLD:.0%} threshold → median imputed with {median_val:.4f}."
            )
        else:
            # Drop rows with missing values in this column
            before = len(df)
            df.dropna(subset=[col], inplace=True)
            after = len(df)
            print(
                f"[preprocessing] Column '{col}' missing ratio {ratio:.1%} <= "
                f"{MISSING_THRESHOLD:.0%} threshold → dropped {before - after} rows."
            )

    # --- Remove duplicates ---
    before_dedup = len(df)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    n_removed = before_dedup - len(df)
    if n_removed > 0:
        print(f"\n[preprocessing] Removed {n_removed} duplicate row(s).")
    else:
        print("\n[preprocessing] No duplicate rows found.")

    print(f"[preprocessing] Dataset size after cleaning: {len(df)} rows.\n")

    # --- Descriptive statistics ---
    print("[preprocessing] Descriptive statistics:")
    print(df[all_required].describe().to_string())
    print()

    # --- Correlation matrix (features vs targets) ---
    print("[preprocessing] Correlation matrix (features → targets):")
    corr = df[all_required].corr()
    cross_corr = corr.loc[feature_cols, target_cols]
    print(cross_corr.to_string())
    print()

    # --- Split X and y ---
    X_raw = df[feature_cols].values
    y = df[target_cols].values

    # --- Scale features ---
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
        print("[preprocessing] StandardScaler fitted on training data.")
    else:
        if scaler is None:
            raise ValueError(
                "fit=False requires a pre-fitted scaler to be passed via the 'scaler' argument."
            )
        X = scaler.transform(X_raw)
        print("[preprocessing] Applied pre-fitted StandardScaler.")

    return X, y, scaler

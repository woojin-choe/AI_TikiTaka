"""
Data loading utilities for the semiconductor process regression project.
"""

import pandas as pd
from src.config import FEATURE_COLUMNS, TARGET_COLUMNS


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load process data from a CSV file and validate required columns.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing process data.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe with all required columns present.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If required feature or target columns are missing from the CSV.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Data file not found at '{filepath}'. "
            "Please provide the correct path to the CSV file."
        )
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV file '{filepath}': {e}") from e

    required_columns = FEATURE_COLUMNS + TARGET_COLUMNS
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The following required columns are missing from '{filepath}': "
            f"{missing_cols}\n"
            f"Found columns: {list(df.columns)}"
        )

    print(f"[data_loader] Loaded {len(df)} rows and {len(df.columns)} columns from '{filepath}'.")
    return df

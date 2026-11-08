"""
Model definitions for the semiconductor process multi-output regression project.

Three models are provided:
  1. Linear Regression (with StandardScaler pipeline, wrapped in MultiOutputRegressor)
  2. Random Forest Regressor (natively multi-output)
  3. XGBoost Regressor (wrapped in MultiOutputRegressor)
"""

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.config import RANDOM_STATE


def get_models() -> dict:
    """
    Return a dictionary of named, untrained regression models.

    The feature data passed to these models is already scaled by the preprocessing
    pipeline, but the LinearRegression model also carries an internal StandardScaler
    for completeness / portability.

    Returns
    -------
    dict
        Keys are model name strings; values are scikit-learn estimator objects.

        - "LinearRegression": MultiOutputRegressor wrapping a Pipeline of
          StandardScaler + LinearRegression.
        - "RandomForest": RandomForestRegressor (natively multi-output).
        - "XGBoost": MultiOutputRegressor wrapping an XGBRegressor.
    """
    linear_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )

    models = {
        "LinearRegression": MultiOutputRegressor(linear_pipeline),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=2,
        ),
        "XGBoost": MultiOutputRegressor(
            XGBRegressor(
                random_state=RANDOM_STATE,
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                verbosity=0,
                n_jobs=2,
            )
        ),
    }
    return models

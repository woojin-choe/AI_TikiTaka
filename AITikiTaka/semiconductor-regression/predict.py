"""
predict.py — Run inference with the trained semiconductor process regression model.

Usage:
    python predict.py --rf-power 300 --pressure 10.2 --temperature 250 --gas-flow-rate 50
"""

import argparse
import json
import math
import os
import sys
from typing import Dict

import joblib
import numpy as np

# Allow running from the project root without installing the package
sys.path.insert(0, os.path.dirname(__file__))

from src.config import ARTIFACTS_DIR


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------

def predict_process_result(
    rf_power: float,
    pressure: float,
    temperature: float,
    gas_flow_rate: float,
) -> Dict[str, float]:
    """
    Predict Thickness and Uniformity for a given set of process parameters.

    Loads the trained model and scaler from artifacts/model.joblib and applies
    the same preprocessing (scaling) used during training before inference.

    Parameters
    ----------
    rf_power : float
        RF Power in watts (W).  Typical range: 250–400.
    pressure : float
        Chamber pressure in milli-Torr (mTorr).  Typical range: 8–15.
    temperature : float
        Process temperature in Celsius (°C).  Typical range: 200–300.
    gas_flow_rate : float
        Gas flow rate in standard cubic centimetres per minute (sccm).
        Typical range: 30–80.

    Returns
    -------
    dict
        {"Thickness": <float>, "Uniformity": <float>}

    Raises
    ------
    FileNotFoundError
        If the model artifact is not found.
    ValueError
        If any input is NaN or non-finite.
    """
    # --- Validate inputs ---
    inputs = {
        "rf_power": rf_power,
        "pressure": pressure,
        "temperature": temperature,
        "gas_flow_rate": gas_flow_rate,
    }
    for name, val in inputs.items():
        if not isinstance(val, (int, float)):
            raise ValueError(
                f"Input '{name}' must be a numeric value; got {type(val).__name__}."
            )
        if math.isnan(val) or math.isinf(val):
            raise ValueError(
                f"Input '{name}' is NaN or infinite ({val}). "
                "Please provide a valid numeric value."
            )

    # --- Load artifacts ---
    model_path = os.path.join(ARTIFACTS_DIR, "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found at '{model_path}'. "
            "Please run train.py first to generate the model."
        )

    artifact = joblib.load(model_path)
    model = artifact["model"]
    scaler = artifact["scaler"]

    target_path = os.path.join(ARTIFACTS_DIR, "target_columns.json")
    with open(target_path) as f:
        target_cols = json.load(f)

    # --- Build input array and scale ---
    X_raw = np.array([[rf_power, pressure, temperature, gas_flow_rate]], dtype=float)
    X_scaled = scaler.transform(X_raw)

    # --- Predict ---
    y_pred = model.predict(X_scaled)  # shape (1, n_targets)

    result = {col: float(y_pred[0, i]) for i, col in enumerate(target_cols)}
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict semiconductor process outputs (Thickness, Uniformity) "
            "from process parameters using the trained model."
        )
    )
    parser.add_argument(
        "--rf-power",
        type=float,
        required=True,
        help="RF Power in watts (W). Typical range: 250–400.",
    )
    parser.add_argument(
        "--pressure",
        type=float,
        required=True,
        help="Chamber pressure in mTorr. Typical range: 8–15.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=True,
        help="Process temperature in °C. Typical range: 200–300.",
    )
    parser.add_argument(
        "--gas-flow-rate",
        type=float,
        required=True,
        help="Gas flow rate in sccm. Typical range: 30–80.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n[predict] Input parameters:")
    print(f"  RF_Power      : {args.rf_power} W")
    print(f"  Pressure      : {args.pressure} mTorr")
    print(f"  Temperature   : {args.temperature} °C")
    print(f"  Gas_Flow_Rate : {args.gas_flow_rate} sccm\n")

    try:
        result = predict_process_result(
            rf_power=args.rf_power,
            pressure=args.pressure,
            temperature=args.temperature,
            gas_flow_rate=args.gas_flow_rate,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[predict] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    for target, value in result.items():
        print(f"Predicted {target}: {value:.1f}")


if __name__ == "__main__":
    main()

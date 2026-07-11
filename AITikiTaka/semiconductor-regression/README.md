# Semiconductor Process Multi-Output Regression

A complete machine-learning pipeline that predicts two key semiconductor process outputs — **Thickness (nm)** and **Uniformity (%)** — from four process parameters using three regression models.

---

## Project Structure

```
semiconductor-regression/
├── data/
│   ├── process_data.csv          # Demo dataset (200 rows, ~5% missing, a few duplicates)
│   └── generate_demo_data.py     # Script to regenerate demo data
├── artifacts/                    # Created at runtime by train.py
│   ├── model.joblib              # Best trained model + fitted scaler
│   ├── feature_columns.json
│   ├── target_columns.json
│   ├── metrics.json              # Val & test metrics for all models
│   └── model_selection.json      # Selection rationale
├── outputs/
│   └── plots/                    # PNG plots created at runtime by train.py
├── src/
│   ├── __init__.py
│   ├── config.py                 # Project-wide constants
│   ├── data_loader.py            # CSV loading + column validation
│   ├── preprocessing.py          # Cleaning, imputation, scaling
│   ├── models.py                 # Three model definitions
│   ├── evaluation.py             # MAE / RMSE / R² metrics + model selection
│   └── visualization.py          # All plot functions
├── train.py                      # End-to-end training CLI
├── predict.py                    # Single-sample inference CLI
├── requirements.txt
└── README.md
```

---

## Inputs and Outputs

| Role    | Column         | Unit   | Typical Range |
|---------|----------------|--------|---------------|
| Input   | RF_Power       | W      | 250 – 400     |
| Input   | Pressure       | mTorr  | 8 – 15        |
| Input   | Temperature    | °C     | 200 – 300     |
| Input   | Gas_Flow_Rate  | sccm   | 30 – 80       |
| Output  | Thickness      | nm     | 90 – 120      |
| Output  | Uniformity     | %      | 94 – 99       |

---

## Models

| # | Model             | Notes                                              |
|---|-------------------|----------------------------------------------------|
| 1 | LinearRegression  | StandardScaler pipeline, wrapped in MultiOutputRegressor |
| 2 | RandomForest      | n_estimators=200, natively multi-output            |
| 3 | XGBoost           | n_estimators=300, lr=0.05, wrapped in MultiOutputRegressor |

The best model is selected by **lowest average MAE on the validation set**.

---

## Installation

```bash
# 1. Clone / navigate to the project root
cd semiconductor-regression

# 2. (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Generate demo data (if not already present)

```bash
python data/generate_demo_data.py
```

### Train

```bash
python train.py --data data/process_data.csv
```

This will:
1. Load and preprocess data (70/15/15 train/val/test split — no data leakage).
2. Train all three models on the training set.
3. Select the best model by validation MAE.
4. Evaluate the best model on the untouched test set.
5. Save all artifacts to `artifacts/` and plots to `outputs/plots/`.

### Predict

```bash
python predict.py \
    --rf-power 300 \
    --pressure 10.2 \
    --temperature 250 \
    --gas-flow-rate 50
```

Example output:
```
Predicted Thickness: 102.4
Predicted Uniformity: 97.8
```

### Programmatic API

```python
from predict import predict_process_result

result = predict_process_result(
    rf_power=300,
    pressure=10.2,
    temperature=250,
    gas_flow_rate=50,
)
print(result)  # {"Thickness": 102.4, "Uniformity": 97.8}
```

---

## Design Decisions

- **No data leakage**: `StandardScaler` is fitted exclusively on the training set and applied to validation and test sets.
- **Test set isolation**: Test data is never used for model selection; only the best-validated model is evaluated on it.
- **Reproducibility**: All random operations use `RANDOM_STATE = 42`.
- **Robust preprocessing**: Columns with > 30% missing values are median-imputed; others trigger row-dropping.
- **Artifact packaging**: The model and scaler are saved together in a single `model.joblib` so `predict.py` is always consistent with training.

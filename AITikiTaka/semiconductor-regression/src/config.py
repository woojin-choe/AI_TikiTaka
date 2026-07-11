"""
Configuration constants for the semiconductor process regression project.
"""

FEATURE_COLUMNS = ["RF_Power", "Pressure", "Temperature", "Gas_Flow_Rate"]
TARGET_COLUMNS = ["Thickness", "Uniformity"]

RANDOM_STATE = 42

DATA_PATH = "data/process_data.csv"
ARTIFACTS_DIR = "artifacts"
PLOTS_DIR = "outputs/plots"

# If missing ratio for a column > this threshold, use median imputation; else drop rows
MISSING_THRESHOLD = 0.3

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

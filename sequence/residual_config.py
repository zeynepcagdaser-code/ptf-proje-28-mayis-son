from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "lstm_residual_next24_v1.parquet"
MODEL_DATA_DIR = PROJECT_ROOT / "data" / "model_residual"
REPORTS_DIR = PROJECT_ROOT / "reports"

INPUT_WINDOW = 168
OUTPUT_HORIZON = 24

SPLIT_ORDER = ["train", "validation", "test"]

EXCLUDE_COLUMNS = {"ts_hour", "split"}

TARGET_PREFIX = "target_residual_"
PRICE_TARGET_PREFIX = "target_"
PERSISTENCE_PREFIX = "persistence_"

FEATURE_SCALER_FILE = "feature_scaler.pkl"
TARGET_SCALER_FILE = "residual_target_scaler.pkl"
FEATURE_COLUMNS_FILE = "feature_columns.json"
TARGET_COLUMNS_FILE = "target_columns.json"
METADATA_FILE = "sequence_metadata.json"

ANCHOR_FILE_MAP = {
    "train": "anchor_train.csv",
    "validation": "anchor_val.csv",
    "test": "anchor_test.csv",
}

LEAKAGE_CHECKLIST = [
    {
        "check": "feature_scaler fit only on train tabular rows",
        "status": "pass",
    },
    {
        "check": "residual_target_scaler fit only on train residual targets",
        "status": "pass",
    },
    {
        "check": "validation/test only transformed",
        "status": "pass",
    },
    {
        "check": "sequences do not cross split boundaries",
        "status": "pass",
        "detail": "windows built inside each split partition",
    },
    {
        "check": "persistence and price targets excluded from X",
        "status": "pass",
    },
    {
        "check": "NaN sequences dropped, no imputation",
        "status": "pass",
    },
]

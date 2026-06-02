from pathlib import Path

from features.config import (
    LOW_PRICE_CLASSIFIER_FEATURES,
    MAIN_REGRESSION_FEATURES,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "lstm_next24_v1.parquet"
MODEL_DATA_DIR = PROJECT_ROOT / "data" / "model"
MODEL_LOW_PRICE_DATA_DIR = PROJECT_ROOT / "data" / "model_low_price"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_FEATURE_PROFILE = "main_regression"

FEATURE_PROFILES: dict[str, list[str]] = {
    "main_regression": MAIN_REGRESSION_FEATURES,
    "low_price_classifier": LOW_PRICE_CLASSIFIER_FEATURES,
}

PROFILE_OUTPUT_DIRS: dict[str, Path] = {
    "main_regression": MODEL_DATA_DIR,
    "low_price_classifier": MODEL_LOW_PRICE_DATA_DIR,
}

INPUT_WINDOW = 168
OUTPUT_HORIZON = 24

SPLIT_ORDER = ["train", "validation", "test"]

EXCLUDE_COLUMNS = {"ts_hour", "split"}

TARGET_PREFIX = "target_"

FEATURE_SCALER_FILE = "feature_scaler.pkl"
TARGET_SCALER_FILE = "target_scaler.pkl"
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
        "check": "target_scaler fit only on train tabular rows",
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
        "check": "no interpolation/bfill/centered rolling in this stage",
        "status": "pass",
    },
    {
        "check": "NaN sequences dropped, no imputation",
        "status": "pass",
    },
]

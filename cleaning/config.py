from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data"
CLEAN_DATA_DIR = PROJECT_ROOT / "data" / "clean"
REPORTS_DIR = PROJECT_ROOT / "reports"

TIMEZONE = "Europe/Istanbul"

HOURLY_CSV_SOURCES = {
    "ptf": {
        "path": RAW_DATA_DIR / "ptf_dataset.csv",
        "dedupe_keys": ["date", "hour"],
        "drop_columns": [],
        "price_columns": ["price", "priceUsd", "priceEur"],
    },
    "kgup": {
        "path": RAW_DATA_DIR / "kgup_combined.csv",
        "dedupe_keys": ["date", "time"],
        "drop_columns": [],
    },
    "load_forecast": {
        "path": RAW_DATA_DIR / "load_forecast.csv",
        "dedupe_keys": ["date", "time"],
        "drop_columns": [],
    },
    "realtime_generation": {
        "path": RAW_DATA_DIR / "realtime_generation.csv",
        "dedupe_keys": ["date", "hour"],
        "drop_columns": ["naphta", "lng"],
        "clip_non_negative": ["sun"],
    },
    "real_consumption": {
        "path": RAW_DATA_DIR / "real_consumption.csv",
        "dedupe_keys": ["date", "time"],
        "drop_columns": [],
    },
    "smf": {
        "path": RAW_DATA_DIR / "smf.csv",
        "dedupe_keys": ["date", "hour"],
        "drop_columns": ["hour"],
        "price_columns": ["systemMarginalPrice"],
    },
    "yal_yat": {
        "path": RAW_DATA_DIR / "yal_yat.csv",
        "dedupe_keys": ["date", "hour"],
        "drop_columns": ["upRegulationTwoCoded", "downRegulationTwoCoded"],
    },
}

WIND_CSV = RAW_DATA_DIR / "wind_forecast.csv"
OUTAGES_CSV = RAW_DATA_DIR / "outages.csv"

EXPECTED_INTERVALS_PER_HOUR = 6
PRICE_CAP_TL = 4800.0

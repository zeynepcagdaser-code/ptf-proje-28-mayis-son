from pathlib import Path

from features.config import LEAKAGE_CHECKLIST, REPORTS_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "lstm_residual_next24_v1.parquet"

RESIDUAL_LEAKAGE_CHECKLIST = LEAKAGE_CHECKLIST + [
    {
        "check": "Persistence uses PTF(t+h-24) only — max lag is PTF(t) at h=24",
        "status": "pass",
        "detail": "persistence_h = ptf_price.shift(24-h) at anchor t",
    },
    {
        "check": "Residual target = future PTF minus persistence (not in feature matrix)",
        "status": "pass",
        "detail": "target_residual_h stored separately from X features",
    },
]

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "lstm_tree_micro_v1.parquet"
MODEL_DIR = PROJECT_ROOT / "models" / "tree_advanced"
ANCHOR_TEST_PATH = PROJECT_ROOT / "data" / "model" / "anchor_test.csv"
PREDICTIONS_CSV = PROJECT_ROOT / "data" / "predictions" / "tree_advanced_test_predictions.csv"
METRICS_JSON = PROJECT_ROOT / "reports" / "tree_advanced_metrics.json"
METRICS_MD = PROJECT_ROOT / "reports" / "tree_advanced_metrics.md"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
RESIDUAL_PRED_CSV = PROJECT_ROOT / "data" / "predictions" / "lstm_residual_test_predictions.csv"

HORIZONS = list(range(1, 25))
HOURS = list(range(24))
SPIKE_THRESHOLD = 4800.0
ZERO_THRESHOLD = 0.5
MAPE_MASK_THRESHOLD = 100.0
EARLY_STOPPING_ROUNDS = 50
MAX_BOOST_ROUNDS = 1500
DEFAULT_RECENCY_DAYS = 60
DEFAULT_RECENCY_BOOST = 3.0
ZERO_CLASS_THRESHOLD = 0.92
SPIKE_CLASS_THRESHOLD = 0.88
APPLY_CLASSIFIER_OVERRIDES = True
MIN_ROWS_PER_HOUR = 80

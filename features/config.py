from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MASTER_PATH = PROJECT_ROOT / "data" / "master" / "master_hourly_v1.parquet"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
REPORTS_DIR = PROJECT_ROOT / "reports"

OUTPUT_PATH = FEATURES_DIR / "lstm_next24_v1.parquet"

INPUT_WINDOW = 168
OUTPUT_HORIZON = 24

PTF_COL = "ptf_price"
SMF_COL = "smf_systemMarginalPrice"

TARGET_HORIZONS = list(range(1, OUTPUT_HORIZON + 1))
LAG_STEPS = [24, 48, 168]

SPLIT_RANGES = {
    "train": (2020, 2024),
    "validation": (2025, 2025),
    "test": (2026, 2026),
}

# Same-hour safe (planned / forecast / outage aggregates at t).
KGUP_FEATURE_COLS = [
    "kgup_toplam",
    "kgup_dogalgaz",
    "kgup_ruzgar",
    "kgup_linyit",
    "kgup_tasKomur",
    "kgup_ithalKomur",
    "kgup_fuelOil",
    "kgup_jeotermal",
    "kgup_barajli",
    "kgup_nafta",
    "kgup_biokutle",
    "kgup_akarsu",
    "kgup_gunes",
    "kgup_diger",
]

LOAD_FEATURE_COLS = ["load_lep"]

WIND_FORECAST_COLS = [
    "wind_quarter1_mean",
    "wind_quarter2_mean",
    "wind_quarter3_mean",
    "wind_quarter4_mean",
    "wind_forecast_mean",
    "wind_forecast_min",
    "wind_forecast_max",
    "wind_forecast_std",
]

OUTAGE_FEATURE_COLS = [
    "outage_event_rows",
    "outage_fault_event_count",
    "outage_fault_mw_loss_sum",
    "outage_fault_mw_loss_max",
    "outage_fault_operator_power_sum",
    "outage_maint_event_count",
    "outage_maint_capacity_sum",
    "outage_maint_operator_power_sum",
]

# Feature selection buckets (for planned next steps: separate heads/models).
# These lists are NOT yet enforced by the dataset builder; they exist as an explicit contract
# for future model wiring and to avoid accidentally stuffing every new feature into the main regressor.
MAIN_REGRESSION_FEATURES: list[str] = [
    # The primary dataset builder still includes the base KGUP/load/wind/outage + lags.
    # This list is intended for "engineered" features to prioritize in price regression.
    "kgup_total_minus_load",
    "kgup_renewable_share",
    "kgup_thermal_share",
    "wind_forecast_share",
    "smf_ptf_spread_lag_24",
    "smf_ptf_spread_lag_168",
    "ptf_lag_24",
    "ptf_lag_168",
    "ptf_roll_mean_24",
    "ptf_roll_std_24",
    "ptf_roll_mean_168",
    "ptf_roll_std_168",
    # Downside / market structure proxies
    "gas_share",
    "coal_share",
    "gas_coal_balance",
    "gas_coal_competition_index",
    "thermal_price_setting_share",
    "renewable_pressure",
    "renewable_suppression_pressure",
]

LOW_PRICE_CLASSIFIER_FEATURES: list[str] = [
    "low_load_flag",
    "holiday_low_load_flag",
    "solar_peak_hour_flag",
    "zero_price_risk_proxy",
]

RISK_DASHBOARD_FEATURES: list[str] = [
    # Placeholder bucket for features we might use in monitoring but not in training.
]

# Realized / balancing — only lagged copies at 24/48/168.
LAGGED_SOURCE_COLS = [
    "gen_total",
    "cons_consumption",
    SMF_COL,
    "yal_yat_net",
    "yal_yat_upRegulationDelivered",
    "yal_yat_downRegulationDelivered",
    "wind_generation_mean",
]

FFILL_LIMIT = 2

LEAKAGE_CHECKLIST = [
    {
        "check": "No same-hour gen_* in feature matrix",
        "status": "pass",
        "detail": "Only gen_total_lag_{24,48,168} included",
    },
    {
        "check": "No same-hour cons_* in feature matrix",
        "status": "pass",
        "detail": "Only cons_consumption_lag_{24,48,168} included",
    },
    {
        "check": "No same-hour smf_* in feature matrix",
        "status": "pass",
        "detail": "Only smf_systemMarginalPrice_lag_{24,48,168} and spread lags",
    },
    {
        "check": "No same-hour yal_yat_* in feature matrix",
        "status": "pass",
        "detail": "Only selected yal_yat_* lag_{24,48,168}",
    },
    {
        "check": "No same-hour wind_generation_* in feature matrix",
        "status": "pass",
        "detail": "Only wind_generation_mean_lag_{24,48,168}",
    },
    {
        "check": "PTF rolling features use data through t-1 only",
        "status": "pass",
        "detail": "shift(1) before rolling window",
    },
    {
        "check": "Targets are future PTF only (t+1..t+24)",
        "status": "pass",
        "detail": "target_kh = ptf_price.shift(-k)",
    },
    {
        "check": "No interpolation or bfill on features",
        "status": "pass",
        "detail": "Optional ffill(limit=2) past-only on features after missing report",
    },
    {
        "check": "kgup_*, load_lep, wind_forecast_* at same hour t",
        "status": "pass",
        "detail": "Planned/forecast availability class",
    },
    {
        "check": "outage_* at same hour t",
        "status": "review",
        "detail": "Event aggregates may be revised retroactively; use with caution for strict DAM cutoff",
    },
    {
        "check": "is_holiday_tr / is_holiday_or_weekend from ts_hour calendar only",
        "status": "pass",
        "detail": "holidays.Turkey on anchor date; no future data",
    },
]

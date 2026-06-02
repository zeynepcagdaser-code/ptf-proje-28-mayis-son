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

# Feature selection buckets — contract for model heads and inventory audit.
# The dataset builder still emits additional columns (lags, aliases); models should
# filter with resolve_feature_list() so missing optional columns never crash pipelines.
MAIN_REGRESSION_FEATURES: list[str] = [
    # PTF history
    "ptf_lag_24",
    "ptf_lag_168",
    "ptf_roll_mean_24",
    "ptf_roll_std_24",
    "ptf_roll_mean_168",
    "ptf_roll_std_168",
    # Calendar
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_holiday_tr",
    # Demand / balance
    "load_lep",
    "kgup_total_minus_load",
    # Market structure (planned KGUP-derived)
    "renewable_pressure",
    "renewable_suppression_pressure",
    "thermal_price_setting_share",
    "gas_share",
    "coal_share",
    "gas_coal_competition_index",
    "kgup_renewable_share",
    "kgup_thermal_share",
    # Wind
    "wind_forecast_mean",
    "wind_forecast_share",
    "wind_quarter1_mean",
    # Selected KGUP MW (not full 14-column mix when shares are present)
    "kgup_toplam",
    "kgup_dogalgaz",
    "kgup_ruzgar",
    "kgup_gunes",
    "kgup_barajli",
    "kgup_akarsu",
    "kgup_ithalKomur",
    "kgup_linyit",
    "kgup_tasKomur",
    # Outage (maint + fault counts)
    "outage_maint_event_count",
    "outage_maint_capacity_sum",
    "outage_fault_event_count",
    "outage_event_rows",
    # FİBA/FİBS (DAM price-independent orders)
    "dam_price_independent_buy_mwh",
    "dam_price_independent_sell_mwh",
    "fiba_fibs_ratio",
    "fiba_fibs_balance",
    "fiba_fibs_pressure",
    # FİBA/FİBS lagged (strict forecast alternative)
    "fiba_fibs_ratio_lag_24",
    "fiba_fibs_pressure_lag_24",
    "fiba_fibs_ratio_lag_168",
    "fiba_fibs_pressure_lag_168",
    # GRF (daily reference price) - prefer timing-safer variants
    "grf_tl_lag_1d",
    "grf_tl_change_7d",
    "grf_tl_rolling_mean_7d",
    "gas_cost_pressure_lag_1d",
    "thermal_cost_pressure_lag_1d",
    "gas_marginal_pressure_lag_1d",
    # DAM microstructure (offer/match/block)
    "dam_bid_volume_mwh",
    "dam_sell_offer_volume_mwh",
    "dam_matched_volume_mwh",
    "dam_bid_to_match_ratio",
    "dam_sell_to_match_ratio",
    "dam_buy_sell_ratio",
    "dam_offer_supply_demand_gap",
    "dam_offer_balance_pressure",
    "dam_match_ratio",
    "dam_unmatched_buy_proxy",
    "dam_block_unmatched_ratio",
    "dam_block_pressure",
    # strict-forecast alternatives
    "dam_bid_volume_lag_24",
    "dam_sell_offer_volume_lag_24",
    "dam_buy_sell_ratio_lag_24",
    "dam_offer_balance_pressure_lag_24",
    "dam_match_ratio_lag_24",
    "dam_block_unmatched_ratio_lag_24",
]

LOW_PRICE_CLASSIFIER_FEATURES: list[str] = [
    "low_load_flag",
    "holiday_low_load_flag",
    "solar_peak_hour_flag",
    "zero_price_risk_proxy",
    "renewable_pressure",
    "renewable_suppression_pressure",
    "res_share",
    "solar_share",
    "hydro_share",
    "kgup_renewable_share",
    "kgup_gunes",
    "kgup_ruzgar",
    "wind_forecast_mean",
    "wind_forecast_share",
    "gas_share",
    "coal_share",
    "thermal_price_setting_share",
    "gas_coal_competition_index",
    "hour_sin",
    "hour_cos",
    "is_holiday_or_weekend",
    "is_weekend",
    "is_holiday_tr",
    "load_lep",
    "ptf_lag_1",
    "ptf_lag_2",
    "ptf_lag_3",
    "ptf_lag_24",
    "ptf_lag_168",
    "ptf_roll_mean_24",
    "ptf_roll_mean_168",
    "ptf_roll_min_24",
    "ptf_roll_max_24",
    "ptf_roll_min_168",
    "ptf_roll_max_168",
    "ptf_low_count_24",
    "ptf_zero_count_24",
    "ptf_low_count_168",
    "ptf_zero_count_168",
    "ptf_low_ratio_24",
    "ptf_zero_ratio_24",
    "ptf_low_ratio_168",
    "ptf_zero_ratio_168",
    # FİBA/FİBS
    "fiba_fibs_ratio",
    "fiba_fibs_pressure",
    "fiba_fibs_ratio_lag_24",
    "fiba_fibs_pressure_lag_24",
    "fiba_fibs_ratio_lag_168",
    "fiba_fibs_pressure_lag_168",
    # DAM microstructure (signal for tightness / imbalance)
    "dam_buy_sell_ratio",
    "dam_offer_balance_pressure",
    "dam_match_ratio",
    "dam_unmatched_buy_proxy",
    "dam_block_unmatched_ratio",
    "dam_buy_sell_ratio_lag_24",
    "dam_offer_balance_pressure_lag_24",
    "dam_match_ratio_lag_24",
]

RISK_DASHBOARD_FEATURES: list[str] = [
    "price_cap",
    "ptf_to_cap_ratio",
    "smf_to_cap_ratio",
    "smf_lag_24",
    "smf_lag_48",
    "smf_lag_168",
    "smf_ptf_spread_lag_24",
    "smf_ptf_spread_lag_168",
    "yal_yat_net_lag_24",
    "yal_yat_net_lag_48",
    "yal_yat_net_lag_168",
    "yal_yat_upRegulationDelivered_lag_24",
    "yal_yat_downRegulationDelivered_lag_24",
    "gen_total_lag_24",
    "cons_consumption_lag_24",
    "wind_generation_mean_lag_24",
    "wind_generation_mean_lag_48",
    "wind_generation_mean_lag_168",
    # GRF (monitoring; timing may be uncertain)
    "grf_tl_1000sm3",
    "grf_usd_1000sm3",
    "grf_eur_mwh",
    "grf_usd_mmbtu",
    "gas_cost_pressure",
    "thermal_cost_pressure",
    "gas_marginal_pressure",
    # DAM microstructure monitoring
    "dam_bid_volume_mwh",
    "dam_sell_offer_volume_mwh",
    "dam_matched_buy_mwh",
    "dam_matched_sell_mwh",
    "dam_block_matched_buy_mwh",
    "dam_block_unmatched_buy_mwh",
    "dam_offer_supply_demand_gap",
]

# Built in parquet but intentionally excluded from MAIN_REGRESSION_FEATURES (audit / docs).
EXCLUDED_FROM_MAIN_REGRESSION: list[str] = [
    # Risk / balancing → RISK_DASHBOARD_FEATURES
    "smf_ptf_spread_lag_24",
    "smf_ptf_spread_lag_168",
    "price_cap",
    "ptf_to_cap_ratio",
    "smf_to_cap_ratio",
    "smf_lag_24",
    "smf_lag_48",
    "smf_lag_168",
    "yal_yat_net_lag_24",
    "yal_yat_net_lag_48",
    "yal_yat_net_lag_168",
    "yal_yat_upRegulationDelivered_lag_24",
    "yal_yat_upRegulationDelivered_lag_48",
    "yal_yat_upRegulationDelivered_lag_168",
    "yal_yat_downRegulationDelivered_lag_24",
    "yal_yat_downRegulationDelivered_lag_48",
    "yal_yat_downRegulationDelivered_lag_168",
    "gen_total_lag_24",
    "gen_total_lag_48",
    "gen_total_lag_168",
    "cons_consumption_lag_24",
    "cons_consumption_lag_48",
    "cons_consumption_lag_168",
    "wind_generation_mean_lag_24",
    "wind_generation_mean_lag_48",
    "wind_generation_mean_lag_168",
    # Low-price classifier only
    "low_load_flag",
    "holiday_low_load_flag",
    "solar_peak_hour_flag",
    "zero_price_risk_proxy",
    "res_share",
    "solar_share",
    "hydro_share",
    "is_holiday_or_weekend",
    # Redundant / duplicate PTF columns
    "gas_coal_balance",
    "ptf_lag_1",
    "ptf_lag_2",
    "ptf_lag_3",
    "ptf_lag_48",
    "ptf_lag_1h",
    "ptf_lag_2h",
    "ptf_lag_3h",
    "ptf_lag_24h",
    "ptf_lag_168h",
    "ptf_rolling_mean_24h",
    "ptf_rolling_std_24h",
    "ptf_rolling_mean_168h",
    # Extra KGUP / wind / outage (not in main contract)
    "kgup_fuelOil",
    "kgup_jeotermal",
    "kgup_nafta",
    "kgup_biokutle",
    "kgup_diger",
    "wind_quarter2_mean",
    "wind_quarter3_mean",
    "wind_quarter4_mean",
    "wind_forecast_min",
    "wind_forecast_max",
    "wind_forecast_std",
    "outage_fault_mw_loss_sum",
    "outage_fault_mw_loss_max",
    "outage_fault_operator_power_sum",
    "outage_maint_operator_power_sum",
]

# Thesis / market raw-data gaps (not in pipeline until sources exist).
THESIS_DATA_DEBT_GROUPS: list[dict[str, str]] = [
    {
        "group": "Fiyattan bağımsız alış/satış oranı",
        "target_features": "buy_sell_ratio, order_book_imbalance_proxy",
        "raw_sources": "GÖP/DAM işlem veya emir defteri (alış/satış hacmi ayrımı)",
    },
    {
        "group": "BOTAŞ doğal gaz tarifesi",
        "target_features": "botas_gas_tariff, botas_tariff_lag_30d",
        "raw_sources": "BOTAŞ / EPDK tarihsel tarife tablosu",
    },
    {
        "group": "USD/TL, EUR/TL",
        "target_features": "usd_try, eur_try, fx_vol_30d",
        "raw_sources": "TCMB / ECB günlük kur serisi",
    },
    {
        "group": "TTF / Brent",
        "target_features": "ttf_gas_price, brent_oil_price, ttf_try_proxy",
        "raw_sources": "ICE TTF, Brent; ttf_try_proxy = ttf * usd_try",
    },
    {
        "group": "TETAŞ-EÜAŞ tarife",
        "target_features": "tetas_euas_tariff, regulated_tariff_index",
        "raw_sources": "EPDK / şirket duyuruları",
    },
    {
        "group": "Mesken AG tarife",
        "target_features": "residential_ag_tariff",
        "raw_sources": "EPDK perakende tarife",
    },
    {
        "group": "Doğal gaz santrali yakıt maliyeti",
        "target_features": "gas_plant_fuel_cost, fuel_cost_index, gas_cost_pressure, gas_marginal_pressure",
        "raw_sources": "BOTAŞ tarife + TTF + verimlilik / kgup gaz MW",
    },
    {
        "group": "Güneş / gökyüzü açıklığı veya solar radiation proxy",
        "target_features": "clearness_index, solar_radiation_proxy, sky_clear_fraction",
        "raw_sources": "Meteoroloji, PVGIS veya bulutluluk (şu an yalnızca solar_peak_hour_flag)",
    },
    {
        "group": "YEKDEM / merchant / non-merchant ayrımı",
        "target_features": "yekdem_unit_price, merchant_proxy_share, non_merchant_proxy_share, yekdem_revenue_loss_proxy",
        "raw_sources": "YEKDEM birim fiyat, KGUP müst run / sözleşme sınıflandırması",
    },
]


def resolve_feature_list(
    requested: list[str],
    available_columns: list[str] | set[str],
) -> tuple[list[str], list[str]]:
    """Return (present, missing) without raising if optional columns are absent."""
    avail = set(available_columns)
    present = [c for c in requested if c in avail]
    missing = [c for c in requested if c not in avail]
    return present, missing

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

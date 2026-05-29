# LSTM Feature Dataset Report

- **Generated (UTC):** 2026-05-29T20:35:57.863289+00:00
- **Output:** `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/features/lstm_next24_v1.parquet`
- **Source:** `data/master/master_hourly_v1.parquet`

## Training format (reference)

- **Input window:** 168 hours
- **Output horizon:** 24 hours
- **Index mapping:** `X[t-167:t] -> y[t+1:t+24] at anchor ts_hour=t`

## Row counts

- Master rows: 56208
- Dropped (missing targets): 24
- Dropped (insufficient history for lags/rolls): 168
- **Final rows:** 56016

## Features & targets

- Feature columns: 75
- Target columns: 24
- ts_hour range: 2020-01-08 00:00:00+03:00 → 2026-05-29 23:00:00+03:00

## Split counts

| Split | Rows |
|-------|-----:|
| train | 43680 |
| validation | 8760 |
| test | 3576 |

## Missing features (before ffill, % > 0)

- `wind_generation_mean_lag_168`: 0.8116%
- `wind_generation_mean_lag_48`: 0.5980%
- `wind_generation_mean_lag_24`: 0.5553%
- `ptf_lag_168`: 0.2990%
- `ptf_roll_mean_168`: 0.2990%
- `ptf_roll_std_168`: 0.2990%
- `smf_ptf_spread_lag_168`: 0.2990%
- `gen_total_lag_168`: 0.2990%
- `cons_consumption_lag_168`: 0.2990%
- `smf_lag_168`: 0.2990%
- `yal_yat_net_lag_168`: 0.2990%
- `yal_yat_upRegulationDelivered_lag_168`: 0.2990%
- `yal_yat_downRegulationDelivered_lag_168`: 0.2990%
- `outage_fault_mw_loss_sum`: 0.2563%
- `outage_fault_mw_loss_max`: 0.2563%
- `outage_fault_operator_power_sum`: 0.2563%
- `outage_maint_capacity_sum`: 0.1353%
- `outage_maint_operator_power_sum`: 0.1353%
- `ptf_lag_48`: 0.0854%
- `gen_total_lag_48`: 0.0854%
- `cons_consumption_lag_48`: 0.0854%
- `smf_lag_48`: 0.0854%
- `yal_yat_net_lag_48`: 0.0854%
- `yal_yat_upRegulationDelivered_lag_48`: 0.0854%
- `yal_yat_downRegulationDelivered_lag_48`: 0.0854%
- `wind_forecast_std`: 0.0765%
- `wind_forecast_share`: 0.0481%
- `wind_quarter2_mean`: 0.0481%
- `wind_quarter3_mean`: 0.0481%
- `wind_quarter4_mean`: 0.0481%
- `wind_forecast_mean`: 0.0481%
- `wind_forecast_min`: 0.0481%
- `wind_forecast_max`: 0.0481%
- `ptf_lag_24`: 0.0427%
- `ptf_roll_mean_24`: 0.0427%
- `ptf_roll_std_24`: 0.0427%
- `smf_ptf_spread_lag_24`: 0.0427%
- `kgup_total_minus_load`: 0.0427%
- `load_lep`: 0.0427%
- `gen_total_lag_24`: 0.0427%
- `cons_consumption_lag_24`: 0.0427%
- `smf_lag_24`: 0.0427%
- `yal_yat_net_lag_24`: 0.0427%
- `yal_yat_upRegulationDelivered_lag_24`: 0.0427%
- `yal_yat_downRegulationDelivered_lag_24`: 0.0427%
- `ptf_lag_1`: 0.0018%

## Missing features (after ffill limit=2, % > 0)

- `wind_generation_mean_lag_168`: 0.7244%
- `wind_generation_mean_lag_48`: 0.5108%
- `wind_generation_mean_lag_24`: 0.4681%
- `ptf_lag_168`: 0.2990%
- `ptf_roll_mean_168`: 0.2990%
- `ptf_roll_std_168`: 0.2990%
- `smf_ptf_spread_lag_168`: 0.2990%
- `gen_total_lag_168`: 0.2990%
- `cons_consumption_lag_168`: 0.2990%
- `smf_lag_168`: 0.2990%
- `yal_yat_net_lag_168`: 0.2990%
- `yal_yat_upRegulationDelivered_lag_168`: 0.2990%
- `yal_yat_downRegulationDelivered_lag_168`: 0.2990%
- `outage_fault_mw_loss_sum`: 0.2492%
- `outage_fault_mw_loss_max`: 0.2492%
- `outage_fault_operator_power_sum`: 0.2492%
- `outage_maint_capacity_sum`: 0.1282%
- `outage_maint_operator_power_sum`: 0.1282%
- `ptf_lag_48`: 0.0854%
- `gen_total_lag_48`: 0.0854%
- `cons_consumption_lag_48`: 0.0854%
- `smf_lag_48`: 0.0854%
- `yal_yat_net_lag_48`: 0.0854%
- `yal_yat_upRegulationDelivered_lag_48`: 0.0854%
- `yal_yat_downRegulationDelivered_lag_48`: 0.0854%
- `ptf_lag_24`: 0.0427%
- `ptf_roll_mean_24`: 0.0427%
- `ptf_roll_std_24`: 0.0427%
- `smf_ptf_spread_lag_24`: 0.0427%
- `gen_total_lag_24`: 0.0427%
- `cons_consumption_lag_24`: 0.0427%
- `smf_lag_24`: 0.0427%
- `yal_yat_net_lag_24`: 0.0427%
- `yal_yat_upRegulationDelivered_lag_24`: 0.0427%
- `yal_yat_downRegulationDelivered_lag_24`: 0.0427%
- `wind_forecast_share`: 0.0409%
- `wind_quarter2_mean`: 0.0409%
- `wind_quarter3_mean`: 0.0409%
- `wind_quarter4_mean`: 0.0409%
- `wind_forecast_mean`: 0.0409%
- `wind_forecast_min`: 0.0409%
- `wind_forecast_max`: 0.0409%
- `wind_forecast_std`: 0.0409%
- `kgup_total_minus_load`: 0.0392%
- `load_lep`: 0.0392%
- `ptf_lag_1`: 0.0018%

## Leakage checklist

- [PASS] No same-hour gen_* in feature matrix: Only gen_total_lag_{24,48,168} included
- [PASS] No same-hour cons_* in feature matrix: Only cons_consumption_lag_{24,48,168} included
- [PASS] No same-hour smf_* in feature matrix: Only smf_systemMarginalPrice_lag_{24,48,168} and spread lags
- [PASS] No same-hour yal_yat_* in feature matrix: Only selected yal_yat_* lag_{24,48,168}
- [PASS] No same-hour wind_generation_* in feature matrix: Only wind_generation_mean_lag_{24,48,168}
- [PASS] PTF rolling features use data through t-1 only: shift(1) before rolling window
- [PASS] Targets are future PTF only (t+1..t+24): target_kh = ptf_price.shift(-k)
- [PASS] No interpolation or bfill on features: Optional ffill(limit=2) past-only on features after missing report
- [PASS] kgup_*, load_lep, wind_forecast_* at same hour t: Planned/forecast availability class
- [REVIEW] outage_* at same hour t: Event aggregates may be revised retroactively; use with caution for strict DAM cutoff
- [PASS] is_holiday_tr / is_holiday_or_weekend from ts_hour calendar only: holidays.Turkey on anchor date; no future data
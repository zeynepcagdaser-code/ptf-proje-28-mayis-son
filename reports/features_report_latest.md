# LSTM Feature Dataset Report

- **Generated (UTC):** 2026-06-02T07:13:49.061951+00:00
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

- Feature columns: 169
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
- `dam_block_unmatched_ratio_lag_24`: 0.4699%
- `dam_block_unmatched_ratio`: 0.4272%
- `grf_tl_change_7d`: 0.3417%
- `grf_tl_rolling_mean_7d`: 0.3400%
- `ptf_lag_168`: 0.2990%
- `ptf_lag_168h`: 0.2990%
- `ptf_roll_mean_168`: 0.2990%
- `ptf_roll_std_168`: 0.2990%
- `ptf_rolling_mean_168h`: 0.2990%
- `ptf_roll_min_168`: 0.2990%
- `ptf_roll_max_168`: 0.2990%
- `smf_ptf_spread_lag_168`: 0.2990%
- `fiba_fibs_ratio_lag_168`: 0.2990%
- `fiba_fibs_pressure_lag_168`: 0.2990%
- `dam_bid_volume_lag_168`: 0.2990%
- `dam_sell_offer_volume_lag_168`: 0.2990%
- `dam_buy_sell_ratio_lag_168`: 0.2990%
- `dam_offer_balance_pressure_lag_168`: 0.2990%
- `gen_total_lag_168`: 0.2990%
- `cons_consumption_lag_168`: 0.2990%
- `smf_lag_168`: 0.2990%
- `yal_yat_net_lag_168`: 0.2990%
- `yal_yat_upRegulationDelivered_lag_168`: 0.2990%
- `yal_yat_downRegulationDelivered_lag_168`: 0.2990%
- `ptf_low_count_168`: 0.2972%
- `ptf_zero_count_168`: 0.2972%
- `ptf_low_ratio_168`: 0.2972%
- `ptf_zero_ratio_168`: 0.2972%
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
- `ptf_lag_24h`: 0.0427%
- `ptf_roll_mean_24`: 0.0427%
- `ptf_roll_std_24`: 0.0427%
- `ptf_rolling_mean_24h`: 0.0427%
- `ptf_rolling_std_24h`: 0.0427%
- `ptf_roll_min_24`: 0.0427%
- `ptf_roll_max_24`: 0.0427%
- `smf_ptf_spread_lag_24`: 0.0427%
- `dam_price_independent_buy_lag_24`: 0.0427%
- `dam_price_independent_sell_lag_24`: 0.0427%
- `fiba_fibs_ratio_lag_24`: 0.0427%
- `fiba_fibs_balance_lag_24`: 0.0427%
- `fiba_fibs_pressure_lag_24`: 0.0427%
- `grf_tl_lag_1d`: 0.0427%
- `gas_cost_pressure_lag_1d`: 0.0427%
- `thermal_cost_pressure_lag_1d`: 0.0427%
- `gas_marginal_pressure_lag_1d`: 0.0427%
- `dam_bid_volume_lag_24`: 0.0427%
- `dam_sell_offer_volume_lag_24`: 0.0427%
- `dam_buy_sell_ratio_lag_24`: 0.0427%
- `dam_offer_balance_pressure_lag_24`: 0.0427%
- `dam_match_ratio_lag_24`: 0.0427%
- `kgup_total_minus_load`: 0.0427%
- `ptf_to_cap_ratio`: 0.0427%
- `smf_to_cap_ratio`: 0.0427%
- `load_lep`: 0.0427%
- `gen_total_lag_24`: 0.0427%
- `cons_consumption_lag_24`: 0.0427%
- `smf_lag_24`: 0.0427%
- `yal_yat_net_lag_24`: 0.0427%
- `yal_yat_upRegulationDelivered_lag_24`: 0.0427%
- `yal_yat_downRegulationDelivered_lag_24`: 0.0427%
- `ptf_low_count_24`: 0.0409%
- `ptf_zero_count_24`: 0.0409%
- `ptf_low_ratio_24`: 0.0409%
- `ptf_zero_ratio_24`: 0.0409%
- `ptf_lag_3`: 0.0053%
- `ptf_lag_3h`: 0.0053%
- `ptf_lag_2`: 0.0036%
- `ptf_lag_2h`: 0.0036%
- `ptf_lag_1`: 0.0018%
- `ptf_lag_1h`: 0.0018%

## Missing features (after ffill limit=2, % > 0)

- `wind_generation_mean_lag_168`: 0.7244%
- `wind_generation_mean_lag_48`: 0.5108%
- `wind_generation_mean_lag_24`: 0.4681%
- `dam_block_unmatched_ratio_lag_24`: 0.4663%
- `dam_block_unmatched_ratio`: 0.4236%
- `grf_tl_change_7d`: 0.3417%
- `grf_tl_rolling_mean_7d`: 0.3400%
- `ptf_lag_168`: 0.2990%
- `ptf_lag_168h`: 0.2990%
- `ptf_roll_mean_168`: 0.2990%
- `ptf_roll_std_168`: 0.2990%
- `ptf_rolling_mean_168h`: 0.2990%
- `ptf_roll_min_168`: 0.2990%
- `ptf_roll_max_168`: 0.2990%
- `smf_ptf_spread_lag_168`: 0.2990%
- `fiba_fibs_ratio_lag_168`: 0.2990%
- `fiba_fibs_pressure_lag_168`: 0.2990%
- `dam_bid_volume_lag_168`: 0.2990%
- `dam_sell_offer_volume_lag_168`: 0.2990%
- `dam_buy_sell_ratio_lag_168`: 0.2990%
- `dam_offer_balance_pressure_lag_168`: 0.2990%
- `gen_total_lag_168`: 0.2990%
- `cons_consumption_lag_168`: 0.2990%
- `smf_lag_168`: 0.2990%
- `yal_yat_net_lag_168`: 0.2990%
- `yal_yat_upRegulationDelivered_lag_168`: 0.2990%
- `yal_yat_downRegulationDelivered_lag_168`: 0.2990%
- `ptf_low_count_168`: 0.2972%
- `ptf_zero_count_168`: 0.2972%
- `ptf_low_ratio_168`: 0.2972%
- `ptf_zero_ratio_168`: 0.2972%
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
- `ptf_lag_24h`: 0.0427%
- `ptf_roll_mean_24`: 0.0427%
- `ptf_roll_std_24`: 0.0427%
- `ptf_rolling_mean_24h`: 0.0427%
- `ptf_rolling_std_24h`: 0.0427%
- `ptf_roll_min_24`: 0.0427%
- `ptf_roll_max_24`: 0.0427%
- `smf_ptf_spread_lag_24`: 0.0427%
- `dam_price_independent_buy_lag_24`: 0.0427%
- `dam_price_independent_sell_lag_24`: 0.0427%
- `fiba_fibs_ratio_lag_24`: 0.0427%
- `fiba_fibs_balance_lag_24`: 0.0427%
- `fiba_fibs_pressure_lag_24`: 0.0427%
- `grf_tl_lag_1d`: 0.0427%
- `gas_cost_pressure_lag_1d`: 0.0427%
- `thermal_cost_pressure_lag_1d`: 0.0427%
- `gas_marginal_pressure_lag_1d`: 0.0427%
- `dam_bid_volume_lag_24`: 0.0427%
- `dam_sell_offer_volume_lag_24`: 0.0427%
- `dam_buy_sell_ratio_lag_24`: 0.0427%
- `dam_offer_balance_pressure_lag_24`: 0.0427%
- `dam_match_ratio_lag_24`: 0.0427%
- `ptf_to_cap_ratio`: 0.0427%
- `smf_to_cap_ratio`: 0.0427%
- `gen_total_lag_24`: 0.0427%
- `cons_consumption_lag_24`: 0.0427%
- `smf_lag_24`: 0.0427%
- `yal_yat_net_lag_24`: 0.0427%
- `yal_yat_upRegulationDelivered_lag_24`: 0.0427%
- `yal_yat_downRegulationDelivered_lag_24`: 0.0427%
- `ptf_low_count_24`: 0.0409%
- `ptf_zero_count_24`: 0.0409%
- `ptf_low_ratio_24`: 0.0409%
- `ptf_zero_ratio_24`: 0.0409%
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
- `ptf_lag_3`: 0.0053%
- `ptf_lag_3h`: 0.0053%
- `ptf_lag_2`: 0.0036%
- `ptf_lag_2h`: 0.0036%
- `ptf_lag_1`: 0.0018%
- `ptf_lag_1h`: 0.0018%

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
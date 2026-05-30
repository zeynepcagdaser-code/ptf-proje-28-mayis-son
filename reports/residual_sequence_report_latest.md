# Sequence Dataset Report

- **Generated (UTC):** 2026-05-29T23:41:22.022382+00:00
- **Source:** `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/features/lstm_residual_next24_v1.parquet`
- **Output dir:** `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/model_residual`

## Configuration

- Window: **168**
- Horizon: **24**
- Mapping: `X[t-167:t] -> y[t+1:t+24]`
- Features: 75
- Targets: 24
- Scaler fit split: **train**

## Tensor shapes

| Split | X shape | y shape |
|-------|---------|---------|
| train | (42847, 168, 75) | (42847, 24) |
| validation | (8501, 168, 75) | (8501, 24) |
| test | (3282, 168, 75) | (3282, 24) |

## Sequence counts

| Split | Sequences |
|-------|----------:|
| train | 42847 |
| validation | 8501 |
| test | 3282 |

## Dropped sequences (NaN)

| Split | Dropped |
|-------|--------:|
| train | 0 |
| validation | 0 |
| test | 0 |

## Leakage checklist

- [PASS] feature_scaler fit only on train tabular rows
- [PASS] residual_target_scaler fit only on train residual targets
- [PASS] validation/test only transformed
- [PASS] sequences do not cross split boundaries — windows built inside each split partition
- [PASS] persistence and price targets excluded from X
- [PASS] NaN sequences dropped, no imputation

## Feature columns

- `hour_sin`
- `hour_cos`
- `dow_sin`
- `dow_cos`
- `month_sin`
- `month_cos`
- `is_weekend`
- `is_holiday_tr`
- `is_holiday_or_weekend`
- `ptf_lag_1`
- `ptf_lag_24`
- `ptf_lag_48`
- `ptf_lag_168`
- `ptf_roll_mean_24`
- `ptf_roll_std_24`
- `ptf_roll_mean_168`
- `ptf_roll_std_168`
- `smf_ptf_spread_lag_24`
- `smf_ptf_spread_lag_168`
- `kgup_total_minus_load`
- `kgup_renewable_share`
- `kgup_thermal_share`
- `wind_forecast_share`
- `kgup_toplam`
- `kgup_dogalgaz`
- `kgup_ruzgar`
- `kgup_linyit`
- `kgup_tasKomur`
- `kgup_ithalKomur`
- `kgup_fuelOil`
- `kgup_jeotermal`
- `kgup_barajli`
- `kgup_nafta`
- `kgup_biokutle`
- `kgup_akarsu`
- `kgup_gunes`
- `kgup_diger`
- `load_lep`
- `wind_quarter1_mean`
- `wind_quarter2_mean`
- `wind_quarter3_mean`
- `wind_quarter4_mean`
- `wind_forecast_mean`
- `wind_forecast_min`
- `wind_forecast_max`
- `wind_forecast_std`
- `outage_event_rows`
- `outage_fault_event_count`
- `outage_fault_mw_loss_sum`
- `outage_fault_mw_loss_max`
- `outage_fault_operator_power_sum`
- `outage_maint_event_count`
- `outage_maint_capacity_sum`
- `outage_maint_operator_power_sum`
- `gen_total_lag_24`
- `gen_total_lag_48`
- `gen_total_lag_168`
- `cons_consumption_lag_24`
- `cons_consumption_lag_48`
- `cons_consumption_lag_168`
- `smf_lag_24`
- `smf_lag_48`
- `smf_lag_168`
- `yal_yat_net_lag_24`
- `yal_yat_net_lag_48`
- `yal_yat_net_lag_168`
- `yal_yat_upRegulationDelivered_lag_24`
- `yal_yat_upRegulationDelivered_lag_48`
- `yal_yat_upRegulationDelivered_lag_168`
- `yal_yat_downRegulationDelivered_lag_24`
- `yal_yat_downRegulationDelivered_lag_48`
- `yal_yat_downRegulationDelivered_lag_168`
- `wind_generation_mean_lag_24`
- `wind_generation_mean_lag_48`
- `wind_generation_mean_lag_168`

## Target columns

- `target_residual_1h`
- `target_residual_2h`
- `target_residual_3h`
- `target_residual_4h`
- `target_residual_5h`
- `target_residual_6h`
- `target_residual_7h`
- `target_residual_8h`
- `target_residual_9h`
- `target_residual_10h`
- `target_residual_11h`
- `target_residual_12h`
- `target_residual_13h`
- `target_residual_14h`
- `target_residual_15h`
- `target_residual_16h`
- `target_residual_17h`
- `target_residual_18h`
- `target_residual_19h`
- `target_residual_20h`
- `target_residual_21h`
- `target_residual_22h`
- `target_residual_23h`
- `target_residual_24h`
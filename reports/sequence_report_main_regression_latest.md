# Sequence Dataset Report

- **Generated (UTC):** 2026-06-02T06:56:12.611317+00:00
- **Source:** `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/features/lstm_next24_v1.parquet`
- **Output dir:** `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/model`

## Configuration

- Window: **168**
- Horizon: **24**
- Mapping: `X[t-167:t] -> y[t+1:t+24]`
- Features: 55
- Targets: 24
- Scaler fit split: **train**

## Tensor shapes

| Split | X shape | y shape |
|-------|---------|---------|
| train | (43489, 168, 55) | (43489, 24) |
| validation | (8570, 168, 55) | (8570, 24) |
| test | (3387, 168, 55) | (3387, 24) |

## Sequence counts

| Split | Sequences |
|-------|----------:|
| train | 43489 |
| validation | 8570 |
| test | 3387 |

## Dropped sequences (NaN)

| Split | Dropped |
|-------|--------:|
| train | 0 |
| validation | 0 |
| test | 0 |

## Leakage checklist

- [PASS] feature_scaler fit only on train tabular rows
- [PASS] target_scaler fit only on train tabular rows
- [PASS] validation/test only transformed
- [PASS] sequences do not cross split boundaries — windows built inside each split partition
- [PASS] no interpolation/bfill/centered rolling in this stage
- [PASS] NaN sequences dropped, no imputation

## Feature columns

- `ptf_lag_24`
- `ptf_lag_168`
- `ptf_roll_mean_24`
- `ptf_roll_std_24`
- `ptf_roll_mean_168`
- `ptf_roll_std_168`
- `hour_sin`
- `hour_cos`
- `dow_sin`
- `dow_cos`
- `month_sin`
- `month_cos`
- `is_weekend`
- `is_holiday_tr`
- `load_lep`
- `kgup_total_minus_load`
- `renewable_pressure`
- `renewable_suppression_pressure`
- `thermal_price_setting_share`
- `gas_share`
- `coal_share`
- `gas_coal_competition_index`
- `kgup_renewable_share`
- `kgup_thermal_share`
- `wind_forecast_mean`
- `wind_forecast_share`
- `wind_quarter1_mean`
- `kgup_toplam`
- `kgup_dogalgaz`
- `kgup_ruzgar`
- `kgup_gunes`
- `kgup_barajli`
- `kgup_akarsu`
- `kgup_ithalKomur`
- `kgup_linyit`
- `kgup_tasKomur`
- `outage_maint_event_count`
- `outage_maint_capacity_sum`
- `outage_fault_event_count`
- `outage_event_rows`
- `dam_price_independent_buy_mwh`
- `dam_price_independent_sell_mwh`
- `fiba_fibs_ratio`
- `fiba_fibs_balance`
- `fiba_fibs_pressure`
- `fiba_fibs_ratio_lag_24`
- `fiba_fibs_pressure_lag_24`
- `fiba_fibs_ratio_lag_168`
- `fiba_fibs_pressure_lag_168`
- `grf_tl_lag_1d`
- `grf_tl_change_7d`
- `grf_tl_rolling_mean_7d`
- `gas_cost_pressure_lag_1d`
- `thermal_cost_pressure_lag_1d`
- `gas_marginal_pressure_lag_1d`

## Target columns

- `target_1h`
- `target_2h`
- `target_3h`
- `target_4h`
- `target_5h`
- `target_6h`
- `target_7h`
- `target_8h`
- `target_9h`
- `target_10h`
- `target_11h`
- `target_12h`
- `target_13h`
- `target_14h`
- `target_15h`
- `target_16h`
- `target_17h`
- `target_18h`
- `target_19h`
- `target_20h`
- `target_21h`
- `target_22h`
- `target_23h`
- `target_24h`
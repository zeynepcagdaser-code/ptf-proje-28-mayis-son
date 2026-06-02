# Sequence Dataset Report

- **Generated (UTC):** 2026-06-02T06:57:23.133200+00:00
- **Source:** `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/features/lstm_next24_v1.parquet`
- **Output dir:** `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/model_low_price`

## Configuration

- Window: **168**
- Horizon: **24**
- Mapping: `X[t-167:t] -> y[t+1:t+24]`
- Features: 49
- Targets: 24
- Scaler fit split: **train**

## Tensor shapes

| Split | X shape | y shape |
|-------|---------|---------|
| train | (43513, 168, 49) | (43513, 24) |
| validation | (8570, 168, 49) | (8570, 24) |
| test | (3387, 168, 49) | (3387, 24) |

## Sequence counts

| Split | Sequences |
|-------|----------:|
| train | 43513 |
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

- `low_load_flag`
- `holiday_low_load_flag`
- `solar_peak_hour_flag`
- `zero_price_risk_proxy`
- `renewable_pressure`
- `renewable_suppression_pressure`
- `res_share`
- `solar_share`
- `hydro_share`
- `kgup_renewable_share`
- `kgup_gunes`
- `kgup_ruzgar`
- `wind_forecast_mean`
- `wind_forecast_share`
- `gas_share`
- `coal_share`
- `thermal_price_setting_share`
- `gas_coal_competition_index`
- `hour_sin`
- `hour_cos`
- `is_holiday_or_weekend`
- `is_weekend`
- `is_holiday_tr`
- `load_lep`
- `ptf_lag_1`
- `ptf_lag_2`
- `ptf_lag_3`
- `ptf_lag_24`
- `ptf_lag_168`
- `ptf_roll_mean_24`
- `ptf_roll_mean_168`
- `ptf_roll_min_24`
- `ptf_roll_max_24`
- `ptf_roll_min_168`
- `ptf_roll_max_168`
- `ptf_low_count_24`
- `ptf_zero_count_24`
- `ptf_low_count_168`
- `ptf_zero_count_168`
- `ptf_low_ratio_24`
- `ptf_zero_ratio_24`
- `ptf_low_ratio_168`
- `ptf_zero_ratio_168`
- `fiba_fibs_ratio`
- `fiba_fibs_pressure`
- `fiba_fibs_ratio_lag_24`
- `fiba_fibs_pressure_lag_24`
- `fiba_fibs_ratio_lag_168`
- `fiba_fibs_pressure_lag_168`

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
# Microstructure Feature Report

- **Generated (UTC):** 2026-06-01T18:20:32.688400+00:00
- **Input:** `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/features/lstm_next24_v1.parquet`
- **Master (for raw lags):** `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/master/master_hourly_v1.parquet`
- **Output:** `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/features/lstm_microstructure_next24_v1.parquet`
- **Rows:** 56016
- **Microstructure spec features:** 24
- **Newly added columns:** 22
- **Already in input (kept as-is):** ['smf_ptf_spread_lag_24', 'renewable_pressure']

## Leakage rules

- Realized SMF/PTF/YAL-YAT: only via `shift(1)` or longer lags
- No interpolation / bfill
- `target_*` and `split` preserved from input
- KGÜP/load/wind forecast ramps use plan values; realized balancing not at same hour

## Null % (new features)

| Feature | Null % |
|---------|-------:|
| `wind_forecast_ramp_24h` | 0.10 |
| `load_forecast_ramp_24h` | 0.09 |
| `wind_forecast_ramp_1h` | 0.05 |
| `load_forecast_ramp_1h` | 0.04 |
| `smf_ptf_spread_lag_1` | 0.01 |
| `smf_ptf_spread_change_3h` | 0.01 |
| `smf_ptf_spread_lag_24` | 0.00 |
| `smf_volatility_24h` | 0.00 |
| `ptf_volatility_24h` | 0.00 |
| `ptf_change_1h` | 0.00 |
| `ptf_change_3h` | 0.00 |
| `ptf_change_24h` | 0.00 |
| `kgup_total_ramp_1h` | 0.00 |
| `kgup_total_ramp_3h` | 0.00 |
| `kgup_total_ramp_24h` | 0.00 |
| `kgup_renewable_ramp_1h` | 0.00 |
| `kgup_thermal_ramp_1h` | 0.00 |
| `renewable_pressure` | 0.00 |
| `thermal_margin` | 0.00 |
| `yal_yat_net_pressure_lag_1` | 0.00 |
| `yal_yat_net_pressure_lag_24` | 0.00 |
| `yal_yat_transition_lag_1` | 0.00 |
| `yal_yat_abs_pressure_lag_1` | 0.00 |
| `gas_marginal_proxy` | 0.00 |

## Top volatility-related features (by std)

| Rank | Feature | Std | Mean |abs| |
|-----:|---------|----:|-----------:|
| 1 | `load_forecast_ramp_24h` | 3425.3943 | 2245.9254 |
| 2 | `kgup_total_ramp_3h` | 3376.0090 | 2652.8556 |
| 3 | `kgup_total_ramp_24h` | 2826.8548 | 1888.4582 |
| 4 | `wind_forecast_ramp_24h` | 1791.6826 | 1330.8083 |
| 5 | `load_forecast_ramp_1h` | 1560.6828 | 1112.5334 |
| 6 | `kgup_total_ramp_1h` | 1318.3811 | 1008.0347 |
| 7 | `yal_yat_transition_lag_1` | 1257.2187 | 904.0946 |
| 8 | `yal_yat_net_pressure_lag_1` | 1257.2187 | 904.0946 |
| 9 | `yal_yat_net_pressure_lag_24` | 1250.2116 | 902.7514 |
| 10 | `kgup_renewable_ramp_1h` | 1026.4312 | 724.5786 |

## Feature summary (new columns)

| Feature | Mean | Std | Min | Median | Max |
|---------|-----:|----:|----:|-------:|----:|
| `smf_ptf_spread_lag_1` | -21.37 | 539.58 | -4344.08 | 0.00 | 4300.01 |
| `smf_ptf_spread_lag_24` | -21.33 | 539.56 | -4344.08 | 0.00 | 4300.01 |
| `smf_ptf_spread_change_3h` | -0.00 | 596.91 | -7781.99 | 0.00 | 6460.00 |
| `smf_volatility_24h` | 496.91 | 417.57 | 0.00 | 431.98 | 2067.07 |
| `ptf_volatility_24h` | 391.57 | 333.65 | 0.00 | 343.37 | 2013.67 |
| `ptf_change_1h` | -0.00 | 357.39 | -3208.99 | 0.00 | 4620.01 |
| `ptf_change_3h` | -0.01 | 606.02 | -4300.01 | 0.00 | 4719.01 |
| `ptf_change_24h` | -0.10 | 550.08 | -4475.52 | 0.00 | 4470.01 |
| `kgup_total_ramp_1h` | -0.05 | 1318.38 | -5657.92 | -59.28 | 7023.69 |
| `kgup_total_ramp_3h` | -0.15 | 3376.01 | -9817.60 | -86.28 | 15008.12 |
| `kgup_total_ramp_24h` | -0.54 | 2826.85 | -14472.30 | -154.12 | 17921.32 |
| `kgup_renewable_ramp_1h` | 0.24 | 1026.43 | -4519.64 | -35.24 | 6189.63 |
| `kgup_thermal_ramp_1h` | -0.29 | 768.51 | -4523.71 | -8.53 | 5259.13 |
| `renewable_pressure` | 0.42 | 0.14 | 0.12 | 0.40 | 0.90 |
| `thermal_margin` | 0.58 | 0.14 | 0.10 | 0.60 | 0.88 |
| `wind_forecast_ramp_1h` | 0.04 | 189.49 | -1277.15 | -5.27 | 2211.12 |
| `wind_forecast_ramp_24h` | 0.62 | 1791.68 | -7258.53 | -59.33 | 7808.42 |
| `load_forecast_ramp_1h` | -0.05 | 1560.68 | -18487.00 | -219.00 | 20686.00 |
| `load_forecast_ramp_24h` | -3.75 | 3425.39 | -22319.00 | -176.00 | 27870.00 |
| `yal_yat_net_pressure_lag_1` | 480.00 | 1257.22 | -21192.64 | 363.26 | 10953.76 |
| `yal_yat_net_pressure_lag_24` | 482.26 | 1250.21 | -21192.64 | 364.19 | 10953.76 |
| `yal_yat_transition_lag_1` | 480.00 | 1257.22 | -21192.64 | 363.26 | 10953.76 |
| `yal_yat_abs_pressure_lag_1` | 904.09 | 996.79 | 0.00 | 647.74 | 21192.64 |
| `gas_marginal_proxy` | 0.22 | 0.11 | 0.00 | 0.23 | 0.60 |

## High |correlation| pairs (|r| > 0.85, new features only)

- `yal_yat_net_pressure_lag_1` ↔ `yal_yat_transition_lag_1`: 1.000
- `renewable_pressure` ↔ `thermal_margin`: -0.991
- `smf_volatility_24h` ↔ `ptf_volatility_24h`: 0.884
- `kgup_total_ramp_24h` ↔ `load_forecast_ramp_24h`: 0.863
- `thermal_margin` ↔ `gas_marginal_proxy`: 0.859
- `renewable_pressure` ↔ `gas_marginal_proxy`: -0.853

Correlation figure: `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/reports/figures/microstructure_correlation.png`
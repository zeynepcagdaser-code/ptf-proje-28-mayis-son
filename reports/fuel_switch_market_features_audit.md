# Fuel Switch / Marginality Feature Audit

Generated: `2026-05-31T21:54:10.119480+00:00`

This layer adds explicit gas marginality, hydro displacement, and demand weakness features to the regime feature store.

- Rows: `56232`
- Coverage: `2020-01-01T00:00:00` -> `2026-05-31T23:00:00`

## New Columns

- `gas_share_of_generation`
- `hydro_share_of_generation`
- `renewable_share_of_generation`
- `renewable_minus_gas_shift`
- `gas_marginality_proxy`
- `hydro_displacement_score`
- `cheap_supply_pressure`
- `low_demand_flag`
- `gas_off_flag`
- `renewable_share_high_flag`
- `hydro_high_flag`
- `zero_price_pressure_score`
- `load_deviation_from_weekly_norm`
- `load_deviation_from_monthly_norm`
- `demand_weakness_score`
- `load_vs_renewable_balance`

## Correlations With Price

| Feature | Corr(price) |
|---|---:|
| `gas_marginality_proxy_vs_price` | 0.1725 |
| `hydro_displacement_score_vs_price` | -0.2225 |
| `renewable_share_of_generation_vs_price` | -0.2031 |
| `gas_share_of_generation_vs_price` | 0.0675 |
| `zero_price_pressure_score_vs_price` | -0.2912 |
| `demand_weakness_score_vs_price` | -0.1600 |
| `load_vs_renewable_balance_vs_price` | 0.3560 |

## Regime Means

| Regime | Rows | Gas marginality | Hydro displacement | Renewable share | Gas share | Demand weakness | Zero-pressure score |
|---|---:|---:|---:|---:|---:|---:|---:|
| `negative_zero_pressure` | 951 | 16.67 | 62.09 | 0.765 | 0.027 | 9.23 | 83.13 |
| `normal` | 23699 | 35.57 | 43.62 | 0.429 | 0.233 | 4.49 | 36.08 |
| `tight` | 29878 | 35.79 | 42.08 | 0.410 | 0.215 | 3.64 | 27.58 |
| `spike_cap` | 1704 | 42.32 | 36.58 | 0.312 | 0.286 | 1.02 | 12.95 |

## Price Slices

| Slice | Rows | Gas marginality | Hydro displacement | Renewable share | Gas share | Zero-pressure score |
|---|---:|---:|---:|---:|---:|---:|
| `zero_price` | 951 | 16.67 | 62.09 | 0.765 | 0.027 | 83.13 |
| `low_price` | 24806 | 34.80 | 44.36 | 0.442 | 0.225 | 37.97 |
| `spike` | 1704 | 42.32 | 36.58 | 0.312 | 0.286 | 12.95 |

## Top Zero-Pressure Hours

- `2025-03-30T05:00:00` score=100.00 price=2249.99
- `2022-07-10T06:00:00` score=100.00 price=250.01
- `2025-03-30T07:00:00` score=100.00 price=1780.0
- `2025-03-30T06:00:00` score=100.00 price=2396.0
- `2022-07-10T07:00:00` score=99.96 price=550.0
- `2022-07-10T05:00:00` score=99.94 price=1800.0
- `2024-06-16T06:00:00` score=99.94 price=925.21
- `2024-06-16T04:00:00` score=99.92 price=1247.99
- `2024-06-16T05:00:00` score=99.92 price=1040.0
- `2025-03-30T08:00:00` score=99.91 price=1000.01

## Top Gas Marginality Hours

- `2021-10-14T03:00:00` score=57.82 price=558.57
- `2021-10-14T02:00:00` score=57.33 price=615.0
- `2021-10-14T05:00:00` score=56.97 price=606.99
- `2021-10-14T04:00:00` score=56.94 price=465.13
- `2021-10-14T06:00:00` score=56.89 price=649.93
- `2021-10-14T01:00:00` score=56.85 price=717.92
- `2021-10-12T03:00:00` score=56.53 price=549.99
- `2021-10-14T07:00:00` score=56.29 price=649.93
- `2021-10-12T06:00:00` score=56.27 price=595.0
- `2021-10-21T03:00:00` score=56.14 price=581.9

## Leakage Notes

- Uses same-hour planned/forecast load and generation shares only.
- Load deviation baselines are trailing rolling windows shifted by 24h.
- No realized future PTF is used in feature construction.

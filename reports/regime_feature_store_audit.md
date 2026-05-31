# Regime Feature Store Audit

Generated: `2026-05-31T13:15:44.049801+00:00`

No training is performed. This table contains anchor-time safe features only.

- Output: `data/features/regime_feature_store.parquet`
- Rows: `56232`
- Coverage: `2020-01-01T00:00:00` -> `2026-05-31T23:00:00`
- Safe K.PTF snapshot rows: `11`
- Forbidden columns present: `[]`

## Leakage Risk By Feature Family

| Feature family | Risk |
|---|---|
| `ptf_lag_1/24/168` | low: lagged finalized PTF only |
| `rolling_volatility` | low: shifted historical price volatility |
| `price_band_persistence` | low: lagged price bands only |
| `calendar` | low: deterministic calendar |
| `KGUP stack` | medium: assumes schedule version is available before anchor; revision timing should be snapshotted later |
| `load forecast` | medium: assumes latest forecast is available before anchor; revision timing should be snapshotted later |
| `renewable pressure` | medium: derived from KGUP renewable schedule, not realized generation |
| `maintenance/outage` | medium-high: publication/revision timing requires audit; uses active operatorPower proxy |
| `lagged SMF/YAL/YAT` | low-medium: lag24 only; same-hour realized values excluded |
| `snapshot KPTF` | low if snapshot_ts <= delivery_hour; sparse until archive matures |

## Highest Missing Rates

| Column | Missing rate |
|---|---:|
| `snapshot_marketTradePrice` | 1.000 |
| `snapshot_publish_state` | 1.000 |
| `snapshot_age_minutes` | 1.000 |
| `solar_cliff_score` | 0.507 |
| `volatility_cluster_score` | 0.009 |
| `ptf_lag_168` | 0.003 |
| `price_band_lag_168` | 0.003 |
| `price_band_persistence` | 0.003 |
| `smf_lag_24` | 0.001 |
| `smf_spread_lagged` | 0.001 |
| `load_ramp_3h` | 0.000 |
| `residual_load_ramp` | 0.000 |
| `load_ramp_1h` | 0.000 |
| `ptf_lag_24` | 0.000 |
| `price_band_lag_24` | 0.000 |
| `load_forecast` | 0.000 |
| `residual_load_forecast` | 0.000 |
| `load_minus_kgup` | 0.000 |
| `wind_relief_score` | 0.000 |
| `renewable_oversupply_score` | 0.000 |

## Explicitly Excluded

- historical raw interim_mcp oracle dataset
- same-hour finalized PTF
- same-hour realized SMF
- same-hour realized YAL/YAT
- same-hour realized generation

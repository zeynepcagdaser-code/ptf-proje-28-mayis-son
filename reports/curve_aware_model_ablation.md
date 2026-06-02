# Curve-Aware PTF Ablation

This is a limited historical smoke test using reconstructed real DAM supply-demand curve features as previous-day inputs for next-day PTF.

## Split

- Train: `2026-05-19 00:00:00` -> `2026-05-28 23:00:00` (`240` rows)
- Validation: `2026-05-29 00:00:00` -> `2026-05-29 23:00:00` (`24` rows)
- Test: `2026-05-30 00:00:00` -> `2026-06-01 23:00:00` (`72` rows)

## Test MAE

- Persistence: `415.20` TL/MWh
- Base market + fuel-switch: `465.65` TL/MWh
- Curve-aware: `465.45` TL/MWh
- Must-run proxy: `466.09` TL/MWh
- Curve + must-run: `466.12` TL/MWh
- Curve vs base delta: `-0.21` TL/MWh
- (Curve+MR) vs base delta: `0.47` TL/MWh

## Regime MAE

| Model | Regime | MAE |
|---|---:|---:|
| persistence | normal | 206.92 |
| persistence | negative_zero_pressure | 4.04 |
| persistence | spike_cap | 2358.25 |
| persistence | tight | 1521.98 |
| base_market_fuel_switch | normal | 285.20 |
| base_market_fuel_switch | negative_zero_pressure | 4.04 |
| base_market_fuel_switch | spike_cap | 2453.21 |
| base_market_fuel_switch | tight | 1638.56 |
| curve_aware | normal | 284.90 |
| curve_aware | negative_zero_pressure | 4.04 |
| curve_aware | spike_cap | 2452.84 |
| curve_aware | tight | 1637.16 |
| must_run_proxy | normal | 285.79 |
| must_run_proxy | negative_zero_pressure | 4.04 |
| must_run_proxy | spike_cap | 2454.47 |
| must_run_proxy | tight | 1638.09 |
| curve_plus_must_run | normal | 285.81 |
| curve_plus_must_run | negative_zero_pressure | 4.04 |
| curve_plus_must_run | spike_cap | 2454.65 |
| curve_plus_must_run | tight | 1638.08 |

## Top Curve-Aware Features

- `kgup_renewable_mw`: `571`
- `ptf_lag_24`: `503`
- `wind_share`: `395`
- `kgup_total`: `311`
- `coal_share`: `273`
- `gas_share`: `248`
- `zero_price_pressure_score`: `233`
- `reconstruction_confidence`: `223`
- `kgup_wind_mw`: `199`
- `weekday`: `185`
- `gas_marginality_proxy`: `72`
- `hydro_share`: `72`
- `analyst_persistence_break_score`: `66`
- `load_deviation_from_weekly_norm`: `64`
- `previous_day_regime_negative_zero_pressure`: `52`

## Caveat

This is a two-week curve-history smoke ablation, not a production-grade backtest. More historical DAM curve days are required before trusting small MAE differences.

A reliable answer needs more historical curve coverage. This run is useful mainly to verify the data plumbing and the direction of signal.

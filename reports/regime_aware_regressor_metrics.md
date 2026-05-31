# Regime-Aware Regressor (LightGBM)

Generated: `2026-05-31T14:23:14.801933+00:00`

## Setup

- Features: `must_run_proxy, load_forecast, lag24_ptf, hour_of_day`
- Regime split: `Normal` vs `Spike_Risk` (`tight` + `spike_cap`).
- Spike expert objective: custom directional residual loss (sign mismatch penalty).

## Must-Run Source

- Source mode: `fallback_proxy`
- Non-null must-run ratio: `0.0`

## Test Metrics

- MAE: `669.9987867439397`
- RMSE: `913.6688479154858`
- Persistence MAE: `535.3891997792495`
- Delta vs persistence: `134.60958696469027`
- Error <= 10 TL: `0.024282560706401765`
- Error <= 50 TL: `0.10126931567328919`

## Tomorrow Morning Forecast

- Rows produced: `0`
- File: `data/predictions/tomorrow_morning_ptf_forecast.csv`

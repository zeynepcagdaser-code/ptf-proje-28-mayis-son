# Migration Summary — rolling_ptf_forecast_system.py

Date: 2026-06-03

## 1. Timezone

- Added `assert_naive_ts()` helper; called in `build_hourly_market_table` before every merge. Raises `ValueError` if any source frame has tz-aware `ts_hour`.
- Fixed `grf_tl_lag_24` and `grf_tl_change_7d` being silently overwritten in `add_market_composites` even when already loaded from parquet. Added `if col not in out.columns` guards.
- Fixed `grf_tl_change_7d` formula: was `shift(24) - shift(24*8)`, now `current - shift(24*7)`.

## 2. Leakage

- Added `REALIZED_DELIVERY_COLS` constant listing all post-settlement columns that must not appear as delivery-hour features (`dam_matched_volume`, `smf`, `gen_*`, `yal_yat_net`, etc.).
- Removed `dam_matched_volume` from `orderbook_cols` (delivery features) in both `build_supervised_dataset` and `forecast_next24`. Added it to `add_history_features` lagged cols instead (`_lag_24`, `_lag_168`).
- Added assertion at end of `build_supervised_dataset`: raises `RuntimeError` if any `delivery_{realized_col}` slips into `feature_cols`.

## 3. Cross-Validation

- Added `walk_forward_cv()`: 3 expanding-window folds (2020-22→2023, 2020-23→2024, 2020-24→2025). Returns MAE, persistence MAE, improvement % per fold.
- `fit_profile()` now calls `walk_forward_cv` and stores results in `cv_folds`.
- `write_metrics_md()` writes a CV table below the main metrics table.

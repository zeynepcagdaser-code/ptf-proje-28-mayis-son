# Project Schema Audit

Generated for repository introspection and assistant coordination.

## Repository Status

- Git branch: `main`
- Remote: `origin`
- Remote URL: `https://github.com/zeynepcagdaser-code/ptf-proje-28-mayis-son.git`
- Local state: dirty working tree with active development artifacts

## Primary Storage Layers

### Raw market tables
- `data/ptf_dataset.csv`
- `data/kgup_combined.csv`
- `data/load_forecast.csv`
- `data/outages.csv`
- `data/realtime_generation.csv`
- `data/smf.csv`
- `data/yal_yat.csv`
- `data/real_consumption.csv`
- `data/wind_forecast.csv`
- `data/raw/interim_mcp.csv`

### Clean / hourly normalized
- `data/clean/ptf_hourly.parquet`
- `data/clean/kgup_hourly.parquet`
- `data/clean/load_forecast_hourly.parquet`
- `data/clean/outages_hourly.parquet`
- `data/clean/realtime_generation_hourly.parquet`
- `data/clean/smf_hourly.parquet`
- `data/clean/yal_yat_hourly.parquet`
- `data/clean/real_consumption_hourly.parquet`
- `data/clean/wind_hourly.parquet`

### Feature tables
- `data/features/regime_feature_store.parquet`
- `data/features/market_reasoning_features.parquet`
- `data/features/must_run_supply_features.parquet`
- `data/features/must_run_proxy_v2.parquet`
- `data/features/tomorrow_morning_features.parquet`
- `data/features/tomorrow_morning_features_enriched.parquet`
- `data/features/supply_demand_curve_features.parquet`
- `data/features/real_supply_demand_curve_features.parquet`
- `data/features/reconstructed_market_curve_features.parquet`
- `data/features/reconstructed_daily_curve_features_2026-06-01.parquet`
- `data/features/reconstructed_weekly_curve_features_2026-06-01_2026-06-07.parquet`
- `data/features/curve_aware_training_dataset.parquet`

## Main Timestamp Conventions

- Raw APIs commonly expose ISO timestamps with `+03:00`.
- Feature stores often normalize to `ts_hour` or `delivery_hour`.
- Some scripts use naive datetimes for joining and reporting.

### Safe standard
- Use `Europe/Istanbul` / `+03:00` as the source-time standard.
- Convert to naive only at the feature-store layer when the script explicitly documents it.

## Target / Label Schema

### PTF regression
- `price`
- `target_ptf`

### Regime labels
- `target_regime`
- `lag24_regime`
- `transition_label`
- `persistence_error`

### Spike / transition detectors
- `is_spike_cap`
- `is_new_spike_transition`

## Leakage-Sensitive Fields

The following columns must be treated as target-only or explicit audit-only unless a script documents otherwise:

- `price`
- `target_ptf`
- `target_regime`
- `transition_label`
- `persistence_error`
- `mcpPrice`
- `matchingQuantity`
- `marketTradePrice`
- `snapshot_marketTradePrice`
- any same-hour finalized output

## Known Schema Risks

### 1. Timestamp mismatch
- `date`/`hour` parsing can produce different shapes across files.
- Merge operations fail silently when one side is timezone-aware and the other is naive.

### 2. Coverage drift
- Curve-derived features may exist for dates later than finalized PTF coverage.
- This creates empty training sets even when the feature pipeline itself is healthy.

### 3. Feature naming drift
- Multiple naming variants are used for the same concept:
  - `price` / `mcpPrice` / `marketTradePrice`
  - `load_forecast` / `lep`
  - `kgup_total` / `toplam`
  - `hour` / `delivery_hour` / `ts_hour`

### 4. Parquet append safety
- Incremental scripts need atomic writes and state files.
- Partial writes can corrupt coverage if not guarded.

### 5. Historical interim semantics
- Historical interim MCP data currently behaves like a retrospective canonical value table, not a trustworthy point-in-time archive.

## Data Lineage Table

| Source | Transformation | Output |
|---|---|---|
| EPİAŞ finalized MCP endpoint | `update_dataset.py` fetch + dedupe + rolling refresh | `data/ptf_dataset.csv` |
| EPİAŞ interim MCP endpoint | `fetch_interim_mcp.py` snapshot/archive | `data/raw/interim_mcp.csv` |
| EPİAŞ KGÜP endpoints | `fetch_kgup_combined.py` + hourly cleanup | `data/kgup_combined.csv` and `data/clean/kgup_hourly.parquet` |
| EPİAŞ load forecast endpoint | `fetch_load_forecast.py` | `data/load_forecast.csv` and `data/clean/load_forecast_hourly.parquet` |
| EPİAŞ outage / maintenance endpoints | `fetch_outages.py` | `data/outages.csv` and `data/clean/outages_hourly.parquet` |
| EPİAŞ realtime generation endpoint | `fetch_realtime_generation.py` | `data/realtime_generation.csv` and `data/clean/realtime_generation_hourly.parquet` |
| EPİAŞ SMF endpoint | `fetch_smf.py` | `data/smf.csv` and `data/clean/smf_hourly.parquet` |
| EPİAŞ YAL/YAT endpoint | `fetch_yal_yat.py` | `data/yal_yat.csv` and `data/clean/yal_yat_hourly.parquet` |
| EPİAŞ real consumption endpoint | `fetch_real_consumption.py` | `data/real_consumption.csv` and `data/clean/real_consumption_hourly.parquet` |
| EPİAŞ wind forecast endpoint | `fetch_wind_forecast.py` | `data/wind_forecast.csv` and `data/clean/wind_hourly.parquet` |
| Plant-level KGÜP smoke/raw | `build_plant_level_kgup_smoke.py` | `data/plant_level_kgup/raw_smoke/` |
| Plant-level KGÜP normalization | `build_plant_level_kgup_pipeline.py` | `data/plant_level_kgup/normalized_plant_level_kgup.parquet` |
| YEKDEM join | `build_plant_level_kgup_pipeline.py` | `data/plant_level_kgup/yekdem_matched_plant_level_kgup.parquet` |
| Must-run proxy | `build_must_run_proxy_v2.py` | `data/features/must_run_proxy_v2.parquet` |
| Regime feature store | `build_regime_feature_store.py` | `data/features/regime_feature_store.parquet` |
| Analyst reasoning layer | `market_reasoning_engine.py` | `data/features/market_reasoning_features.parquet` |
| Supply-demand curve proxy features | `build_supply_demand_curve_features.py` | `data/features/supply_demand_curve_features.parquet` |
| Raw curve reconstruction features | `build_real_supply_demand_curve_engine.py` / `reconstruct_real_market_curve.py` | `data/features/real_supply_demand_curve_features.parquet`, `data/features/reconstructed_market_curve_features.parquet` |
| Daily reconstruction | `fetch_and_reconstruct_daily_dam_curves.py` | `data/features/reconstructed_daily_curve_features_2026-06-01.parquet` |
| Weekly reconstruction | `fetch_and_reconstruct_weekly_dam_curves.py` | `data/features/reconstructed_weekly_curve_features_2026-06-01_2026-06-07.parquet` |
| Curve-aware training dataset | `build_curve_aware_training_dataset.py` | `data/features/curve_aware_training_dataset.parquet` |

## Dashboard Entry Points

- `src/dashboard/streamlit_regime_dashboard.py`
- Run command: `streamlit run src/dashboard/streamlit_regime_dashboard.py`

## Training / Inference Scripts

- `train_regime_classifier.py`
- `spike_cap_risk_detector.py`
- `train_spike_transition_detector.py`
- `train_regime_aware_regressor.py`
- `train_high_precision_ptf_model.py`
- `run_tomorrow_forecast.py`
- `train_lstm.py`
- `train_lstm_residual.py`
- `train_tree_horizon.py`
- `train_tree_advanced.py`
- `train_short_horizon_expert.py`

## Curve Reconstruction Flow

1. Smoke probe the DAM supply-demand curve endpoint.
2. Parse raw supply and PTF response bodies.
3. Rebuild supply/demand curves on the same price axis.
4. Estimate clearing intersection and compare against EPİAŞ MCP.
5. Persist raw bodies, debug plots, and feature parquet outputs.

## Repo-specific Warnings

- The repo is currently active and dirty; many generated artifacts are present.
- Some outputs are historical and may not be reproducible against today’s EPİAŞ state.
- The assistant should not assume that a dataset suffix implies freshness.
- When a dataset looks empty, first check coverage boundaries before debugging feature code.

## Recommended Prompt Discipline

When asking for changes in this repo, include:

- source file(s)
- desired output file(s)
- target date range
- whether same-day realized data is allowed
- whether a `ts_hour` or `delivery_hour` join is expected
- whether timezone normalization is needed

This should reduce schema drift in future assistant passes.

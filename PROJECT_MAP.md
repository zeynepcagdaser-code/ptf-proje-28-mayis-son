# PROJECT MAP

Repository: `zeynepcagdaser-code/ptf-proje-28-mayis-son`

Remote:
- `origin` -> `https://github.com/zeynepcagdaser-code/ptf-proje-28-mayis-son.git`

Branch:
- `main`

This repo is already pushed to GitHub and tracked by `origin/main`.

## Top-Level Layout

```text
.
├── .github/workflows/
├── build_*.py
├── fetch_*.py
├── train_*.py
├── evaluate_*.py
├── audit_*.py
├── cleaning/
├── features/
├── master/
├── models/
├── data/
├── reports/
└── src/dashboard/
```

## Core Dataset Flow

### Finalized PTF / MCP
- Source: `update_dataset.py`
- Output: `data/ptf_dataset.csv`
- Target columns:
  - `price`
  - `priceUsd`
  - `priceEur`
- Time standard:
  - `date` values are `+03:00` aware ISO timestamps
  - downstream feature builders often convert to naive `ts_hour` for merges

### Interim K.PTF / MCP
- Source: `fetch_interim_mcp.py`
- Outputs:
  - `data/raw/interim_mcp.csv`
  - `data/raw/interim_mcp_fetch_state.json`
- Used for snapshot research only, not as a leakage-safe historical truth table.

### KGÜP
- Source: `fetch_kgup_combined.py`
- Output: `data/kgup_combined.csv`
- Downstream cleaned output:
  - `data/clean/kgup_hourly.parquet`

### Load Forecast
- Source: `fetch_load_forecast.py`
- Output: `data/load_forecast.csv`
- Cleaned:
  - `data/clean/load_forecast_hourly.parquet`

### Outages
- Source: `fetch_outages.py`
- Output: `data/outages.csv`
- Cleaned:
  - `data/clean/outages_hourly.parquet`

### Real-time Generation
- Source: `fetch_realtime_generation.py`
- Output: `data/realtime_generation.csv`
- Cleaned:
  - `data/clean/realtime_generation_hourly.parquet`

### SMF
- Source: `fetch_smf.py`
- Output: `data/smf.csv`
- Cleaned:
  - `data/clean/smf_hourly.parquet`

### YAL / YAT
- Source: `fetch_yal_yat.py`
- Output: `data/yal_yat.csv`
- Cleaned:
  - `data/clean/yal_yat_hourly.parquet`

### Real Consumption
- Source: `fetch_real_consumption.py`
- Output: `data/real_consumption.csv`
- Cleaned:
  - `data/clean/real_consumption_hourly.parquet`

### Wind Forecast
- Source: `fetch_wind_forecast.py`
- Output: `data/wind_forecast.csv`
- Cleaned:
  - `data/clean/wind_hourly.parquet`

### Plant-level KGÜP / YEKDEM
- Smoke source: `build_plant_level_kgup_smoke.py`
- Pipeline: `build_plant_level_kgup_pipeline.py`
- Outputs:
  - `data/plant_level_kgup/raw_smoke/`
  - `data/plant_level_kgup/normalized_plant_level_kgup.parquet`
  - `data/plant_level_kgup/yekdem_matched_plant_level_kgup.parquet`
  - `data/features/must_run_supply_features.parquet`

## Feature Tables

- `data/features/regime_feature_store.parquet`
- `data/features/market_reasoning_features.parquet`
- `data/features/must_run_proxy_v2.parquet`
- `data/features/tomorrow_morning_features.parquet`
- `data/features/tomorrow_morning_features_enriched.parquet`
- `data/features/supply_demand_curve_features.parquet`
- `data/features/real_supply_demand_curve_features.parquet`
- `data/features/reconstructed_market_curve_features.parquet`
- `data/features/reconstructed_daily_curve_features_2026-06-01.parquet`
- `data/features/reconstructed_weekly_curve_features_2026-06-01_2026-06-07.parquet`
- `data/features/curve_aware_training_dataset.parquet`

## Model Artefacts

- `models/regime_classifier/`
- `models/spike_cap_detector/`
- `models/spike_transition_detector/`
- `models/regime_aware_regressor/`
- `models/high_precision_ptf_model/`
- `models/lstm_baseline.pt`
- `models/lstm_residual.pt`
- `models/tree_advanced/`
- `models/microstructure_h1h4/`
- `models/microstructure_h5h12/`

## Dashboard Entry Point

- `src/dashboard/streamlit_regime_dashboard.py`
- Run:
  - `streamlit run src/dashboard/streamlit_regime_dashboard.py`

## Training / Inference Scripts

### Regime / Spike
- `build_regime_labels.py`
- `build_regime_feature_store.py`
- `market_reasoning_engine.py`
- `train_regime_classifier.py`
- `spike_cap_risk_detector.py`
- `train_spike_transition_detector.py`

### Price Forecasting
- `train_high_precision_ptf_model.py`
- `train_regime_aware_regressor.py`
- `run_tomorrow_forecast.py`

### Curve Intelligence
- `fetch_dam_supply_demand_curve.py`
- `reconstruct_real_market_curve.py`
- `fetch_and_reconstruct_daily_dam_curves.py`
- `fetch_and_reconstruct_weekly_dam_curves.py`
- `build_supply_demand_curve_features.py`
- `build_real_supply_demand_curve_engine.py`
- `build_curve_aware_training_dataset.py`

### Persistence / Diagnostics
- `audit_persistence_failures.py`
- `audit_interim_dataset.py`
- `audit_unfinalized_ptf_data.py`
- `audit_final_h1h4_pipeline.py`

## Which Script Produces Which Dataset

| Script | Main Output |
|---|---|
| `update_dataset.py` | `data/ptf_dataset.csv` |
| `fetch_kgup_combined.py` | `data/kgup_combined.csv` |
| `fetch_load_forecast.py` | `data/load_forecast.csv` |
| `fetch_outages.py` | `data/outages.csv` |
| `fetch_realtime_generation.py` | `data/realtime_generation.csv` |
| `fetch_smf.py` | `data/smf.csv` |
| `fetch_yal_yat.py` | `data/yal_yat.csv` |
| `fetch_real_consumption.py` | `data/real_consumption.csv` |
| `fetch_wind_forecast.py` | `data/wind_forecast.csv` |
| `build_regime_feature_store.py` | `data/features/regime_feature_store.parquet` |
| `market_reasoning_engine.py` | `data/features/market_reasoning_features.parquet` |
| `build_must_run_proxy_v2.py` | `data/features/must_run_proxy_v2.parquet` |
| `build_plant_level_kgup_pipeline.py` | `data/features/must_run_supply_features.parquet` |
| `fetch_dam_supply_demand_curve.py` | smoke raw curve files under `data/raw/dam_supply_demand_curve_smoke/` |
| `reconstruct_real_market_curve.py` | `data/features/reconstructed_market_curve_features.parquet` |
| `fetch_and_reconstruct_daily_dam_curves.py` | `data/features/reconstructed_daily_curve_features_2026-06-01.parquet` |
| `fetch_and_reconstruct_weekly_dam_curves.py` | `data/features/reconstructed_weekly_curve_features_2026-06-01_2026-06-07.parquet` |
| `build_curve_aware_training_dataset.py` | `data/features/curve_aware_training_dataset.parquet` |

## Timezone Standard

- Raw market CSVs often store delivery timestamps as ISO strings with `+03:00`.
- Feature builders commonly normalize to naive hourly timestamps named `ts_hour` or `delivery_hour`.
- For merges, prefer explicit conversion and do not assume naive timestamps equal UTC.
- The safe operating assumption in this repo is `Europe/Istanbul` / `+03:00` unless a file explicitly documents otherwise.

## Target Columns

### Finalized PTF
- `price`

### Curve Reconstruction
- `mcpPrice`
- `matchingQuantity`
- `reconstructed_clearing_price`
- `reconstructed_clearing_volume`

### Regime / Research Labels
- `target_regime`
- `transition_label`
- `persistence_error`
- `is_spike_cap`
- `is_new_spike_transition`

### Curve-Aware Dataset
- `target_ptf`

## Leakage-Sensitive Columns

Do not use as features unless a script explicitly marks them as safe:

- `price`
- `target_ptf`
- `target_regime`
- `transition_label`
- `persistence_error`
- `mcpPrice`
- `matchingQuantity`
- `marketTradePrice`
- `snapshot_marketTradePrice`
- any same-hour realized SMF / YAL / YAT / finalized PTF
- any historical interim series that behaves like an oracle

## Curve Reconstruction Pipeline

1. `fetch_dam_supply_demand_curve.py`
   - smoke tests one delivery hour.
   - stores raw supply and PTF bodies.

2. `reconstruct_real_market_curve.py`
   - reconstructs supply and demand curves from raw response body.
   - finds clearing intersection.
   - compares reconstructed clearing price with `mcpPrice`.

3. `fetch_and_reconstruct_daily_dam_curves.py`
   - fetches and reconstructs a full day.

4. `fetch_and_reconstruct_weekly_dam_curves.py`
   - fetches a week with resume support and state tracking.

5. `build_supply_demand_curve_features.py`
   - builds proxy microstructure features.

6. `build_real_supply_demand_curve_engine.py`
   - builds raw-curve feature extraction from the real supply-demand table.

## Known Schema Risks

### Timestamp mismatch
- Some CSVs store `date` as ISO strings with `+03:00`.
- Some feature tables use naive `ts_hour`.
- Mixed merge keys can silently empty joins if not normalized first.

### tz-aware vs naive
- `tz-aware` and `naive` datetimes must be normalized before joins.
- This repo has both styles in different layers.

### Missing future targets
- Many future-facing feature tables exist before the corresponding finalized target arrives.
- This is expected for snapshot / inference work, but it makes training datasets empty until the target coverage catches up.

### Feature naming drift
- `price` vs `mcpPrice` vs `marketTradePrice` appear in different layers.
- `load_forecast` vs `lep` and `kgup_total` vs `toplam` are common aliases.

### Parquet append risks
- Several scripts now append or rebuild parquet outputs incrementally.
- Use atomic writes and state files.
- Never assume a partial parquet write is valid after an interrupted run.

### Coverage drift
- A feature table can be structurally valid but still have zero target coverage.
- Always validate `max_timestamp`, duplicate hours, and missing-hour coverage before training.

## Notes for Future Prompts

- If the task involves a dataset, always check whether the target timestamp exists in the source coverage first.
- If a merge is empty, inspect timezone normalization before debugging the model.
- If a feature is “missing”, check if it is actually a future-only field rather than a parse bug.
- Prefer `ts_hour` / `delivery_hour` as normalized anchors inside the repo.

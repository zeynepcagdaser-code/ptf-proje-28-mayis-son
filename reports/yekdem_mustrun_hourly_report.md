# YEKDEM Must-Run Hourly Proxy

- Rows: `24`
- Coverage: `2026-05-31 00:00:00` → `2026-05-31 23:00:00`
- Rows with finalized PTF: `24`
- Rows with regime label: `24`

## Interpretation

This table is a structural must-run proxy, not a strict plant-level YEKDEM archive.
It is built from `must_run_proxy_v2` because the repository currently lacks a dense plant-level KGUP archive with publication timestamps.

## Leakage Policy

- `structural_proxy_only`
- Missing real plant-level reason: `No dense plant-level KGUP archive with publication timestamps exists in the repository yet.`

## Useful Columns

- `must_run_supply`
- `must_run_wind`
- `must_run_solar`
- `must_run_hydro`
- `must_run_biomass`
- `must_run_geothermal`
- `must_run_share`
- `residual_load_after_must_run`
- `renewable_concentration_score`
- `solar_oversupply_score`
- `renewable_curtailment_pressure_proxy`


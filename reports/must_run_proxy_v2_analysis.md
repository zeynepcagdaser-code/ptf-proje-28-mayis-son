# Must Run Proxy V2 Analysis

Generated: `2026-05-31T14:45:55.313435+00:00`

- Rows analyzed: `24`
- Coverage: `2026-05-31 00:00:00` → `2026-05-31 23:00:00`

## Correlations

- must_run_supply_proxy vs price: `-0.388131415768074`
- renewable_concentration_score vs price: `0.4239693259608822`
- solar_oversupply_score vs price midday: `nan`
- renewable_curtailment_pressure_proxy vs price: `-0.5229621877525723`

## Regime Summary

| target_regime | rows | must_run_supply_proxy_mean | must_run_supply_proxy_median | renewable_concentration_score_mean | solar_oversupply_score_mean | hydro_pressure_score_mean | renewable_curtailment_pressure_proxy_mean | price_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| negative_zero_pressure | 12 | 1.9625000000000001 | 1.75 | 0.4134363585017397 | 0.0 | 0.0 | 1.774904770037263e-05 | 4.083333333333333 |
| normal | 8 | 0.41000000000000003 | 0.395 | 0.927734375 | 0.0 | 0.0 | 1.1354851002289247e-05 | 412.25125 |
| spike_cap | 3 | 0.29000000000000004 | 0.27 | 1.0 | 0.0 | 0.0 | 7.65249367696293e-06 | 4300.003333333333 |
| tight | 1 | 0.32 | 0.32 | 1.0 | 0.0 | 0.0 | 9.119927040583676e-06 | 1849.99 |

## Slices

{
  "low_price_hours": {
    "rows": 12,
    "must_run_supply_proxy_mean": 1.9625000000000001,
    "renewable_concentration_score_mean": 0.4134363585017397
  },
  "zero_pressure_hours": {
    "rows": 12,
    "must_run_supply_proxy_mean": 1.9625000000000001,
    "renewable_concentration_score_mean": 0.4134363585017397
  },
  "midday_hours": {
    "rows": 6,
    "solar_oversupply_score_mean": 0.0,
    "renewable_curtailment_pressure_proxy_mean": 2.2720725608384602e-05
  }
}

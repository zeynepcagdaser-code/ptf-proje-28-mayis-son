# Market Reasoning Engine Design

Generated: `2026-05-31T21:54:10.383378+00:00`

No training is performed. This is a deterministic analyst-reasoning layer built on the leakage-safe feature store.

- Input: `data/features/regime_feature_store.parquet`
- Output: `data/features/market_reasoning_features.parquet`
- Rows: `56232`

## Produced Columns

- `ts_hour`
- `analyst_zero_score`
- `analyst_spike_score`
- `analyst_tight_score`
- `analyst_persistence_break_score`
- `analyst_expected_regime`
- `analyst_confidence_score`
- `analyst_reason_text`

## Score Ranges

| Score | Min | Mean | Max |
|---|---:|---:|---:|
| `analyst_zero_score` | 0.00 | 40.95 | 100.00 |
| `analyst_spike_score` | 0.00 | 40.07 | 94.40 |
| `analyst_tight_score` | 4.70 | 47.60 | 89.72 |
| `analyst_persistence_break_score` | 2.41 | 35.72 | 94.90 |
| `analyst_confidence_score` | 46.62 | 66.21 | 95.82 |

## Expected Regime Counts

| Regime | Rows |
|---|---:|
| `tight` | 27033 |
| `negative_zero_pressure` | 21252 |
| `normal` | 4107 |
| `spike_cap` | 3840 |

## Reasoning Logic

- `analyst_zero_score`: renewable oversupply, low residual load, low load-KGÜP gap, wind relief.
- `analyst_spike_score`: high residual load, residual ramp, solar cliff, maintenance stress, gas dependency, low wind relief, evening ramp.
- `analyst_tight_score`: residual load, load-KGÜP gap, thermal/gas/hydro dependency, maintenance and volatility.
- `analyst_persistence_break_score`: lagged band disagreement, residual ramp, solar cliff, volatility and maintenance.

## Leakage Policy

- Uses only `regime_feature_store.parquet`.
- Does not use finalized target PTF.
- Does not use same-hour realized SMF/YAL/YAT.
- Does not use historical oracle `interim-mcp`.

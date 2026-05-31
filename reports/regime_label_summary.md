# Regime Label Summary

Generated: `2026-05-31T12:47:32.330476+00:00`

Finalized PTF is used here only to create labels and evaluation fields.
These target columns must not enter the forecasting feature matrix.

- Rows: `56232`
- Valid lag24 rows: `56208`
- Coverage: `2020-01-01T00:00:00` -> `2026-05-31T23:00:00`
- Persistence MAE, lag24: `294.87`
- Persistence median absolute error, lag24: `100.00`

## Regime Counts

| Regime | Rows |
|---|---:|
| `tight` | 29878 |
| `normal` | 23699 |
| `spike_cap` | 1704 |
| `negative_zero_pressure` | 951 |

## Top Transition Counts

| Transition | Rows |
|---|---:|
| `tight -> tight` | 26775 |
| `normal -> normal` | 20685 |
| `tight -> normal` | 2565 |
| `normal -> tight` | 2486 |
| `spike_cap -> spike_cap` | 1166 |
| `spike_cap -> tight` | 511 |
| `tight -> spike_cap` | 500 |
| `normal -> negative_zero_pressure` | 482 |
| `negative_zero_pressure -> negative_zero_pressure` | 432 |
| `negative_zero_pressure -> normal` | 401 |
| `negative_zero_pressure -> tight` | 106 |
| `normal -> spike_cap` | 38 |
| `tight -> negative_zero_pressure` | 37 |
| `spike_cap -> normal` | 24 |

## Leakage Policy

- `price`, `target_regime`, `transition_label`, and `persistence_error` are target/evaluation columns.
- Downstream feature stores must use only anchor-time safe inputs.
- Historical `interim-mcp` oracle data must not be used as a feature.

# High Precision PTF Model Metrics

Generated: `2026-05-31T21:21:18.402046+00:00`

This is a real final PTF price forecast prototype using `ptf_lag_24` as a persistence anchor and leakage-guarded residual experts.

## Selected Validation Configuration

- Selected prediction config: `global_residual_pred`
- Selected cap floor strength: `0.0`
- Selection criterion: validation MAE only.
- Important caveat: this is delivery-hour evaluation, not the older anchor-based h1-h4 format.

## Test Summary

| Model | MAE | RMSE | Median AE | P90 AE | <=2 TL | <=10 TL | <=50 TL |
|---|---:|---:|---:|---:|---:|---:|---:|
| `high_precision_pred` | 531.45 | 832.19 | 258.86 | 1453.00 | 0.0036 | 0.0439 | 0.2348 |
| `persistence_pred` | 535.39 | 839.79 | 257.50 | 1468.73 | 0.1393 | 0.1609 | 0.2315 |
| `global_residual_pred` | 531.45 | 832.19 | 258.86 | 1453.00 | 0.0036 | 0.0439 | 0.2348 |
| `regime_soft_pred_no_floor` | 530.84 | 802.21 | 276.81 | 1358.32 | 0.0905 | 0.0988 | 0.1471 |
| `regime_classifier_routing_only_pred` | 538.24 | 803.90 | 300.01 | 1354.63 | 0.0927 | 0.1021 | 0.1366 |

## Comparison

- Delta vs persistence MAE: `-3.94` TL/MWh.
- Delta vs global residual MAE: `0.00` TL/MWh.
- Delta vs regime classifier routing only MAE: `-6.79` TL/MWh.
- Previous best h1-h4 ensemble: `443.87` TL/MWh mean MAE from `reports/final_h1h4_summary.md`; not directly comparable because that benchmark is anchor/horizon based.

## Regime-Wise Test MAE

| Regime | Rows | Model MAE | Persistence MAE | Delta |
|---|---:|---:|---:|---:|
| `negative_zero_pressure` | 387 | 154.31 | 139.01 | 15.30 |
| `normal` | 1395 | 496.16 | 501.00 | -4.84 |
| `spike_cap` | 73 | 1167.07 | 1183.67 | -16.60 |
| `tight` | 1769 | 615.56 | 622.47 | -6.91 |

## Stress Slices

- Persistence failure rows: `1673`; model MAE `1032.75` TL/MWh.
- Delivery hour 1-4 MAE: `425.25` TL/MWh.
- Spike/cap rows: `73`; cap miss rate pred<4000: `0.5068`; mean shortfall to 4000: `939.78`.

## Worst Transition MAE

| Transition | Rows | Model MAE | Persistence MAE | Delta |
|---|---:|---:|---:|---:|
| `normal -> spike_cap` | 12 | 3511.30 | 3553.40 | -42.11 |
| `spike_cap -> normal` | 5 | 3033.68 | 3023.38 | 10.30 |
| `tight -> negative_zero_pressure` | 6 | 1992.93 | 1984.23 | 8.70 |
| `negative_zero_pressure -> tight` | 5 | 1722.83 | 1764.35 | -41.52 |
| `tight -> spike_cap` | 25 | 1554.63 | 1585.63 | -31.00 |
| `spike_cap -> tight` | 29 | 1491.75 | 1480.62 | 11.13 |
| `normal -> tight` | 212 | 1430.27 | 1467.57 | -37.30 |
| `tight -> normal` | 238 | 1375.98 | 1365.56 | 10.42 |
| `tight -> tight` | 1523 | 481.83 | 484.74 | -2.91 |
| `normal -> normal` | 1014 | 307.92 | 312.42 | -4.50 |
| `normal -> negative_zero_pressure` | 149 | 290.17 | 277.86 | 12.31 |
| `negative_zero_pressure -> normal` | 138 | 270.03 | 304.24 | -34.21 |
| `spike_cap -> spike_cap` | 36 | 116.53 | 114.62 | 1.91 |
| `negative_zero_pressure -> negative_zero_pressure` | 232 | 19.49 | 2.11 | 17.38 |

## Critical Evaluation

The prototype does not reach the 10-50 TL band globally; test MAE is 531.45 TL/MWh.
The 1-2 TL objective is not achieved: only 0.36% of test hours are within 2 TL.
Main hard slice remains spike/cap and large regime transitions, where hidden bid-curve and participant strategy information is still missing.

## Leakage Checks

- **Forbidden feature columns absent**: `pass` - Forbidden columns present in model feature matrix: []
- **price target only**: `pass` - price is used for residual target and evaluation, not as a feature.
- **regime labels excluded from features**: `pass` - target_regime, lag24_regime, transition_label, and persistence_error are dropped from X.
- **same-hour realized balancing excluded**: `pass` - Only lagged SMF/YAL/YAT columns from the feature store are used.
- **historical interim oracle excluded**: `pass` - No historical interim_mcp source is read; sparse point-in-time snapshot fields remain optional.

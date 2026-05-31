# Fuel Switch Routed PTF Model Metrics

Generated: `2026-05-31T22:07:36.099858+00:00`

This model explicitly routes price residual experts through zero-pressure and gas-marginality state probabilities.

## Test Summary

| Model | MAE | RMSE | Median AE | P90 AE | <=2 TL | <=10 TL | <=50 TL |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fuel_switch_pred` | 510.56 | 783.62 | 283.44 | 1314.91 | 0.0682 | 0.0842 | 0.1551 |
| `persistence_pred` | 535.39 | 839.79 | 257.50 | 1468.73 | 0.1393 | 0.1609 | 0.2315 |

- Delta vs persistence: `-24.83` TL/MWh

## Regime-Wise Test MAE

| Regime | Rows | Model MAE | Persistence MAE | Delta |
|---|---:|---:|---:|---:|
| `negative_zero_pressure` | 387 | 80.66 | 139.01 | -58.35 |
| `normal` | 1395 | 412.28 | 501.00 | -88.73 |
| `spike_cap` | 73 | 1790.76 | 1183.67 | 607.09 |
| `tight` | 1769 | 629.29 | 622.47 | 6.82 |

## State Classifiers

| State | Positives | PR-AUC | ROC-AUC | Balanced acc @0.5 |
|---|---:|---:|---:|---:|
| `zero_pressure_state` | 387 | 0.5002 | 0.9289 | 0.5510 |
| `low_price_state` | 1798 | 0.9597 | 0.9640 | 0.9052 |
| `high_price_state` | 1842 | 0.9697 | 0.9646 | 0.9055 |
| `spike_state` | 73 | 0.1779 | 0.8887 | 0.5200 |

## Selected Validation Parameters

```json
{
  "zero_scale": 1.0,
  "low_scale": 0.8,
  "gas_scale": 1.0,
  "spike_scale": 0.7,
  "global_scale": 0.7,
  "zero_pull_strength": 0.35,
  "zero_pull_power": 1.0,
  "zero_anchor_price": 80.0,
  "cap_reference": 4000.0,
  "cap_floor_strength": 0.0
}
```

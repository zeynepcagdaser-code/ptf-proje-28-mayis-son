# Hybrid Fuel Switch PTF Metrics

Generated: `2026-05-31T22:12:15.221158+00:00`

This hybrid uses the fuel-switch model when the zero/cheap-supply mechanism is active and falls back to the high precision model when spike/gas risk is present.

## Test Summary

| Model | MAE | RMSE | Median AE | P90 AE | <=2 TL | <=10 TL | <=50 TL |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hybrid_pred` | 494.37 | 771.72 | 258.78 | 1306.25 | 0.0651 | 0.0792 | 0.1937 |
| `high_precision_pred` | 531.45 | 832.19 | 258.86 | 1453.00 | 0.0036 | 0.0439 | 0.2348 |
| `fuel_switch_pred` | 510.56 | 783.62 | 283.44 | 1314.91 | 0.0682 | 0.0842 | 0.1551 |
| `persistence_pred` | 535.39 | 839.79 | 257.50 | 1468.73 | 0.1393 | 0.1609 | 0.2315 |

- Delta vs high precision: `-37.08` TL/MWh
- Delta vs persistence: `-41.02` TL/MWh
- Fuel-switch usage rate on test: `0.6705`

## Regime-Wise Test MAE

| Regime | Rows | Hybrid MAE | Persistence MAE | High precision MAE | Fuel-switch MAE |
|---|---:|---:|---:|---:|---:|
| `negative_zero_pressure` | 387 | 80.66 | 139.01 | 154.31 | 80.66 |
| `normal` | 1395 | 419.46 | 501.00 | 496.16 | 412.28 |
| `spike_cap` | 73 | 1174.49 | 1183.67 | 1167.07 | 1790.76 |
| `tight` | 1769 | 615.88 | 622.47 | 615.56 | 629.29 |

## Selected Parameters

```json
{
  "zero_threshold": 0.55,
  "cheap_threshold": 0.45,
  "fuel_weight": 1.0,
  "spike_block_threshold": 0.005,
  "gas_block_threshold": 0.4,
  "high_price_block_threshold": 0.98,
  "cap_risk_block_threshold": 0.005,
  "zero_prob_scale": 0.1,
  "transition_scale": 10.0
}
```

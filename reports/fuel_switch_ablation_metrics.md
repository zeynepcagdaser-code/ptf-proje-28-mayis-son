# Fuel Switch Ablation Metrics

Generated: `2026-05-31T21:57:35.360869+00:00`

This compares the same high precision residual model with and without explicit fuel-switch / marginality columns.

## Test Summary

| Scenario | Features | Test MAE | RMSE | Median AE | P90 AE | Delta vs persistence | <=50 TL |
|---|---:|---:|---:|---:|---:|---:|---:|
| `without_explicit_fuel_switch` | 85 | 531.46 | 832.19 | 258.76 | 1453.02 | -3.93 | 0.2340 |
| `with_explicit_fuel_switch` | 100 | 531.50 | 832.20 | 258.76 | 1452.97 | -3.89 | 0.2337 |

## Fuel-Switch Delta

- MAE delta, full minus without: `0.03` TL/MWh
- RMSE delta, full minus without: `0.01` TL/MWh
- P90 AE delta, full minus without: `-0.05` TL/MWh

Negative delta means the explicit fuel-switch columns improved the model.

## Regime-Wise MAE

### `without_explicit_fuel_switch`

| Regime | Rows | Model MAE | Persistence MAE | Delta |
|---|---:|---:|---:|---:|
| `negative_zero_pressure` | 387 | 154.23 | 139.01 | 15.22 |
| `normal` | 1395 | 496.24 | 501.00 | -4.76 |
| `spike_cap` | 73 | 1167.15 | 1183.67 | -16.52 |
| `tight` | 1769 | 615.53 | 622.47 | -6.94 |

- Cap miss rate: `0.5068493150684932`
- Spike/cap MAE: `1167.1500583176967`

### `with_explicit_fuel_switch`

| Regime | Rows | Model MAE | Persistence MAE | Delta |
|---|---:|---:|---:|---:|
| `negative_zero_pressure` | 387 | 154.42 | 139.01 | 15.41 |
| `normal` | 1395 | 496.27 | 501.00 | -4.73 |
| `spike_cap` | 73 | 1167.17 | 1183.67 | -16.50 |
| `tight` | 1769 | 615.53 | 622.47 | -6.94 |

- Cap miss rate: `0.5068493150684932`
- Spike/cap MAE: `1167.1695522979903`

## Columns Tested

- `gas_marginality_proxy`
- `hydro_displacement_score`
- `renewable_share_of_generation`
- `gas_share_of_generation`
- `renewable_minus_gas_shift`
- `cheap_supply_pressure`
- `low_demand_flag`
- `gas_off_flag`
- `renewable_share_high_flag`
- `hydro_high_flag`
- `zero_price_pressure_score`
- `load_deviation_from_weekly_norm`
- `load_deviation_from_monthly_norm`
- `demand_weakness_score`
- `load_vs_renewable_balance`

## Notes

- Both scenarios use the same time-based split and model family.
- The without scenario drops explicit fuel-switch columns from the feature matrix.
- Updated analyst scores remain available in both scenarios, so this isolates direct column contribution rather than the full reasoning-layer contribution.

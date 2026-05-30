# Persistence Baseline Metrics

Naive seasonal rule: `prediction(t+h) = PTF(t+h-24)` (same clock hour, previous day).

- Test anchors: 3282
- Prediction rows: 78768
- Valid rows: 78768

## Metrics (TL/MWh)

- MAE: 545.8065
- RMSE: 845.4696
- MAPE (all): 545.2224%
- MAPE (actual > 100.0): 68.5788%
- Zero-price MAE: 154.48470236439502
- Worst horizon: h23 (MAE 547.1319)

## Horizon MAE

| Hour | MAE |
|-----:|----:|
| 1 | 544.2368 |
| 2 | 544.1822 |
| 3 | 543.9286 |
| 4 | 543.6900 |
| 5 | 543.6144 |
| 6 | 544.1127 |
| 7 | 545.0633 |
| 8 | 545.7270 |
| 9 | 546.0395 |
| 10 | 546.6521 |
| 11 | 547.0917 |
| 12 | 546.9638 |
| 13 | 546.8631 |
| 14 | 546.4765 |
| 15 | 546.0298 |
| 16 | 545.7328 |
| 17 | 545.7496 |
| 18 | 546.3304 |
| 19 | 546.4119 |
| 20 | 546.4558 |
| 21 | 546.9513 |
| 22 | 546.9348 |
| 23 | 547.1319 |
| 24 | 546.9857 |

## LSTM vs persistence

- Persistence MAE: 545.8065
- LSTM MAE: 1084.9127
- Delta (LSTM − persistence): 539.1062
- Relative change: 98.77%
- **Persistence baseline beats or matches LSTM**
- Persistence is stronger → revisit LSTM features/architecture.
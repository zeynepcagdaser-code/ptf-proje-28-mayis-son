# LSTM Baseline Metrics

- **Device:** cpu
- **Epochs run:** 24
- **Best epoch:** 14
- **Final train loss:** 0.018514
- **Final validation loss:** 0.010847
- **Best validation loss:** 0.010847
- **Fit assessment:** reasonable_fit (train and validation losses close at stop)

## Test metrics (TL/MWh, inverse scaled)

- MAE: 1084.9127
- RMSE: 1279.6343
- MAPE: 4247.4388%
- Daily mean MAE: 1094.0770

- **Worst horizon:** h8 (MAE 1179.6735)

## Segment performance

- Zero-price MAE: 1641.9184284915843
- Spike (≥4800.0) MAE: None

## Horizon MAE

| Hour | MAE |
|-----:|----:|
| 1 | 966.0789 |
| 2 | 1025.9117 |
| 3 | 1086.5770 |
| 4 | 1130.1570 |
| 5 | 1160.0931 |
| 6 | 1175.7496 |
| 7 | 1179.6653 |
| 8 | 1179.6735 |
| 9 | 1175.6528 |
| 10 | 1171.1261 |
| 11 | 1166.7261 |
| 12 | 1164.7231 |
| 13 | 1155.0008 |
| 14 | 1136.4990 |
| 15 | 1105.9484 |
| 16 | 1078.2584 |
| 17 | 1045.9639 |
| 18 | 1019.2088 |
| 19 | 998.3819 |
| 20 | 983.5319 |
| 21 | 971.3557 |
| 22 | 968.2486 |
| 23 | 977.7575 |
| 24 | 1015.6149 |
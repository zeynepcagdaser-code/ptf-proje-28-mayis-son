# LSTM Baseline Metrics

- **Device:** cpu
- **Epochs run:** 16
- **Best epoch:** 6
- **Final train loss:** 0.036185
- **Final validation loss:** 0.011803
- **Best validation loss:** 0.011803
- **Fit assessment:** reasonable_fit (train and validation losses close at stop)

## Test metrics (TL/MWh, inverse scaled)

- MAE: 1183.3643
- RMSE: 1385.1092
- MAPE: 4429.4195%
- Daily mean MAE: 1191.9044

- **Worst horizon:** h11 (MAE 1284.3871)

## Segment performance

- Zero-price MAE: 1972.9561781839468
- Spike (≥4800.0) MAE: None

## Horizon MAE

| Hour | MAE |
|-----:|----:|
| 1 | 1042.8073 |
| 2 | 1103.3785 |
| 3 | 1166.6570 |
| 4 | 1192.9037 |
| 5 | 1215.4681 |
| 6 | 1227.8196 |
| 7 | 1241.7627 |
| 8 | 1251.1525 |
| 9 | 1262.3036 |
| 10 | 1272.0576 |
| 11 | 1284.3871 |
| 12 | 1281.8200 |
| 13 | 1281.2676 |
| 14 | 1252.8358 |
| 15 | 1237.5902 |
| 16 | 1206.8168 |
| 17 | 1193.5346 |
| 18 | 1176.7870 |
| 19 | 1144.8661 |
| 20 | 1110.0452 |
| 21 | 1077.4715 |
| 22 | 1045.8103 |
| 23 | 1048.4178 |
| 24 | 1082.7829 |
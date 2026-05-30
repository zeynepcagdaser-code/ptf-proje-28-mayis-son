# Tree Baseline (Advanced Pipeline) Metrics

- Backend: **lightgbm**
- Residual target: **yes** (final = persistence + residual)
- Hour-specific models: **24 × 24 horizons**
- Classifiers: zero-price + spike
- Rolling online refit: **True**
- Recency weight: last **60**d (boost **3.0×**), partial **90**d

## Aligned test (LSTM anchors)

- MAE: **602.87**
- RMSE: 793.02
- MAPE (actual > 100.0): 119.29%
- Zero-price MAE: 504.30282241409816
- Persistence MAE: 545.81
- Improvement vs persistence: **-10.45%**
- Residual LSTM MAE: 535.9692191571739
- Improvement vs residual LSTM: -12.48160176328456

## Horizon MAE

| h | MAE |
|--:|----:|
| 1 | 398.73 |
| 2 | 438.49 |
| 3 | 470.44 |
| 4 | 504.36 |
| 5 | 524.79 |
| 6 | 543.73 |
| 7 | 573.67 |
| 8 | 597.02 |
| 9 | 611.23 |
| 10 | 629.91 |
| 11 | 635.17 |
| 12 | 646.38 |
| 13 | 661.61 |
| 14 | 660.39 |
| 15 | 659.08 |
| 16 | 664.85 |
| 17 | 666.64 |
| 18 | 664.77 |
| 19 | 666.86 |
| 20 | 656.40 |
| 21 | 652.00 |
| 22 | 648.16 |
| 23 | 633.86 |
| 24 | 660.26 |
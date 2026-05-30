# LSTM Residual Metrics

_Evaluated from predictions CSV only (no model load)._

- **Source:** `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/predictions/lstm_residual_test_predictions.csv`
- **Test anchors:** 3282
- **Prediction rows:** 78768

## Final prediction metrics (TL/MWh)

- MAE: 535.9692
- RMSE: 772.9271
- MAPE (actual > 100.0): 80.4395%
- Zero-price MAE: 316.24175540270693
- Zero-price hours: 7190

- **Worst horizon:** h19 (MAE 549.6308)

## vs persistence

- Persistence MAE (this CSV): 545.8065
- Residual LSTM MAE: 535.9692
- Improvement vs persistence: 1.80%
- **Residual LSTM beats persistence baseline**

- Persistence MAE (`persistence_metrics.json`): 545.8065

## Horizon MAE (final prediction)

| Hour | MAE |
|-----:|----:|
| 1 | 539.6780 |
| 2 | 540.5037 |
| 3 | 538.8962 |
| 4 | 535.5320 |
| 5 | 532.9443 |
| 6 | 531.7234 |
| 7 | 532.1339 |
| 8 | 532.8189 |
| 9 | 535.4515 |
| 10 | 535.6640 |
| 11 | 535.7720 |
| 12 | 534.3205 |
| 13 | 533.1081 |
| 14 | 530.9591 |
| 15 | 529.6603 |
| 16 | 533.5222 |
| 17 | 539.6628 |
| 18 | 547.0911 |
| 19 | 549.6308 |
| 20 | 546.5309 |
| 21 | 540.4420 |
| 22 | 533.6751 |
| 23 | 528.0832 |
| 24 | 525.4574 |
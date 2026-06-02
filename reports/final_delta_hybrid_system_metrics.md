## Final delta-hybrid system metrics (test)

**Model eğitimi yapılmadı (deep learning yok; yalnızca tabular baseline regressor fit).**

- **Dataset**: feature_profile=`main_regression`, feature_count=**73**
- **Backend**: `lightgbm`

### Overall metrics (test, flattened h1-h24)

| model | MAE | RMSE | WAPE% | sMAPE% | MAPE% (actual>50) | MAPE% (actual>100) |
|---|---:|---:|---:|---:|---:|---:|
| persistence | 538.93 | 841.26 | 34.22 | 58.69 | 72.40 | 68.41 |
| delta_model | 535.85 | 758.99 | 34.03 | 63.52 | 93.20 | 88.67 |
| final_delta_hybrid | 486.52 | 749.17 | 30.89 | 60.52 | 67.55 | 64.91 |

### Regime MAE (test)

| model | zero-only | low<=50 | normal>100 | spike>=1000 |
|---|---:|---:|---:|---:|
| persistence | 144.40 | 148.49 | 586.30 | 671.94 |
| delta_model | 421.97 | 428.57 | 548.62 | 541.52 |
| final_delta_hybrid | 154.24 | 153.44 | 527.56 | 587.28 |

### Success criteria

- **final_better_than_persistence_mae**: True
- **final_better_than_prev_hybrid_mae_520**: True
- **final_better_than_prev_hybrid_mae_from_report**: True
- **mape_gt_100_close_to_8pct**: False

### Per-horizon MAE (test)

| h | persistence | delta_model | final_delta_hybrid |
|--:|------------:|-----------:|-------------------:|
| 1 | 540.87 | 425.05 | 421.19 |
| 2 | 540.59 | 456.91 | 443.15 |
| 3 | 540.23 | 494.74 | 452.29 |
| 4 | 539.93 | 507.96 | 459.39 |
| 5 | 539.65 | 527.93 | 468.32 |
| 6 | 539.44 | 538.89 | 472.92 |
| 7 | 539.20 | 521.06 | 468.43 |
| 8 | 539.11 | 518.15 | 476.17 |
| 9 | 538.89 | 543.23 | 481.19 |
| 10 | 538.89 | 513.95 | 485.78 |
| 11 | 538.89 | 526.35 | 489.21 |
| 12 | 538.86 | 539.57 | 493.82 |
| 13 | 538.77 | 549.53 | 499.20 |
| 14 | 538.43 | 538.28 | 499.09 |
| 15 | 538.05 | 556.40 | 500.40 |
| 16 | 537.81 | 549.32 | 505.10 |
| 17 | 537.86 | 543.11 | 500.31 |
| 18 | 538.54 | 573.36 | 510.19 |
| 19 | 538.63 | 574.05 | 515.32 |
| 20 | 538.16 | 554.87 | 506.15 |
| 21 | 538.56 | 565.51 | 508.90 |
| 22 | 538.39 | 572.85 | 510.93 |
| 23 | 538.30 | 573.39 | 505.41 |
| 24 | 538.15 | 596.03 | 503.59 |

Predictions: `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/predictions/final_delta_hybrid_predictions.csv`


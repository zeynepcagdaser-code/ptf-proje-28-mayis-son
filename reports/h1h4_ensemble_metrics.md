# h1–h4 prediction ensemble (no training)

Combines existing test predictions on LSTM-anchor aligned rows.

- **Rows:** 13124
- **Advanced tree baseline (mean h1–h4 MAE):** 453.09
- **Primary result:** `primary_weighted_ensemble` (0.7 advanced + 0.3 micro, fixed — not tuned on test)

## Strategy MAE (TL/MWh)

| Strategy | h1 | h2 | h3 | h4 | Mean h1–h4 | vs advanced |
|----------|-----:|-----:|-----:|-----:|-----:|-----:|
| advanced_tree_only | 398.84 | 438.56 | 470.51 | 504.47 | **453.09** | +0.00 |
| microstructure_only | 400.43 | 450.60 | 486.45 | 509.57 | **461.76** | +8.67 |
| short_expert_only | 431.30 | 476.14 | 495.83 | 500.46 | **475.93** | +22.84 |
| persistence_only | 544.30 | 544.33 | 544.08 | 543.84 | **544.14** | +91.04 |
| avg_advanced_micro | 377.64 | 427.02 | 462.52 | 490.65 | **439.46** | -13.64 |
| weighted_0.8_0.2 | 384.35 | 429.00 | 463.55 | 494.99 | **442.97** | -10.12 |
| weighted_0.7_0.3 | 380.17 | 426.79 | 461.98 | 492.23 | **440.29** | -12.80 |
| weighted_0.6_0.4 | 377.83 | 426.17 | 461.61 | 490.81 | **439.11** | -13.99 |
| weighted_0.5_0.5 | 377.64 | 427.02 | 462.52 | 490.65 | **439.46** | -13.64 |
| primary_weighted_ensemble **PRIMARY** | 380.17 | 426.79 | 461.98 | 492.23 | **440.29** | -12.80 |
| horizon_best_oracle_test *(test oracle — leakage)* | 243.82 | 282.07 | 306.65 | 317.99 | **287.63** | -165.46 |

## Horizon-specific best selector

No validation prediction CSVs available; horizon-specific selector uses test-set oracle only (labeled horizon_best_oracle_test — not used as primary result).

Oracle test MAE mean: **287.63**
Oracle model picks: {'advanced_tree': 5063, 'microstructure': 4033, 'short_expert': 4028}

## Verdict

Primary weighted ensemble (0.7/0.3) beats advanced tree on mean h1–h4 MAE (440.29 vs 453.09, Δ -12.80).

Primary ensemble mean MAE: **440.29**

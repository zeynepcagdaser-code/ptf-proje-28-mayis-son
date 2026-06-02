# Main Tabular Regression Metrics

- **Backend:** `hist_gradient_boosting`
- **Tabular features:** 400 (from 40 sequence features)

## MAE by split

| Split | main | persistence | hybrid |
|-------|-----:|------------:|-------:|
| train | 128.69 | 256.08 | 138.14 |
| validation | 432.63 | 401.41 | 450.10 |
| test | 820.12 | 538.93 | 520.30 |

## Test slices (main model)

- Overall MAE: **820.12**
- Zero-only MAE: **1203.13**
- Low<=50 MAE: **1216.22**
- Normal-price MAE: **769.94**

## Per-horizon MAE (test, main)

| h | MAE |
|--:|----:|
| 1 | 157.41 |
| 2 | 176.99 |
| 3 | 192.69 |
| 6 | 211.19 |
| 12 | 225.37 |
| 18 | 235.58 |
| 24 | 209.15 |

# Rolling Next-24 PTF Metrics

Generated: `2026-06-03T21:07:53.830826+00:00`
Selected profile: `full_market`

## Final Train/Val/Test Split

| Profile | Val MAE | Val persistence | Test MAE | Test persistence | Features | Rows |
|---|---:|---:|---:|---:|---:|---:|
| full_market | 246.40 | 548.52 | 375.38 | 537.35 | 202 | 1346388 |

## Walk-Forward Cross-Validation (expanding window)

| Profile | Fold | Train rows | Val rows | Model MAE | Persistence MAE | Improvement |
|---|---|---:|---:|---:|---:|---:|
| full_market | train=2020-2022 val=2023 | 626964 | 210240 | 316.23 | 366.24 | 13.7% |
| full_market | train=2020-2023 val=2024 | 837204 | 210816 | 204.17 | 367.19 | 44.4% |
| full_market | train=2020-2024 val=2025 | 1048020 | 210240 | 227.42 | 398.51 | 42.9% |
| full_market | train=2020-2025 val=2026 | 1258260 | 88128 | 387.95 | 537.35 | 27.8% |

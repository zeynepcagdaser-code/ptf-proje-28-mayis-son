# Rolling Next-24 PTF Metrics

Generated: `2026-06-04T03:58:52.286831+00:00`
Selected profile: `full_market`

## Final Train/Val/Test Split

| Profile | Val MAE | Val persistence | Test MAE | Test persistence | Features | Rows |
|---|---:|---:|---:|---:|---:|---:|
| full_market | 251.41 | 548.52 | 359.34 | 538.29 | 208 | 1346964 |

## Walk-Forward Cross-Validation (expanding window)

| Profile | Fold | Train rows | Val rows | Model MAE | Persistence MAE | Improvement |
|---|---|---:|---:|---:|---:|---:|
| full_market | train=2020-2022 val=2023 | 626964 | 210240 | 305.90 | 366.24 | 16.5% |
| full_market | train=2020-2023 val=2024 | 837204 | 210816 | 205.37 | 367.19 | 44.1% |
| full_market | train=2020-2024 val=2025 | 1048020 | 210240 | 222.14 | 398.51 | 44.3% |
| full_market | train=2020-2025 val=2026 | 1258260 | 88704 | 380.67 | 538.29 | 29.3% |

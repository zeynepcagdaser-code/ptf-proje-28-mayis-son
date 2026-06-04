# Rolling Next-24 PTF Metrics

Generated: `2026-06-04T01:31:25.306052+00:00`
Selected profile: `full_market`

## Final Train/Val/Test Split

| Profile | Val MAE | Val persistence | Test MAE | Test persistence | Features | Rows |
|---|---:|---:|---:|---:|---:|---:|
| full_market | 250.76 | 548.52 | 366.31 | 537.35 | 217 | 1346388 |

## Walk-Forward Cross-Validation (expanding window)

| Profile | Fold | Train rows | Val rows | Model MAE | Persistence MAE | Improvement |
|---|---|---:|---:|---:|---:|---:|
| full_market | train=2020-2022 val=2023 | 626964 | 210240 | 316.04 | 366.24 | 13.7% |
| full_market | train=2020-2023 val=2024 | 837204 | 210816 | 207.65 | 367.19 | 43.4% |
| full_market | train=2020-2024 val=2025 | 1048020 | 210240 | 221.54 | 398.51 | 44.4% |
| full_market | train=2020-2025 val=2026 | 1258260 | 88128 | 378.74 | 537.35 | 29.5% |

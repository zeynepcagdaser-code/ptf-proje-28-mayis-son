# Advanced Tree h1–h4 Feature Selection

Importance cells loaded: 96

## Experiments

| Set | #feat (h1) | Val MAE | Test MAE | Overfit (test−val) |
|-----|----------:|--------:|---------:|-------------------:|
| 10 | 10 | 271.79 | 491.92 | 220.14 |
| 20 | 20 | 264.50 | 468.98 | 204.48 |
| 30 | 30 | 259.21 | 484.69 | 225.48 |
| 50 | 50 | 258.12 | 485.71 | 227.58 |
| all | 108 | 259.69 | 485.16 | 225.47 |

## Baselines (h1–h4 mean MAE)

- **advanced_tree**: 453.00
- **short_expert**: 475.92
- **residual_lstm**: 538.65
- **persistence**: 544.01

**Best set:** `20`

No feature subset beat current advanced tree h1–h4 mean MAE (453.0). Best tuned: '20' at 469.0.
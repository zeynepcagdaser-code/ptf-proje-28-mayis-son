# h1–h4 validation-weighted ensemble

Per-horizon blend weights chosen on **validation** only; applied to test.

- Advanced tree on validation: regressor (train-only), not regressor_online (train+val)
- Test predictions use existing CSVs (online/advanced pipeline). Weights are chosen on validation with out-of-sample advanced tree regressors.

- Validation rows (aligned): 33964
- Test rows (aligned): 13124

## Selected weights (w × advanced + (1−w) × microstructure)

| Horizon | w (advanced) | w (micro) |
|--------|-------------:|----------:|
| h1 | 0.6 | 0.4 |
| h2 | 0.7 | 0.3 |
| h3 | 0.8 | 0.2 |
| h4 | 1.0 | 0.0 |

## Test MAE (TL/MWh)

| Model | h1 | h2 | h3 | h4 | Mean h1–h4 | vs advanced |
|-------|-----:|-----:|-----:|-----:|-----:|-----:|
| advanced_tree_only | 398.84 | 438.56 | 470.51 | 504.47 | **453.09** | +0.00 |
| microstructure_only | 394.88 | 445.05 | 485.02 | 510.97 | **458.98** | +5.88 |
| validation_weighted_ensemble **PRIMARY** | 376.29 | 425.50 | 462.54 | 504.47 | **442.20** | -10.89 |
| fixed_0.7_0.3 | 379.23 | 425.50 | 460.49 | 493.02 | **439.56** | -13.53 |
| test_oracle_weights *(test oracle — leakage)* | 375.40 | 424.23 | 459.73 | 491.82 | **437.80** | -15.30 |

## Verdict

Validation-weighted ensemble beats advanced tree on test (442.20 vs 453.09, Δ -10.89).

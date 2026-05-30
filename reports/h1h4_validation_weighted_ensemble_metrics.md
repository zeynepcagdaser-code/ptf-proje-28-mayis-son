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
| h3 | 0.9 | 0.1 |
| h4 | 1.0 | 0.0 |

## Test MAE (TL/MWh)

| Model | h1 | h2 | h3 | h4 | Mean h1–h4 | vs advanced |
|-------|-----:|-----:|-----:|-----:|-----:|-----:|
| advanced_tree_only | 398.84 | 438.56 | 470.51 | 504.47 | **453.09** | +0.00 |
| microstructure_only | 400.43 | 450.60 | 486.45 | 509.57 | **461.76** | +8.67 |
| validation_weighted_ensemble **PRIMARY** | 377.83 | 426.79 | 466.39 | 504.47 | **443.87** | -9.22 |
| fixed_0.7_0.3 | 380.17 | 426.79 | 461.98 | 492.23 | **440.29** | -12.80 |
| test_oracle_weights *(test oracle — leakage)* | 377.64 | 426.17 | 461.61 | 490.65 | **439.02** | -14.08 |

## Verdict

Validation-weighted ensemble beats advanced tree on test (443.87 vs 453.09, Δ -9.22).

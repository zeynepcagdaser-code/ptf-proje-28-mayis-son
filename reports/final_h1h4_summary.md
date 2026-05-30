# Final h1–h4 pipeline checkpoint

Generated: 2026-05-30T12:08:23.486178+00:00

## Checkpoint

- **Best model (test):** `validation_weighted_ensemble` — mean h1–h4 MAE **443.87**
- **Aligned test rows:** 13124 (anchors: 3281)
- **Global seed:** 42

## Pipeline audit

- Split alignment: **pass**
- Persistence shift(24): **pass**
- Prediction CSV alignment: **pass**
- Leakage / inference policy: **pass**

### Inference policy (summary)

| Stage | Advanced tree | Microstructure | Ensemble weights |
|-------|---------------|----------------|------------------|
| Test deploy | `regressor_online` | saved boosters | from validation only |
| Weight search | `regressor` (train-only) | saved boosters | grid 0.0–1.0 per horizon |

## Model comparison (test, TL/MWh)

| Model | h1 | h2 | h3 | h4 | Mean | vs persistence |
|-------|-----:|-----:|-----:|-----:|-----:|---------------:|
| validation_weighted_ensemble ★ | 377.83 | 426.79 | 466.39 | 504.47 | **443.87** | +18.4% |
| advanced_tree | 398.84 | 438.56 | 470.51 | 504.47 | **453.09** | +16.7% |
| microstructure | 400.43 | 450.60 | 486.45 | 509.57 | **461.76** | +15.1% |
| short_expert | 431.30 | 476.14 | 495.83 | 500.46 | **475.93** | +12.5% |
| residual_lstm | 539.66 | 540.41 | 538.82 | 535.48 | **538.59** | +1.0% |
| persistence | 544.30 | 544.33 | 544.08 | 543.84 | **544.14** | +0.0% |

## Validation-selected ensemble weights

{'1': 0.6, '2': 0.7, '3': 0.9, '4': 1.0}

## Reproducibility artifacts

- `reports/reproducibility/model_configs.json`
- `reports/reproducibility/advanced_tree_base_features.json`
- `reports/reproducibility/microstructure_horizon_features.json`

## Notes

Primary deployable short-horizon stack: validation-weighted blend of advanced tree and microstructure. Test uses online advanced tree CSV; weights tuned on validation with train-only regressors. Do not use test oracle or fixed 0.7/0.3 as primary metrics.

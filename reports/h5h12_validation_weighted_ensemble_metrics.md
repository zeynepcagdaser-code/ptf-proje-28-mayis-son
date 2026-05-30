# h5–h12 validation-weighted ensemble

Weights chosen on **validation** only; applied to test. h1–h4 checkpoint untouched.

- Validation rows: 67928
- Test rows: 26248
- Advanced tree validation inference: regressor (train-only), not regressor_online

## Selected weights

| h | w_adv | w_micro |
|--:|------:|--------:|
| 5 | 1.0 | 0.0 |
| 6 | 1.0 | 0.0 |
| 7 | 1.0 | 0.0 |
| 8 | 1.0 | 0.0 |
| 9 | 1.0 | 0.0 |
| 10 | 1.0 | 0.0 |
| 11 | 1.0 | 0.0 |
| 12 | 1.0 | 0.0 |

## Test MAE (TL/MWh)

| Model | h5 | h6 | h7 | h8 | h9 | h10 | h11 | h12 | Mean h5–h12 | vs adv | vs pers % |
|-------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| persistence | 543.78 | 544.27 | 545.22 | 545.89 | 546.21 | 546.82 | 547.26 | 547.13 | **545.82** | -49.41 | +0.0% |
| advanced_tree | 524.86 | 543.72 | 573.44 | 596.90 | 611.27 | 629.95 | 635.32 | 646.42 | **595.23** | +0.00 | -9.1% |
| microstructure | 525.48 | 524.89 | 561.97 | 546.92 | 577.31 | 583.94 | 616.60 | 614.20 | **568.91** | -26.32 | -4.2% |
| validation_weighted_ensemble **PRIMARY** | 524.86 | 543.72 | 573.44 | 596.90 | 611.27 | 629.95 | 635.32 | 646.42 | **595.23** | +0.00 | -9.1% |
| test_oracle_weights *(test oracle)* | 510.03 | 516.26 | 550.06 | 544.60 | 570.23 | 577.36 | 603.37 | 605.41 | **559.66** | -35.57 | -2.5% |

## Verdict

Validation selected w=1.0 for all h5–h12 (advanced tree best on validation). Test ensemble equals advanced tree (mean MAE 595.23). Microstructure alone is lower on test (568.91) but not used — no validation gain.

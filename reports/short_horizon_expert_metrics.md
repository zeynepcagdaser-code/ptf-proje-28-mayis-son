# Short Horizon Expert (h1–h4)

Final prediction: `persistence_h + predicted_residual`

- Features: 17 base + `persistence_h` per horizon
- Test anchors (aligned): 3282

## Expert MAE (aligned)

| Horizon | MAE |
|--------|-----:|
| h1 | 431.30 |
| h2 | 476.16 |
| h3 | 495.81 |
| h4 | 500.40 |
| **Mean h1–h4** | **475.92** |

### Persistence

| Horizon | MAE |
|--------|-----:|
| h1 | 544.24 |
| h2 | 544.18 |
| h3 | 543.93 |
| h4 | 543.69 |
| Mean h1–h4 | 544.01 |

### Advanced tree

| Horizon | MAE |
|--------|-----:|
| h1 | 398.73 |
| h2 | 438.49 |
| h3 | 470.44 |
| h4 | 504.36 |
| Mean h1–h4 | 453.00 |

### Residual LSTM

| Horizon | MAE |
|--------|-----:|
| h1 | 539.68 |
| h2 | 540.50 |
| h3 | 538.90 |
| h4 | 535.53 |
| Mean h1–h4 | 538.65 |

## vs persistence (expert − persistence MAE)

- h1: -112.94 TL/MWh
- h2: -68.02 TL/MWh
- h3: -48.12 TL/MWh
- h4: -43.29 TL/MWh
- Mean: -68.09 TL/MWh

## Summary

- Beats persistence (mean): **True**
- Beats advanced tree (mean): **False**
- Beats residual LSTM (mean): **True**

**Short expert beats persistence (475.9 vs 544.0) but is worse than advanced tree h1–h4 (453.0).**
# Microstructure h1–h4 LightGBM

- **Source:** `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/features/lstm_microstructure_next24_v1.parquet`
- **Features per horizon:** 99 (all non-target + `persistence_h`)
- **Method:** persistence + residual

## Model MAE (aligned test)

| Horizon | MAE |
|--------|-----:|
| h1 | 400.43 |
| h2 | 450.60 |
| h3 | 486.45 |
| h4 | 509.57 |
| **Mean h1–h4** | **461.76** |

### Persistence
| Horizon | MAE |
|--------|-----:|
| h1 | 544.30 |
| h2 | 544.33 |
| h3 | 544.08 |
| h4 | 543.84 |
| Mean | 544.14 |

### Advanced Tree
| Horizon | MAE |
|--------|-----:|
| h1 | 398.84 |
| h2 | 438.56 |
| h3 | 470.51 |
| h4 | 504.47 |
| Mean | 453.09 |

### Short Expert
| Horizon | MAE |
|--------|-----:|
| h1 | 431.30 |
| h2 | 476.14 |
| h3 | 495.83 |
| h4 | 500.46 |
| Mean | 475.93 |

## Delta vs baselines (model − baseline, TL/MWh)

- **persistence** mean: -82.38
- **advanced_tree** mean: +8.67
- **short_expert** mean: -14.17

**Microstructure h1–h4 is WORSE than advanced tree on mean MAE (461.8 vs 453.1). Advanced tree remains best for short horizons.**
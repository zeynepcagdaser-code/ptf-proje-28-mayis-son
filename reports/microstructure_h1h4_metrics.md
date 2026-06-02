# Microstructure h1–h4 LightGBM

- **Source:** `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/features/lstm_microstructure_next24_v1.parquet`
- **Features per horizon:** 120 (all non-target + `persistence_h`)
- **Method:** persistence + residual

## Model MAE (aligned test)

| Horizon | MAE |
|--------|-----:|
| h1 | 394.88 |
| h2 | 445.05 |
| h3 | 485.02 |
| h4 | 510.97 |
| **Mean h1–h4** | **458.98** |

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

- **persistence** mean: -85.16
- **advanced_tree** mean: +5.88
- **short_expert** mean: -16.95

**Microstructure h1–h4 is WORSE than advanced tree on mean MAE (459.0 vs 453.1). Advanced tree remains best for short horizons.**
# Regime Classifier Metrics

Generated: `2026-05-31T13:31:44.003806+00:00`

This run trains only the `target_regime` classifier. It does not train price experts, ensembles, or final PTF forecasts.

## Split

| Split | Years | Rows |
|---|---|---:|
| `train` | `2020-2024` | 43848 |
| `validation` | `2025-2025` | 8760 |
| `test` | `2026-2026` | 3624 |

## Test Metrics

- Accuracy: `0.8722`
- Balanced accuracy: `0.6690`
- Macro F1: `0.6699`
- Spike/cap recall: `0.0274`
- Negative/zero pressure recall: `0.8708`
- Normal/tight accuracy: `0.8919`

## Baselines

| Baseline | Accuracy | Balanced accuracy |
|---|---:|---:|
| model | 0.8722 | 0.6690 |
| lag24_regime | 0.7740 | 0.6701 |
| most frequent regime | 0.3849 | 0.2500 |

## Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `negative_zero_pressure` | 0.8619 | 0.8708 | 0.8663 | 387 |
| `normal` | 0.8558 | 0.8638 | 0.8598 | 1395 |
| `tight` | 0.8870 | 0.9141 | 0.9003 | 1769 |
| `spike_cap` | 1.0000 | 0.0274 | 0.0533 | 73 |

## Persistence Failure Hours

- Threshold: train persistence_error p75 = `304.12`
- Rows: `1673`
- Model recall/accuracy on failure hours: `0.8404`
- Lag24 baseline recall/accuracy on failure hours: `0.6736`

## Transition Recall - Test

| Transition | Rows | Model recall | Lag24 baseline recall |
|---|---:|---:|---:|
| `tight -> tight` | 1523 | 0.9363 | 1.0000 |
| `normal -> normal` | 1014 | 0.9014 | 1.0000 |
| `tight -> normal` | 238 | 0.7017 | 0.0000 |
| `negative_zero_pressure -> negative_zero_pressure` | 232 | 0.9224 | 1.0000 |
| `normal -> tight` | 212 | 0.7642 | 0.0000 |
| `normal -> negative_zero_pressure` | 149 | 0.7852 | 0.0000 |
| `negative_zero_pressure -> normal` | 138 | 0.8768 | 0.0000 |
| `spike_cap -> spike_cap` | 36 | 0.0556 | 1.0000 |
| `spike_cap -> tight` | 29 | 0.9310 | 0.0000 |
| `tight -> spike_cap` | 25 | 0.0000 | 0.0000 |
| `normal -> spike_cap` | 12 | 0.0000 | 0.0000 |

## Critical Evaluation

The classifier does not collapse to only the majority class. Spike/cap recall is weak and must improve before cap expert routing (0.027). Zero-pressure recall is acceptable as a first prototype (0.871). On persistence-failure hours, model recall is 0.840 vs lag24 baseline 0.674 (delta +0.167).

## Leakage Checks

- **Forbidden feature columns absent**: `pass` - Forbidden columns present in model feature matrix: []
- **analyst_reason_text excluded**: `pass` - Text reason column is kept out of numeric feature matrix.
- **historical interim-mcp oracle excluded**: `pass` - Only point-in-time snapshot columns from feature store are eligible.
- **same-hour realized SMF/YAL-YAT excluded**: `pass` - Feature store exposes only lagged SMF/YAL/YAT fields.

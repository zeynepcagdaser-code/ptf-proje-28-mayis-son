# Spike / Cap Risk Detector Metrics

Generated: `2026-05-31T13:37:02.956529+00:00`

This run trains only a binary `is_spike_cap` classifier. It does not train price experts, ensembles, or final PTF regressors.

## Split

| Split | Years | Rows | Positives | Positive rate |
|---|---|---:|---:|---:|
| `train` | `2020-2024` | 43848 | 1631 | 0.0372 |
| `validation` | `2025-2025` | 8760 | 0 | 0.0000 |
| `test` | `2026-2026` | 3624 | 73 | 0.0201 |

## Test Summary

- PR-AUC: `0.7081`
- ROC-AUC: `0.9641`
- Operating threshold: `0.2`
- Recall: `0.2603`
- Precision: `0.9500`
- Balanced accuracy: `0.6300`
- Cap miss rate: `0.7397`
- False alarm rate: `0.0003`

## Threshold Analysis

| Threshold | Precision | Recall | F1 | Balanced acc | Cap miss | False alarm | TP | FP | FN | TN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 1.0000 | 0.1096 | 0.1975 | 0.5548 | 0.8904 | 0.0000 | 8 | 0 | 65 | 3551 |
| 0.30 | 1.0000 | 0.1507 | 0.2619 | 0.5753 | 0.8493 | 0.0000 | 11 | 0 | 62 | 3551 |
| 0.20 | 0.9500 | 0.2603 | 0.4086 | 0.6300 | 0.7397 | 0.0003 | 19 | 1 | 54 | 3550 |
| 0.10 | 0.8824 | 0.4110 | 0.5607 | 0.7049 | 0.5890 | 0.0011 | 30 | 4 | 43 | 3547 |

## Baseline Comparison - Test

| Method | Precision | Recall | Balanced acc | False alarm | Cap miss |
|---|---:|---:|---:|---:|---:|
| spike detector @ 0.2 | 0.9500 | 0.2603 | 0.6300 | 0.0003 | 0.7397 |
| lag24_regime baseline | 0.5143 | 0.4932 | 0.7418 | 0.0096 | 0.5068 |
| multiclass classifier | 1.0000 | 0.0274 | 0.5137 | 0.0000 | 0.9726 |

## Persistence Failure Hours

- Rows: `1673`
- Spike positives: `41`
- Detector recall: `0.0976`
- Lag24 baseline recall: `0.0976`
- Multiclass recall: `0.0244`

## Spike Transition Recall

| Transition | Spike count | Detector recall | Lag24 recall | Multiclass recall |
|---|---:|---:|---:|---:|
| `spike_cap -> spike_cap` | 36 | 0.4444 | 1.0000 | 0.0556 |
| `tight -> spike_cap` | 25 | 0.1200 | 0.0000 | 0.0000 |
| `normal -> spike_cap` | 12 | 0.0000 | 0.0000 | 0.0000 |

## Top Feature Importances

| Feature | Importance |
|---|---:|
| `ptf_lag_1` | 1332 |
| `gas_share` | 1160 |
| `load_minus_kgup` | 970 |
| `residual_load_ramp` | 969 |
| `hour` | 935 |
| `ptf_lag_168` | 909 |
| `volatility_cluster_score` | 903 |
| `analyst_persistence_break_score` | 902 |
| `rolling_volatility` | 872 |
| `analyst_spike_score` | 859 |
| `gas_maintenance` | 836 |
| `coal_maintenance` | 812 |
| `hydro_share` | 805 |
| `yal_lagged` | 783 |
| `ptf_lag_24` | 749 |
| `load_ramp_3h` | 749 |
| `load_ramp_1h` | 723 |
| `coal_share` | 700 |
| `hydro_maintenance` | 663 |
| `thermal_share` | 641 |

## Critical Evaluation

Spike/cap recall is still weak at threshold 0.2: 0.260. Precision is 0.950; false alarm rate is 0.000. Top features are: ptf_lag_1, gas_share, load_minus_kgup, residual_load_ramp, hour, ptf_lag_168, volatility_cluster_score, analyst_persistence_break_score. Recall comparison: detector 0.260, lag24 baseline 0.493, multiclass 0.027. The binary detector adds value over the multiclass classifier for spike screening.

## Leakage Checks

- **Forbidden feature columns absent**: `pass` - Forbidden columns present in model feature matrix: []
- **historical interim-mcp oracle excluded**: `pass` - Only point-in-time snapshot columns from feature store are eligible.
- **same-hour finalized PTF excluded**: `pass` - Target regime and price are labels/evaluation only.
- **same-hour realized SMF/YAL-YAT excluded**: `pass` - Feature store exposes only lagged SMF/YAL-YAT fields.

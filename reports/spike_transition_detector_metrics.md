# Spike Transition Detector Metrics

Generated: `2026-05-31T13:45:42.425844+00:00`

This model detects only new spike/cap transitions where `target_regime == spike_cap` and `lag24_regime != spike_cap`.
It does not train price experts, ensembles, or final PTF regressors.

## Split

| Split | Years | Rows | Positives | Positive rate |
|---|---|---:|---:|---:|
| `train` | `2020-2024` | 43848 | 501 | 0.0114 |
| `validation` | `2025-2025` | 8760 | 0 | 0.0000 |
| `test` | `2026-2026` | 3624 | 37 | 0.0102 |

## Test Summary

- PR-AUC: `0.4899`
- ROC-AUC: `0.9011`
- Operating threshold: `0.001`
- Recall: `0.4595`
- Precision: `0.5312`
- False alarm rate: `0.0042`
- Miss rate: `0.5405`

## Threshold Analysis

| Threshold | Precision | Recall | F1 | Balanced acc | Miss | False alarm | TP | FP | FN | TN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.500 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 1.0000 | 0.0000 | 0 | 0 | 37 | 3587 |
| 0.300 | 1.0000 | 0.0270 | 0.0526 | 0.5135 | 0.9730 | 0.0000 | 1 | 0 | 36 | 3587 |
| 0.200 | 1.0000 | 0.0270 | 0.0526 | 0.5135 | 0.9730 | 0.0000 | 1 | 0 | 36 | 3587 |
| 0.100 | 1.0000 | 0.0270 | 0.0526 | 0.5135 | 0.9730 | 0.0000 | 1 | 0 | 36 | 3587 |
| 0.050 | 1.0000 | 0.0811 | 0.1500 | 0.5405 | 0.9189 | 0.0000 | 3 | 0 | 34 | 3587 |
| 0.010 | 1.0000 | 0.2162 | 0.3556 | 0.6081 | 0.7838 | 0.0000 | 8 | 0 | 29 | 3587 |
| 0.005 | 0.9091 | 0.2703 | 0.4167 | 0.6350 | 0.7297 | 0.0003 | 10 | 1 | 27 | 3586 |
| 0.002 | 0.6190 | 0.3514 | 0.4483 | 0.6746 | 0.6486 | 0.0022 | 13 | 8 | 24 | 3579 |
| 0.001 | 0.5312 | 0.4595 | 0.4928 | 0.7276 | 0.5405 | 0.0042 | 17 | 15 | 20 | 3572 |

## Comparison

| Method | Precision | Recall | False alarm | Miss |
|---|---:|---:|---:|---:|
| transition detector @ 0.001 | 0.5312 | 0.4595 | 0.0042 | 0.5405 |
| lag24 baseline | 0.0000 | 0.0000 | 0.0195 | 1.0000 |
| binary spike detector @0.2 | 0.1500 | 0.0811 | 0.0047 | 0.9189 |
| binary spike detector @0.1 | 0.2941 | 0.2703 | 0.0067 | 0.7297 |

## Transition Recall

| Transition | Positives | Detector | Lag24 | Binary @0.2 | Binary @0.1 |
|---|---:|---:|---:|---:|---:|
| `normal -> spike_cap` | 12 | 0.1667 | 0.0000 | 0.0000 | 0.0833 |
| `tight -> spike_cap` | 25 | 0.6000 | 0.0000 | 0.1200 | 0.3600 |
| `negative_zero_pressure -> spike_cap` | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Top Feature Importances

| Feature | Importance |
|---|---:|
| `ptf_lag_1` | 1477 |
| `ptf_lag_24` | 1007 |
| `ptf_lag_168` | 957 |
| `load_minus_kgup` | 938 |
| `residual_load_ramp_vs_lag24` | 925 |
| `outage_stress_vs_lag24` | 865 |
| `hour` | 830 |
| `residual_load_ramp` | 786 |
| `load_minus_kgup_vs_lag24` | 783 |
| `load_ramp_3h` | 753 |
| `yal_lagged` | 749 |
| `wind_today_vs_lag24` | 744 |
| `gas_share` | 692 |
| `rolling_volatility` | 675 |
| `hydro_share` | 651 |
| `residual_load_vs_lag24` | 648 |
| `load_ramp_1h` | 647 |
| `smf_lag_24` | 645 |
| `gas_share_vs_lag24` | 624 |
| `volatility_cluster_score` | 600 |
| `analyst_spike_score` | 589 |
| `analyst_persistence_break_score` | 576 |
| `smf_spread_lagged` | 569 |
| `gas_maintenance` | 559 |
| `analyst_confidence_score` | 552 |

## Critical Evaluation

Operating threshold 0.001 recall is 0.459, precision 0.531, false alarm 0.004. Comparison recall: transition detector 0.459, binary spike detector @0.1 0.270, lag24 baseline 0.000. The transition detector improves new-spike recall over the broad binary spike detector. Recall is still weak for cap-entry routing; feature/threshold design needs more work.

## Leakage Checks

- **Forbidden feature columns absent**: `pass` - Forbidden columns present in model feature matrix: []
- **same-hour finalized PTF excluded**: `pass` - target_regime and transition_label are labels/evaluation only.
- **same-hour realized SMF/YAL/YAT excluded**: `pass` - Feature store exposes only lagged SMF/YAL/YAT fields.
- **historical interim oracle excluded**: `pass` - No raw historical interim_mcp source is read.

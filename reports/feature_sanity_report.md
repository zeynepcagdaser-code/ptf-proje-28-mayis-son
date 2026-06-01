# Feature Sanity Report

Slices are defined using `target_1h` (next-hour realized PTF) to approximate regime buckets.

## Correlation with target_1h (focus features)

| Feature | corr(target_1h) |
|---------|-----------------:|
| `renewable_pressure` | -0.1984 |
| `thermal_price_setting_share` | 0.1984 |
| `gas_coal_competition_index` | 0.1578 |
| `renewable_suppression_pressure` | -0.2383 |
| `gas_share` | 0.0603 |
| `coal_share` | 0.3063 |
| `low_load_flag` | -0.1473 |
| `holiday_low_load_flag` | -0.1461 |
| `zero_price_risk_proxy` | -0.2144 |
| `ptf_lag_24` | 0.8732 |

## Slice summaries

### actual_eq_0 (rows=424)

| Feature | mean | p50 | p90 | non-null |
|---------|-----:|----:|----:|--------:|
| `renewable_pressure` | 0.7993 | 0.8100 | 0.8778 | 424 |
| `thermal_price_setting_share` | 0.2007 | 0.1900 | 0.2731 | 424 |
| `gas_coal_competition_index` | 0.1835 | 0.1541 | 0.3595 | 424 |
| `renewable_suppression_pressure` | 0.7371 | 0.8072 | 0.8926 | 424 |
| `gas_share` | 0.0193 | 0.0141 | 0.0380 | 424 |
| `coal_share` | 0.1706 | 0.1605 | 0.2291 | 424 |
| `low_load_flag` | 0.5920 | 1.0000 | 1.0000 | 424 |
| `holiday_low_load_flag` | 0.5472 | 1.0000 | 1.0000 | 424 |
| `zero_price_risk_proxy` | 0.7734 | 0.8527 | 0.9230 | 424 |
| `ptf_lag_24` | 241.8796 | 10.2150 | 955.2830 | 424 |

### actual_le_50 (rows=926)

| Feature | mean | p50 | p90 | non-null |
|---------|-----:|----:|----:|--------:|
| `renewable_pressure` | 0.7587 | 0.7625 | 0.8398 | 926 |
| `thermal_price_setting_share` | 0.2413 | 0.2375 | 0.3345 | 926 |
| `gas_coal_competition_index` | 0.2282 | 0.2254 | 0.4051 | 926 |
| `renewable_suppression_pressure` | 0.7511 | 0.8000 | 0.8701 | 926 |
| `gas_share` | 0.0279 | 0.0264 | 0.0502 | 926 |
| `coal_share` | 0.2013 | 0.1963 | 0.2790 | 926 |
| `low_load_flag` | 0.7333 | 1.0000 | 1.0000 | 926 |
| `holiday_low_load_flag` | 0.6210 | 1.0000 | 1.0000 | 926 |
| `zero_price_risk_proxy` | 0.7938 | 0.8511 | 0.9060 | 926 |
| `ptf_lag_24` | 321.5256 | 135.6100 | 1198.4450 | 926 |

### actual_le_100 (rows=1177)

| Feature | mean | p50 | p90 | non-null |
|---------|-----:|----:|----:|--------:|
| `renewable_pressure` | 0.7442 | 0.7498 | 0.8355 | 1177 |
| `thermal_price_setting_share` | 0.2558 | 0.2502 | 0.3551 | 1177 |
| `gas_coal_competition_index` | 0.2414 | 0.2392 | 0.4118 | 1177 |
| `renewable_suppression_pressure` | 0.7460 | 0.7932 | 0.8610 | 1177 |
| `gas_share` | 0.0314 | 0.0296 | 0.0566 | 1177 |
| `coal_share` | 0.2121 | 0.2045 | 0.2939 | 1177 |
| `low_load_flag` | 0.7502 | 1.0000 | 1.0000 | 1177 |
| `holiday_low_load_flag` | 0.6279 | 1.0000 | 1.0000 | 1177 |
| `zero_price_risk_proxy` | 0.7909 | 0.8442 | 0.8964 | 1177 |
| `ptf_lag_24` | 386.8534 | 170.0000 | 1374.6940 | 1177 |

### normal_price (rows=53138)

| Feature | mean | p50 | p90 | non-null |
|---------|-----:|----:|----:|--------:|
| `renewable_pressure` | 0.4169 | 0.3967 | 0.6057 | 53138 |
| `thermal_price_setting_share` | 0.5831 | 0.6033 | 0.7420 | 53138 |
| `gas_coal_competition_index` | 0.7018 | 0.7760 | 0.9638 | 53138 |
| `renewable_suppression_pressure` | 0.3975 | 0.3553 | 0.6346 | 53138 |
| `gas_share` | 0.2240 | 0.2345 | 0.3650 | 53138 |
| `coal_share` | 0.3487 | 0.3458 | 0.4426 | 53138 |
| `low_load_flag` | 0.3522 | 0.0000 | 1.0000 | 53138 |
| `holiday_low_load_flag` | 0.1564 | 0.0000 | 1.0000 | 53138 |
| `zero_price_risk_proxy` | 0.4693 | 0.4219 | 0.7131 | 53138 |
| `ptf_lag_24` | 1681.3503 | 1799.9700 | 3125.0000 | 53138 |

### spike_price (rows=1701)

| Feature | mean | p50 | p90 | non-null |
|---------|-----:|----:|----:|--------:|
| `renewable_pressure` | 0.3093 | 0.2898 | 0.4117 | 1701 |
| `thermal_price_setting_share` | 0.6907 | 0.7102 | 0.7931 | 1701 |
| `gas_coal_competition_index` | 0.8033 | 0.8532 | 0.9785 | 1701 |
| `renewable_suppression_pressure` | 0.2574 | 0.2123 | 0.4961 | 1701 |
| `gas_share` | 0.2824 | 0.3021 | 0.3750 | 1701 |
| `coal_share` | 0.3936 | 0.3988 | 0.4675 | 1701 |
| `low_load_flag` | 0.1364 | 0.0000 | 1.0000 | 1701 |
| `holiday_low_load_flag` | 0.0694 | 0.0000 | 0.0000 | 1701 |
| `zero_price_risk_proxy` | 0.3391 | 0.2904 | 0.5728 | 1701 |
| `ptf_lag_24` | 3909.7995 | 4200.0000 | 4800.0000 | 1701 |


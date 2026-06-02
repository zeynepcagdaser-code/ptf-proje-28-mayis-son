# Low-Price Rule Baselines (Test)

- **Generated (UTC):** 2026-06-01T20:43:17.231935+00:00
- **Test anchors:** 3387

## Overall (stacked horizons)

| Rule | low recall | zero recall | precision | F1 |
|------|----------:|------------:|----------:|---:|
| rule_1_ptf_lag_1_le_50 | 0.3077 | 0.3115 | 0.3133 | 0.3105 |
| rule_2_ptf_lag_1_le_100 | 0.3245 | 0.3287 | 0.3128 | 0.3185 |
| rule_3_ptf_zero_ratio_24_gt_0 | 0.7852 | 0.7937 | 0.2379 | 0.3651 |
| rule_4_ptf_low_ratio_24_gt_0 | 0.8063 | 0.8116 | 0.2389 | 0.3686 |
| rule_5_ptf_zero_ratio_168_gt_0.05 | 0.8537 | 0.8511 | 0.2159 | 0.3447 |
| rule_6_zero_price_risk_proxy_eq_1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| rule_7_ren_high_therm_low | 0.9435 | 0.9436 | 0.1639 | 0.2793 |
| combined_any_rule | 0.9726 | 0.9714 | 0.1567 | 0.2699 |

## Per-horizon low recall (h1, h6, h12, h24)

| Rule | h1 | h6 | h12 | h24 |
|------|---:|---:|----:|----:|
| rule_1_ptf_lag_1_le_50 | 0.724 | 0.223 | 0.061 | 0.559 |
| rule_2_ptf_lag_1_le_100 | 0.752 | 0.251 | 0.064 | 0.594 |
| rule_3_ptf_zero_ratio_24_gt_0 | 0.907 | 0.792 | 0.771 | 0.757 |
| rule_4_ptf_low_ratio_24_gt_0 | 0.927 | 0.817 | 0.793 | 0.774 |
| rule_5_ptf_zero_ratio_168_gt_0.05 | 0.870 | 0.851 | 0.854 | 0.850 |
| rule_6_zero_price_risk_proxy_eq_1 | 0.000 | 0.000 | 0.000 | 0.000 |
| rule_7_ren_high_therm_low | 1.000 | 0.994 | 0.931 | 0.967 |
| combined_any_rule | 1.000 | 1.000 | 0.967 | 0.978 |

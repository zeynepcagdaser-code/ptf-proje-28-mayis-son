# Supply-Demand Curve Analysis

Generated: `2026-05-31T18:48:04.245188+00:00`

## Data Availability

- Available: `True`
- Rows: `56208`
- Coverage: `2020-01-01 00:00:00+03:00 -> 2026-05-31 23:00:00+03:00`
- PTF zero hours: `439`
- PTF low-price hours: `944`
- PTF tight hours: `29876`
- PTF spike hours: `1704`

## Key Correlations

- supply_gap vs price: `-0.12137345479744135`
- bid_stack_density vs price: `-0.05774207233795052`
- offer_stack_density vs price: `0.04960380433565748`
- oversupply_curve_pressure vs price: `0.04201892102548087`
- marginality_risk_score vs price: `0.183191252223629`
- curve_slope_near_ptf vs price: `0.022162553666010647`

## Regime Summary

- `negative_zero_pressure`: rows=944, mean_price=9.71, mean_supply_gap=-1767.41, mean_cap_risk_from_curve=0.0
- `normal`: rows=23684, mean_price=550.17, mean_supply_gap=-3505.33, mean_cap_risk_from_curve=0.0
- `spike_cap`: rows=1704, mean_price=4432.54, mean_supply_gap=-5779.72, mean_cap_risk_from_curve=0.9070399574530517
- `tight`: rows=29876, mean_price=2540.88, mean_supply_gap=-3918.02, mean_cap_risk_from_curve=0.010000480318650423

## Slice Diagnostics

- `ptf_zero`: {'rows': 439, 'mean_supply_gap': -158.11938496583122, 'mean_oversupply_curve_pressure': 0.05532317573123482, 'mean_bid_stack_density': 1.0064859913599649}
- `ptf_low`: {'rows': 944, 'mean_supply_gap': -1767.4103389830507, 'mean_oversupply_curve_pressure': 0.08888505051442112, 'mean_bid_stack_density': 0.9427221088959663}
- `ptf_tight`: {'rows': 29876, 'mean_supply_gap': -3918.0169681349576, 'mean_marginality_risk_score': 0.7077429087363523, 'mean_curve_slope_near_ptf': 0.2248663532650872}
- `ptf_spike`: {'rows': 1704, 'mean_supply_gap': -5779.718474178403, 'mean_marginality_risk_score': 0.8597858504571328, 'mean_cap_risk_from_curve': 0.9070399574530517}

## Interpretation

These features are curve proxies built from hourly supply, load, and finalized PTF because raw EPİAŞ supply-demand curve files are not present in the repository. The strongest signal should be read as structural pressure, not literal bid-curve elasticity.

## Missing Raw Curve Note

No raw supply-demand curve snapshot file was found in the repo, so `clearing_price_proxy` is anchored to finalized PTF and the slope/elasticity terms are derived proxies.

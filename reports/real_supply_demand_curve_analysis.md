# Real Supply-Demand Curve Analysis

Generated: `2026-05-31T18:57:57.475909+00:00`

## Availability

- Available: `True`
- Mode: `proxy_fallback`
- Rows: `56208`
- Coverage: `2020-01-01 00:00:00+03:00 -> 2026-05-31 23:00:00+03:00`

## Key Signals

- cap_risk_from_curve_vs_price: `0.44390162269632727`
- marginality_risk_vs_price: `0.183191252223629`
- oversupply_pressure_vs_price: `0.04201892102548087`
- supply_gap_score_vs_price: `-0.057742072337950485`
- curve_break_prob_vs_price: `0.4153203315084472`

## Regime Slices

- `ptf_zero`: {'rows': 439, 'mean_oversupply_curve_pressure': 0.05532317573123482, 'mean_low_price_pressure': 1.0}
- `ptf_spike`: {'rows': 1704, 'mean_cap_risk_from_curve': 0.9070399574530517, 'mean_marginality_risk_score': 0.8597858504571328}
- `ptf_tight`: {'rows': 29876, 'mean_curve_convexity_score': 384.6430412371134, 'mean_clearing_fragility_score': 0.13014217952174933}
- `ptf_normal`: {'rows': 24189, 'mean_slope_near_clearing': 0.10160534109773874}

## Transition Slices

- `normal_to_spike_proxy`: {'rows': 8, 'mean_curve_break_probability': 1.0}
- `tight_to_zero_proxy`: {'rows': 0, 'mean_oversupply_curve_pressure': None}

## Feature Means

- `clearing_price_proxy`: `1716.9047455878167`
- `clearing_volume_proxy`: `3870.1384130372903`
- `slope_near_clearing`: `0.1656907664715046`
- `local_curve_density`: `0.9010605586120223`
- `local_elasticity`: `1.116685399284713`
- `zero_price_supply_excess`: `555.6429100400948`
- `low_price_pressure`: `0.015797233489894678`
- `oversupply_mass_below_100`: `555.6429100400948`
- `renewable_oversupply_zone`: `0.10069321124072338`
- `curve_convexity_score`: `280.8882657272986`
- `cap_risk_from_curve`: `0.03281330838136919`
- `supply_gap_above_clearing`: `3870.1384130372903`
- `steepness_above_ptf`: `0.1656907664715046`
- `marginality_jump_score`: `190.96909372331342`
- `demand_curve_steepness`: `0.1656907664715046`
- `demand_elasticity`: `1.116685399284713`
- `demand_cliff_score`: `0.2989515072587532`
- `offer_stack_density`: `1.116685399284713`
- `bid_stack_density`: `0.9010605586120223`
- `supply_concentration_score`: `0.8643050365210707`
- `clearing_fragility_score`: `0.11883043899022828`
- `volume_needed_for_500TL_move`: `3870.1384130372903`
- `volume_needed_for_1000TL_move`: `3870.1384130372903`
- `curve_break_probability`: `0.3831816271705096`
- `imbalance_pressure_proxy`: `0.1356949634789293`

## Notes

This pipeline prefers raw EPİAŞ GÖP curve tables when available. In this repository, no raw curve files were discovered under `data/raw`, so the current run builds compact market microstructure features from the existing hourly curve proxy layer and the finalized market series. The outputs remain leakage-safe because only anchor-time observable hourly data are used.

Debug plots written to `reports/curve_debug_examples/`.

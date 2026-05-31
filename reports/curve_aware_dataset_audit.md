# Curve Aware Training Dataset Audit

- Rows: `336`
- Coverage start: `2026-05-19 00:00:00`
- Coverage end: `2026-06-01 23:00:00`

## Leakage Checks

- Uses same-day curve as feature: `False`
- Target PTF is future only: `True`
- Contains realized same-hour future market features: `False`

## Target Distribution

- `normal`: `178`
- `negative_zero_pressure`: `122`
- `spike_cap`: `19`
- `tight`: `17`

## Missing Ratios

- `analyst_confidence_score`: `0.071`
- `analyst_expected_regime`: `0.071`
- `analyst_persistence_break_score`: `0.071`
- `analyst_spike_score`: `0.071`
- `analyst_tight_score`: `0.071`
- `analyst_zero_score`: `0.071`
- `cheap_supply_pressure`: `0.071`
- `coal_share`: `0.071`
- `demand_weakness_score`: `0.071`
- `gas_marginality_proxy`: `0.071`
- `gas_off_flag`: `0.071`
- `gas_share`: `0.071`
- `gas_share_of_generation`: `0.071`
- `hour`: `0.000`
- `hydro_displacement_score`: `0.071`
- `hydro_high_flag`: `0.071`
- `hydro_share`: `0.071`
- `hydro_share_of_generation`: `0.071`
- `kgup_renewable_mw`: `0.071`
- `kgup_solar_mw`: `0.071`
- `kgup_total`: `0.071`
- `kgup_wind_mw`: `0.071`
- `load_deviation_from_monthly_norm`: `0.071`
- `load_deviation_from_weekly_norm`: `0.071`
- `load_forecast`: `0.071`
- `load_vs_renewable_balance`: `0.071`
- `low_demand_flag`: `0.071`
- `month`: `0.000`
- `prev_day_cap_risk_score`: `0.000`
- `prev_day_curve_fragility_score`: `0.000`
- `prev_day_elasticity_near_clearing`: `0.000`
- `prev_day_oversupply_pressure`: `0.000`
- `prev_day_slope_near_clearing`: `0.000`
- `prev_day_spike_pressure_from_curve`: `0.000`
- `prev_day_volume_needed_for_100TL_move`: `0.000`
- `prev_day_volume_needed_for_500TL_move`: `0.000`
- `prev_day_zero_pressure_from_curve`: `0.000`
- `previous_day_regime`: `0.000`
- `ptf_lag_24`: `0.000`
- `reconstruction_confidence`: `0.000`
- `renewable_minus_gas_shift`: `0.071`
- `renewable_share_high_flag`: `0.071`
- `renewable_share_of_generation`: `0.071`
- `residual_load_forecast`: `0.071`
- `solar_share`: `0.071`
- `thermal_share`: `0.071`
- `weekday`: `0.000`
- `weekend`: `0.000`
- `wind_share`: `0.071`
- `zero_price_pressure_score`: `0.071`

## Feature Correlations with Target PTF

- `analyst_confidence_score`: `-0.514479544077469`
- `analyst_persistence_break_score`: `0.11907349300830679`
- `analyst_spike_score`: `0.506933115619984`
- `analyst_tight_score`: `0.6138172431695513`
- `analyst_zero_score`: `-0.5723367423417048`
- `cheap_supply_pressure`: `-0.06949833257520659`
- `coal_share`: `0.31298845994282315`
- `demand_weakness_score`: `-0.32632032994941335`
- `gas_marginality_proxy`: `0.39392684788514387`
- `gas_share`: `0.6108247304716738`
- `gas_share_of_generation`: `0.6108247304716738`
- `hydro_displacement_score`: `0.3421673965108277`
- `hydro_share`: `0.4059517700766382`
- `kgup_renewable_mw`: `-0.18028393217287933`
- `kgup_solar_mw`: `-0.34096964127534846`
- `kgup_total`: `0.222551407366396`
- `kgup_wind_mw`: `0.025265450654798967`
- `load_forecast`: `0.36688043537978887`
- `load_vs_renewable_balance`: `0.4765035460848805`
- `prev_day_cap_risk_score`: `0.6572190539505615`
- `prev_day_curve_fragility_score`: `-0.0645795779942522`
- `prev_day_elasticity_near_clearing`: `-0.06876539146539123`
- `prev_day_oversupply_pressure`: `-0.2581967046636219`
- `prev_day_slope_near_clearing`: `-0.05018635957206806`
- `prev_day_volume_needed_for_100TL_move`: `-0.4159260489315823`
- `prev_day_volume_needed_for_500TL_move`: `-0.5056298959794185`
- `ptf_lag_24`: `0.7686517119343221`
- `reconstruction_confidence`: `0.383477960925134`
- `renewable_share_of_generation`: `-0.5117419053659217`
- `residual_load_forecast`: `0.5525184307747638`
- `solar_share`: `-0.36149854103152707`
- `thermal_share`: `0.528337313162821`
- `wind_share`: `-0.0742920037320636`
- `zero_price_pressure_score`: `-0.30425888841364573`

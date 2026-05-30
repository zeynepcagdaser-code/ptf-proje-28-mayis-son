# LSTM Feature Dataset Report

- **Generated (UTC):** 2026-05-30T09:36:58.118297+00:00
- **Output:** `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/features/lstm_tree_micro_v1.parquet`
- **Source:** `data/master/master_hourly_v1.parquet`

## Training format (reference)

- **Input window:** None hours
- **Output horizon:** None hours
- **Index mapping:** `None`

## Row counts

- Master rows: 56208
- Dropped (missing targets): 24
- Dropped (insufficient history for lags/rolls): 168
- **Final rows:** 56016

## Features & targets

- Feature columns: 107
- Target columns: 24
- ts_hour range: 2020-01-08 00:00:00+03:00 → 2026-05-29 23:00:00+03:00

## Split counts

| Split | Rows |
|-------|-----:|
| train | 43680 |
| validation | 8760 |
| test | 3576 |

## Missing features (before ffill, % > 0)

- `ptf_hours_since_spike`: 41.6667%
- `ptf_hours_since_zero`: 4.0759%
- `spread_zscore_24h`: 1.0003%
- `wind_generation_mean_lag_168`: 0.8116%
- `wind_generation_mean_lag_48`: 0.5980%
- `wind_generation_mean_lag_24`: 0.5553%
- `ptf_vol_ratio_24_168`: 0.4610%
- `ptf_lag_168`: 0.2990%
- `ptf_roll_mean_168`: 0.2990%
- `ptf_roll_std_168`: 0.2990%
- `smf_ptf_spread_lag_168`: 0.2990%
- `gen_total_lag_168`: 0.2990%
- `cons_consumption_lag_168`: 0.2990%
- `smf_lag_168`: 0.2990%
- `yal_yat_net_lag_168`: 0.2990%
- `yal_yat_upRegulationDelivered_lag_168`: 0.2990%
- `yal_yat_downRegulationDelivered_lag_168`: 0.2990%
- `yal_yat_net_stress_168h`: 0.2990%
- `yal_up_stress_168h`: 0.2990%
- `yal_down_stress_168h`: 0.2990%
- `outage_fault_mw_loss_sum`: 0.2563%
- `outage_fault_mw_loss_max`: 0.2563%
- `outage_fault_operator_power_sum`: 0.2563%
- `ptf_zscore_24h`: 0.1459%
- `wind_forecast_ramp_24h`: 0.1388%
- `outage_maint_capacity_sum`: 0.1353%
- `outage_maint_operator_power_sum`: 0.1353%
- `supply_gap_change_24h`: 0.1282%
- `ptf_lag_48`: 0.0854%
- `gen_total_lag_48`: 0.0854%
- `cons_consumption_lag_48`: 0.0854%
- `smf_lag_48`: 0.0854%
- `yal_yat_net_lag_48`: 0.0854%
- `yal_yat_upRegulationDelivered_lag_48`: 0.0854%
- `yal_yat_downRegulationDelivered_lag_48`: 0.0854%
- `ptf_zero_share_168h`: 0.0837%
- `ptf_spike_share_168h`: 0.0837%
- `supply_gap_zscore_24h`: 0.0819%
- `wind_forecast_std`: 0.0765%
- `wind_forecast_ramp_1h`: 0.0534%
- `spread_change_24h`: 0.0516%
- `wind_forecast_share`: 0.0481%
- `wind_quarter2_mean`: 0.0481%
- `wind_quarter3_mean`: 0.0481%
- `wind_quarter4_mean`: 0.0481%
- `wind_forecast_mean`: 0.0481%
- `wind_forecast_min`: 0.0481%
- `wind_forecast_max`: 0.0481%
- `supply_gap_change_1h`: 0.0463%
- `ptf_lag_24`: 0.0427%
- `ptf_roll_mean_24`: 0.0427%
- `ptf_roll_std_24`: 0.0427%
- `smf_ptf_spread_lag_24`: 0.0427%
- `kgup_total_minus_load`: 0.0427%
- `load_lep`: 0.0427%
- `gen_total_lag_24`: 0.0427%
- `cons_consumption_lag_24`: 0.0427%
- `smf_lag_24`: 0.0427%
- `yal_yat_net_lag_24`: 0.0427%
- `yal_yat_upRegulationDelivered_lag_24`: 0.0427%
- `yal_yat_downRegulationDelivered_lag_24`: 0.0427%
- `ptf_return_24h`: 0.0427%
- `ptf_momentum_24h`: 0.0427%
- `ptf_range_24h`: 0.0427%
- `supply_gap`: 0.0427%
- `renewable_share_change_24h`: 0.0427%
- `yal_yat_net_stress_24h`: 0.0427%
- `yal_up_stress_24h`: 0.0427%
- `yal_down_stress_24h`: 0.0427%
- `hour_x_ptf_lag24`: 0.0427%
- `hour_x_spread_lag24`: 0.0427%
- `weekend_x_ptf_vol`: 0.0427%
- `ptf_zero_share_24h`: 0.0196%
- `ptf_spike_share_24h`: 0.0196%
- `spread_nowcast_lag1`: 0.0107%
- `ptf_return_1h`: 0.0036%
- `ptf_lag_1`: 0.0018%

## Missing features (after ffill limit=2, % > 0)

- `ptf_hours_since_spike`: 41.6667%
- `ptf_hours_since_zero`: 4.0759%
- `spread_zscore_24h`: 0.7903%
- `wind_generation_mean_lag_168`: 0.7244%
- `wind_generation_mean_lag_48`: 0.5108%
- `wind_generation_mean_lag_24`: 0.4681%
- `ptf_vol_ratio_24_168`: 0.4574%
- `ptf_lag_168`: 0.2990%
- `ptf_roll_mean_168`: 0.2990%
- `ptf_roll_std_168`: 0.2990%
- `smf_ptf_spread_lag_168`: 0.2990%
- `gen_total_lag_168`: 0.2990%
- `cons_consumption_lag_168`: 0.2990%
- `smf_lag_168`: 0.2990%
- `yal_yat_net_lag_168`: 0.2990%
- `yal_yat_upRegulationDelivered_lag_168`: 0.2990%
- `yal_yat_downRegulationDelivered_lag_168`: 0.2990%
- `yal_yat_net_stress_168h`: 0.2990%
- `yal_up_stress_168h`: 0.2990%
- `yal_down_stress_168h`: 0.2990%
- `outage_fault_mw_loss_sum`: 0.2492%
- `outage_fault_mw_loss_max`: 0.2492%
- `outage_fault_operator_power_sum`: 0.2492%
- `ptf_zscore_24h`: 0.1317%
- `outage_maint_capacity_sum`: 0.1282%
- `outage_maint_operator_power_sum`: 0.1282%
- `supply_gap_change_24h`: 0.1246%
- `wind_forecast_ramp_24h`: 0.1246%
- `ptf_lag_48`: 0.0854%
- `gen_total_lag_48`: 0.0854%
- `cons_consumption_lag_48`: 0.0854%
- `smf_lag_48`: 0.0854%
- `yal_yat_net_lag_48`: 0.0854%
- `yal_yat_upRegulationDelivered_lag_48`: 0.0854%
- `yal_yat_downRegulationDelivered_lag_48`: 0.0854%
- `ptf_zero_share_168h`: 0.0837%
- `ptf_spike_share_168h`: 0.0837%
- `supply_gap_zscore_24h`: 0.0783%
- `spread_change_24h`: 0.0481%
- `wind_forecast_ramp_1h`: 0.0463%
- `ptf_lag_24`: 0.0427%
- `ptf_roll_mean_24`: 0.0427%
- `ptf_roll_std_24`: 0.0427%
- `smf_ptf_spread_lag_24`: 0.0427%
- `gen_total_lag_24`: 0.0427%
- `cons_consumption_lag_24`: 0.0427%
- `smf_lag_24`: 0.0427%
- `yal_yat_net_lag_24`: 0.0427%
- `yal_yat_upRegulationDelivered_lag_24`: 0.0427%
- `yal_yat_downRegulationDelivered_lag_24`: 0.0427%
- `ptf_return_24h`: 0.0427%
- `ptf_momentum_24h`: 0.0427%
- `ptf_range_24h`: 0.0427%
- `supply_gap_change_1h`: 0.0427%
- `renewable_share_change_24h`: 0.0427%
- `yal_yat_net_stress_24h`: 0.0427%
- `yal_up_stress_24h`: 0.0427%
- `yal_down_stress_24h`: 0.0427%
- `hour_x_ptf_lag24`: 0.0427%
- `hour_x_spread_lag24`: 0.0427%
- `weekend_x_ptf_vol`: 0.0427%
- `wind_forecast_share`: 0.0409%
- `wind_quarter2_mean`: 0.0409%
- `wind_quarter3_mean`: 0.0409%
- `wind_quarter4_mean`: 0.0409%
- `wind_forecast_mean`: 0.0409%
- `wind_forecast_min`: 0.0409%
- `wind_forecast_max`: 0.0409%
- `wind_forecast_std`: 0.0409%
- `kgup_total_minus_load`: 0.0392%
- `load_lep`: 0.0392%
- `supply_gap`: 0.0392%
- `ptf_zero_share_24h`: 0.0196%
- `ptf_spike_share_24h`: 0.0196%
- `spread_nowcast_lag1`: 0.0071%
- `ptf_return_1h`: 0.0036%
- `ptf_lag_1`: 0.0018%

## Leakage checklist

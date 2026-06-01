# Feature Inventory Report

- Dataset: `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/features/lstm_next24_v1.parquet`
- Rows: 56016
- Feature count: 102

## Counts by Family

- ptf_lag_rolling: 18
- calendar_holiday: 9
- load_demand: 2
- kgup_source_mix: 16
- renewable_pressure: 6
- thermal_price_setting: 5
- smf_yal_yat_lagged: 24
- wind_forecast: 8
- outage: 8
- fuel_currency: 0
- cap_imbalance: 2
- yekdem_merchant_proxy: 0
- low_zero_price_risk: 4
- other: 0

## Counts by Recommended Usage

- main_regression: 72
- low_price_classifier: 4
- risk_dashboard_only: 2
- exclude: 24

## Inventory

| Feature | Family | Sources | Availability | Leakage | Suggested | Null % |
|---------|--------|---------|--------------|---------|----------|-------:|
| `ptf_lag_1` | ptf_lag_rolling | ptf_price | lag_required | low | main_regression | 0.00 |
| `ptf_lag_168` | ptf_lag_rolling | ptf_price | lag_required | low | main_regression | 0.00 |
| `ptf_lag_168h` | ptf_lag_rolling | ptf_price | lag_required | low | main_regression | 0.00 |
| `ptf_lag_1h` | ptf_lag_rolling | ptf_price | lag_required | low | main_regression | 0.00 |
| `ptf_lag_2` | ptf_lag_rolling | ptf_price | lag_required | low | main_regression | 0.00 |
| `ptf_lag_24` | ptf_lag_rolling | ptf_price | lag_required | low | main_regression | 0.00 |
| `ptf_lag_24h` | ptf_lag_rolling | ptf_price | lag_required | low | main_regression | 0.00 |
| `ptf_lag_2h` | ptf_lag_rolling | ptf_price | lag_required | low | main_regression | 0.00 |
| `ptf_lag_3` | ptf_lag_rolling | ptf_price | lag_required | low | main_regression | 0.00 |
| `ptf_lag_3h` | ptf_lag_rolling | ptf_price | lag_required | low | main_regression | 0.00 |
| `ptf_lag_48` | ptf_lag_rolling | ptf_price | lag_required | low | main_regression | 0.00 |
| `ptf_roll_mean_168` | ptf_lag_rolling | ptf_price | lag_required | low | main_regression | 0.00 |
| `ptf_roll_mean_24` | ptf_lag_rolling | ptf_price | lag_required | low | main_regression | 0.00 |
| `ptf_roll_std_168` | ptf_lag_rolling | ptf_price | lag_required | low | main_regression | 0.00 |
| `ptf_roll_std_24` | ptf_lag_rolling | ptf_price | lag_required | low | main_regression | 0.00 |
| `ptf_rolling_mean_168h` | ptf_lag_rolling | ptf_price | lag_required | low | main_regression | 0.00 |
| `ptf_rolling_mean_24h` | ptf_lag_rolling | ptf_price | lag_required | low | main_regression | 0.00 |
| `ptf_rolling_std_24h` | ptf_lag_rolling | ptf_price | lag_required | low | main_regression | 0.00 |
| `dow_cos` | calendar_holiday | ts_hour | same_hour_ok | low | main_regression | 0.00 |
| `dow_sin` | calendar_holiday | ts_hour | same_hour_ok | low | main_regression | 0.00 |
| `hour_cos` | calendar_holiday | ts_hour | same_hour_ok | low | main_regression | 0.00 |
| `hour_sin` | calendar_holiday | ts_hour | same_hour_ok | low | main_regression | 0.00 |
| `is_holiday_or_weekend` | calendar_holiday | ts_hour, holidays.Turkey() | same_hour_ok | low | main_regression | 0.00 |
| `is_holiday_tr` | calendar_holiday | ts_hour, holidays.Turkey() | same_hour_ok | low | main_regression | 0.00 |
| `is_weekend` | calendar_holiday | ts_hour | same_hour_ok | low | main_regression | 0.00 |
| `month_cos` | calendar_holiday | ts_hour | same_hour_ok | low | main_regression | 0.00 |
| `month_sin` | calendar_holiday | ts_hour | same_hour_ok | low | main_regression | 0.00 |
| `kgup_total_minus_load` | load_demand | kgup_total_minus_load | same_hour_ok | low | main_regression | 0.04 |
| `load_lep` | load_demand | load_lep | same_hour_ok | low | main_regression | 0.04 |
| `kgup_akarsu` | kgup_source_mix | kgup_akarsu | same_hour_ok | low | main_regression | 0.00 |
| `kgup_barajli` | kgup_source_mix | kgup_barajli | same_hour_ok | low | main_regression | 0.00 |
| `kgup_biokutle` | kgup_source_mix | kgup_biokutle | same_hour_ok | low | main_regression | 0.00 |
| `kgup_diger` | kgup_source_mix | kgup_diger | same_hour_ok | low | main_regression | 0.00 |
| `kgup_dogalgaz` | kgup_source_mix | kgup_dogalgaz | same_hour_ok | low | main_regression | 0.00 |
| `kgup_fuelOil` | kgup_source_mix | kgup_fuelOil | same_hour_ok | low | main_regression | 0.00 |
| `kgup_gunes` | kgup_source_mix | kgup_gunes | same_hour_ok | low | main_regression | 0.00 |
| `kgup_ithalKomur` | kgup_source_mix | kgup_ithalKomur | same_hour_ok | low | main_regression | 0.00 |
| `kgup_jeotermal` | kgup_source_mix | kgup_jeotermal | same_hour_ok | low | main_regression | 0.00 |
| `kgup_linyit` | kgup_source_mix | kgup_linyit | same_hour_ok | low | main_regression | 0.00 |
| `kgup_nafta` | kgup_source_mix | kgup_nafta | same_hour_ok | low | main_regression | 0.00 |
| `kgup_renewable_share` | kgup_source_mix | kgup_renewable_share | same_hour_ok | low | main_regression | 0.00 |
| `kgup_ruzgar` | kgup_source_mix | kgup_ruzgar | same_hour_ok | low | main_regression | 0.00 |
| `kgup_tasKomur` | kgup_source_mix | kgup_tasKomur | same_hour_ok | low | main_regression | 0.00 |
| `kgup_thermal_share` | kgup_source_mix | kgup_thermal_share | same_hour_ok | low | main_regression | 0.00 |
| `kgup_toplam` | kgup_source_mix | kgup_toplam | same_hour_ok | low | main_regression | 0.00 |
| `hydro_share` | renewable_pressure | kgup_toplam, kgup_ruzgar, kgup_gunes, kgup_barajli, kgup_akarsu, kgup_biokutle, kgup_jeotermal | same_hour_ok | low | main_regression | 0.00 |
| `renewable_pressure` | renewable_pressure | kgup_toplam, kgup_ruzgar, kgup_gunes, kgup_barajli, kgup_akarsu, kgup_biokutle, kgup_jeotermal | same_hour_ok | low | main_regression | 0.00 |
| `renewable_suppression_pressure` | renewable_pressure | kgup_toplam, kgup_ruzgar, kgup_gunes, kgup_barajli, kgup_akarsu, kgup_biokutle, kgup_jeotermal | same_hour_ok | low | main_regression | 0.00 |
| `res_share` | renewable_pressure | kgup_toplam, kgup_ruzgar, kgup_gunes, kgup_barajli, kgup_akarsu, kgup_biokutle, kgup_jeotermal | same_hour_ok | low | main_regression | 0.00 |
| `solar_share` | renewable_pressure | kgup_toplam, kgup_ruzgar, kgup_gunes, kgup_barajli, kgup_akarsu, kgup_biokutle, kgup_jeotermal | same_hour_ok | low | main_regression | 0.00 |
| `wind_forecast_share` | renewable_pressure | wind_* (master) | same_hour_ok | low | main_regression | 0.04 |
| `coal_share` | thermal_price_setting | kgup_toplam, kgup_dogalgaz, kgup_ithalKomur, kgup_linyit, kgup_tasKomur | same_hour_ok | low | main_regression | 0.00 |
| `gas_coal_balance` | thermal_price_setting | kgup_toplam, kgup_dogalgaz, kgup_ithalKomur, kgup_linyit, kgup_tasKomur | same_hour_ok | low | main_regression | 0.00 |
| `gas_coal_competition_index` | thermal_price_setting | kgup_toplam, kgup_dogalgaz, kgup_ithalKomur, kgup_linyit, kgup_tasKomur | same_hour_ok | low | main_regression | 0.00 |
| `gas_share` | thermal_price_setting | kgup_toplam, kgup_dogalgaz, kgup_ithalKomur, kgup_linyit, kgup_tasKomur | same_hour_ok | low | main_regression | 0.00 |
| `thermal_price_setting_share` | thermal_price_setting | kgup_toplam, kgup_dogalgaz, kgup_ithalKomur, kgup_linyit, kgup_tasKomur, kgup_ruzgar, kgup_gunes, kgup_barajli, kgup_akarsu, kgup_biokutle, kgup_jeotermal | same_hour_ok | low | main_regression | 0.00 |
| `cons_consumption_lag_168` | smf_yal_yat_lagged | - | lag_required | low | exclude | 0.00 |
| `cons_consumption_lag_24` | smf_yal_yat_lagged | - | lag_required | low | exclude | 0.00 |
| `cons_consumption_lag_48` | smf_yal_yat_lagged | - | lag_required | low | exclude | 0.00 |
| `gen_total_lag_168` | smf_yal_yat_lagged | - | lag_required | low | exclude | 0.00 |
| `gen_total_lag_24` | smf_yal_yat_lagged | - | lag_required | low | exclude | 0.00 |
| `gen_total_lag_48` | smf_yal_yat_lagged | - | lag_required | low | exclude | 0.00 |
| `smf_lag_168` | smf_yal_yat_lagged | smf_systemMarginalPrice, ptf_price | lag_required | low | exclude | 0.00 |
| `smf_lag_24` | smf_yal_yat_lagged | smf_systemMarginalPrice, ptf_price | lag_required | low | exclude | 0.00 |
| `smf_lag_48` | smf_yal_yat_lagged | smf_systemMarginalPrice, ptf_price | lag_required | low | exclude | 0.00 |
| `smf_ptf_spread_lag_168` | smf_yal_yat_lagged | smf_systemMarginalPrice, ptf_price | lag_required | low | exclude | 0.00 |
| `smf_ptf_spread_lag_24` | smf_yal_yat_lagged | smf_systemMarginalPrice, ptf_price | lag_required | low | exclude | 0.00 |
| `smf_to_cap_ratio` | smf_yal_yat_lagged | smf_systemMarginalPrice, ptf_price | lag_required | low | exclude | 0.00 |
| `wind_generation_mean_lag_168` | smf_yal_yat_lagged | wind_* (master) | lag_required | low | exclude | 0.43 |
| `wind_generation_mean_lag_24` | smf_yal_yat_lagged | wind_* (master) | lag_required | low | exclude | 0.43 |
| `wind_generation_mean_lag_48` | smf_yal_yat_lagged | wind_* (master) | lag_required | low | exclude | 0.43 |
| `yal_yat_downRegulationDelivered_lag_168` | smf_yal_yat_lagged | yal_yat_* (master) | lag_required | low | exclude | 0.00 |
| `yal_yat_downRegulationDelivered_lag_24` | smf_yal_yat_lagged | yal_yat_* (master) | lag_required | low | exclude | 0.00 |
| `yal_yat_downRegulationDelivered_lag_48` | smf_yal_yat_lagged | yal_yat_* (master) | lag_required | low | exclude | 0.00 |
| `yal_yat_net_lag_168` | smf_yal_yat_lagged | yal_yat_* (master) | lag_required | low | exclude | 0.00 |
| `yal_yat_net_lag_24` | smf_yal_yat_lagged | yal_yat_* (master) | lag_required | low | exclude | 0.00 |
| `yal_yat_net_lag_48` | smf_yal_yat_lagged | yal_yat_* (master) | lag_required | low | exclude | 0.00 |
| `yal_yat_upRegulationDelivered_lag_168` | smf_yal_yat_lagged | yal_yat_* (master) | lag_required | low | exclude | 0.00 |
| `yal_yat_upRegulationDelivered_lag_24` | smf_yal_yat_lagged | yal_yat_* (master) | lag_required | low | exclude | 0.00 |
| `yal_yat_upRegulationDelivered_lag_48` | smf_yal_yat_lagged | yal_yat_* (master) | lag_required | low | exclude | 0.00 |
| `wind_forecast_max` | wind_forecast | wind_* (master) | same_hour_ok | low | main_regression | 0.04 |
| `wind_forecast_mean` | wind_forecast | wind_* (master) | same_hour_ok | low | main_regression | 0.04 |
| `wind_forecast_min` | wind_forecast | wind_* (master) | same_hour_ok | low | main_regression | 0.04 |
| `wind_forecast_std` | wind_forecast | wind_* (master) | same_hour_ok | low | main_regression | 0.04 |
| `wind_quarter1_mean` | wind_forecast | wind_* (master) | same_hour_ok | low | main_regression | 0.00 |
| `wind_quarter2_mean` | wind_forecast | wind_* (master) | same_hour_ok | low | main_regression | 0.04 |
| `wind_quarter3_mean` | wind_forecast | wind_* (master) | same_hour_ok | low | main_regression | 0.04 |
| `wind_quarter4_mean` | wind_forecast | wind_* (master) | same_hour_ok | low | main_regression | 0.04 |
| `outage_event_rows` | outage | outages_* (master) | same_hour_ok | low | main_regression | 0.00 |
| `outage_fault_event_count` | outage | outages_* (master) | same_hour_ok | low | main_regression | 0.00 |
| `outage_fault_mw_loss_max` | outage | outages_* (master) | same_hour_ok | low | main_regression | 0.25 |
| `outage_fault_mw_loss_sum` | outage | outages_* (master) | same_hour_ok | low | main_regression | 0.25 |
| `outage_fault_operator_power_sum` | outage | outages_* (master) | same_hour_ok | low | main_regression | 0.25 |
| `outage_maint_capacity_sum` | outage | outages_* (master) | same_hour_ok | low | main_regression | 0.00 |
| `outage_maint_event_count` | outage | outages_* (master) | same_hour_ok | low | main_regression | 0.00 |
| `outage_maint_operator_power_sum` | outage | outages_* (master) | same_hour_ok | low | main_regression | 0.00 |
| `price_cap` | cap_imbalance | engineered_constant | same_hour_ok | medium | risk_dashboard_only | 0.00 |
| `ptf_to_cap_ratio` | cap_imbalance | ptf_price | same_hour_ok | medium | risk_dashboard_only | 0.00 |
| `holiday_low_load_flag` | low_zero_price_risk | load_lep, ts_hour (holiday/weekend flag) | same_hour_ok | low | low_price_classifier | 0.00 |
| `low_load_flag` | low_zero_price_risk | load_lep | same_hour_ok | low | low_price_classifier | 0.00 |
| `solar_peak_hour_flag` | low_zero_price_risk | ts_hour | same_hour_ok | low | low_price_classifier | 0.00 |
| `zero_price_risk_proxy` | low_zero_price_risk | kgup_* (renewable/gas shares), load_lep, ts_hour (holiday/weekend flag) | same_hour_ok | low | low_price_classifier | 0.00 |

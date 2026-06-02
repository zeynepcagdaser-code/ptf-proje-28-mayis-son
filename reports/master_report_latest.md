# Master Dataset Report

- **Generated (UTC):** 2026-06-02T07:13:44.178427+00:00
- **Output:** `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/master/master_hourly_v1.parquet`
- **Rows:** 56208 (spine: 56208, match: True)
- **Columns:** 89
- **ts_hour unique:** 56208
- **ts_hour range:** 2020-01-01 00:00:00+03:00 → 2026-05-30 23:00:00+03:00

## Columns by dataset

| Dataset | Column count |
|---------|-------------:|
| ptf | 10 |
| kgup | 15 |
| load_forecast | 1 |
| realtime_generation | 16 |
| real_consumption | 1 |
| smf | 3 |
| yal_yat | 7 |
| wind | 14 |
| outages | 8 |
| open_meteo_temperature | 2 |
| dam_price_independent_buy | 1 |
| dam_price_independent_sell | 1 |
| dam_bid_volume | 1 |
| dam_sell_offer_volume | 1 |
| dam_matched_volume | 2 |
| dam_block_buy_volume | 2 |
| grf_daily_reference_price | 4 |

## Availability summary

- **balancing:** 8 columns
- **forecast:** 9 columns
- **metadata:** 28 columns
- **outage_event:** 8 columns
- **planned:** 14 columns
- **realized:** 22 columns

## Missing values (% > 0)

| Column | Missing % | Availability |
|--------|----------:|----------------|
| `temp_2m` | 6.4048 | n/a |
| `apparent_temp` | 6.4048 | n/a |
| `wind_generation_mean` | 0.5586 | realized |
| `wind_generation_min` | 0.5586 | realized |
| `wind_generation_max` | 0.5586 | realized |
| `outage_fault_mw_loss_sum` | 0.2989 | outage_event |
| `outage_fault_mw_loss_max` | 0.2989 | outage_event |
| `outage_fault_operator_power_sum` | 0.2989 | outage_event |
| `outage_maint_capacity_sum` | 0.1352 | outage_event |
| `outage_maint_operator_power_sum` | 0.1352 | outage_event |
| `wind_forecast_std` | 0.0783 | forecast |
| `smf_systemMarginalPrice` | 0.0534 | balancing |
| `smf_is_systemMarginalPrice_zero` | 0.0534 | metadata |
| `smf_is_systemMarginalPrice_capped` | 0.0534 | metadata |
| `wind_quarter2_mean` | 0.0480 | forecast |
| `wind_quarter3_mean` | 0.0480 | forecast |
| `wind_quarter4_mean` | 0.0480 | forecast |
| `wind_forecast_mean` | 0.0480 | forecast |
| `wind_forecast_min` | 0.0480 | forecast |
| `wind_forecast_max` | 0.0480 | forecast |
| `gen_total` | 0.0463 | realized |
| `gen_naturalGas` | 0.0463 | realized |
| `gen_dammedHydro` | 0.0463 | realized |
| `gen_lignite` | 0.0463 | realized |
| `gen_river` | 0.0463 | realized |
| `gen_importCoal` | 0.0463 | realized |
| `gen_wind` | 0.0463 | realized |
| `gen_sun` | 0.0463 | realized |
| `gen_fueloil` | 0.0463 | realized |
| `gen_geothermal` | 0.0463 | realized |
| `gen_asphaltiteCoal` | 0.0463 | realized |
| `gen_blackCoal` | 0.0463 | realized |
| `gen_biomass` | 0.0463 | realized |
| `gen_importExport` | 0.0463 | realized |
| `gen_wasteheat` | 0.0463 | realized |
| `gen_was_sun_clipped` | 0.0463 | metadata |
| `cons_consumption` | 0.0463 | realized |
| `load_lep` | 0.0427 | forecast |

## Schema

| Column | dtype | Availability |
|--------|-------|----------------|
| `ts_hour` | datetime64[us, Europe/Istanbul] | metadata |
| `ptf_price` | float64 | realized |
| `ptf_priceUsd` | float64 | realized |
| `ptf_priceEur` | float64 | realized |
| `ptf_is_price_zero` | bool | metadata |
| `ptf_is_price_capped` | bool | metadata |
| `ptf_is_priceUsd_zero` | bool | metadata |
| `ptf_is_priceUsd_capped` | bool | metadata |
| `ptf_is_priceEur_zero` | bool | metadata |
| `ptf_is_priceEur_capped` | bool | metadata |
| `kgup_toplam` | float64 | planned |
| `kgup_dogalgaz` | float64 | planned |
| `kgup_ruzgar` | float64 | planned |
| `kgup_linyit` | float64 | planned |
| `kgup_tasKomur` | float64 | planned |
| `kgup_ithalKomur` | float64 | planned |
| `kgup_fuelOil` | float64 | planned |
| `kgup_jeotermal` | float64 | planned |
| `kgup_barajli` | float64 | planned |
| `kgup_nafta` | float64 | planned |
| `kgup_biokutle` | float64 | planned |
| `kgup_akarsu` | float64 | planned |
| `kgup_gunes` | float64 | planned |
| `kgup_diger` | float64 | planned |
| `kgup_source_type` | str | metadata |
| `load_lep` | float64 | forecast |
| `gen_total` | float64 | realized |
| `gen_naturalGas` | float64 | realized |
| `gen_dammedHydro` | float64 | realized |
| `gen_lignite` | float64 | realized |
| `gen_river` | float64 | realized |
| `gen_importCoal` | float64 | realized |
| `gen_wind` | float64 | realized |
| `gen_sun` | float64 | realized |
| `gen_fueloil` | float64 | realized |
| `gen_geothermal` | float64 | realized |
| `gen_asphaltiteCoal` | float64 | realized |
| `gen_blackCoal` | float64 | realized |
| `gen_biomass` | float64 | realized |
| `gen_importExport` | float64 | realized |
| `gen_wasteheat` | float64 | realized |
| `gen_was_sun_clipped` | object | metadata |
| `cons_consumption` | float64 | realized |
| `smf_systemMarginalPrice` | float64 | balancing |
| `smf_is_systemMarginalPrice_zero` | object | metadata |
| `smf_is_systemMarginalPrice_capped` | object | metadata |
| `yal_yat_upRegulationZeroCoded` | float64 | balancing |
| `yal_yat_upRegulationOneCoded` | float64 | balancing |
| `yal_yat_upRegulationDelivered` | float64 | balancing |
| `yal_yat_net` | float64 | balancing |
| `yal_yat_downRegulationZeroCoded` | float64 | balancing |
| `yal_yat_downRegulationOneCoded` | float64 | balancing |
| `yal_yat_downRegulationDelivered` | float64 | balancing |
| `wind_interval_count` | int64 | metadata |
| `wind_quarter1_mean` | float64 | forecast |
| `wind_quarter2_mean` | float64 | forecast |
| `wind_quarter3_mean` | float64 | forecast |
| `wind_quarter4_mean` | float64 | forecast |
| `wind_generation_mean` | float64 | realized |
| `wind_generation_min` | float64 | realized |
| `wind_generation_max` | float64 | realized |
| `wind_forecast_mean` | float64 | forecast |
| `wind_forecast_min` | float64 | forecast |
| `wind_forecast_max` | float64 | forecast |
| `wind_forecast_std` | float64 | forecast |
| `wind_is_partial_hour` | bool | metadata |
| `wind_was_generation_clipped` | bool | metadata |
| `outage_event_rows` | int64 | outage_event |
| `outage_fault_event_count` | int64 | outage_event |
| `outage_fault_mw_loss_sum` | float64 | outage_event |
| `outage_fault_mw_loss_max` | float64 | outage_event |
| `outage_fault_operator_power_sum` | float64 | outage_event |
| `outage_maint_event_count` | int64 | outage_event |
| `outage_maint_capacity_sum` | float64 | outage_event |
| `outage_maint_operator_power_sum` | float64 | outage_event |
| `temp_2m` | float64 | n/a |
| `apparent_temp` | float64 | n/a |
| `dam_price_independent_buy_mwh` | float64 | planned |
| `dam_price_independent_sell_mwh` | float64 | planned |
| `dam_bid_volume_mwh` | float64 | planned |
| `dam_sell_offer_volume_mwh` | float64 | planned |
| `dam_matched_buy_mwh` | float64 | planned |
| `dam_matched_sell_mwh` | float64 | planned |
| `dam_block_matched_buy_mwh` | float64 | planned |
| `dam_block_unmatched_buy_mwh` | float64 | planned |
| `grf_tl_1000sm3` | float64 | realized |
| `grf_usd_1000sm3` | float64 | realized |
| `grf_eur_mwh` | float64 | realized |
| `grf_usd_mmbtu` | float64 | realized |
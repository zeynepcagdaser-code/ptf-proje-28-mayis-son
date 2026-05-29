# Cleaning Report

- **Generated (UTC):** 2026-05-29T20:08:34.973788+00:00
- **Output directory:** `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/clean`

## Rules applied

- PTF zero prices are never removed, modified, or imputed.
- Price spikes are never removed.
- No interpolation, bfill, or centered rolling.
- All outputs use ts_hour in Europe/Istanbul.
- Wind: 10-minute → hourly aggregation (mean/min/max/std).
- Outages: event rows → hourly aggregates.

## Datasets

### ptf

- **Output:** `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/clean/ptf_hourly.parquet`
- **Rows in:** 56208
- **Rows out:** 56208
- **ts_hour range:** 2020-01-01 00:00:00+03:00 → 2026-05-30 23:00:00+03:00
- **ts_parse_fail:** 0
- **rows_after_ts_drop:** 56208
- **duplicate_key_rows:** 0
- **rows_after_dedupe:** 56208
- **duplicate_ts_hour_removed:** 0
- **dropped_columns:** []
- **numeric_na_pct:** {}

### kgup

- **Output:** `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/clean/kgup_hourly.parquet`
- **Rows in:** 56208
- **Rows out:** 56208
- **ts_hour range:** 2020-01-01 00:00:00+03:00 → 2026-05-30 23:00:00+03:00
- **ts_parse_fail:** 0
- **rows_after_ts_drop:** 56208
- **duplicate_key_rows:** 0
- **rows_after_dedupe:** 56208
- **duplicate_ts_hour_removed:** 0
- **dropped_columns:** []
- **numeric_na_pct:** {}

### load_forecast

- **Output:** `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/clean/load_forecast_hourly.parquet`
- **Rows in:** 56184
- **Rows out:** 56184
- **ts_hour range:** 2020-01-01 00:00:00+03:00 → 2026-05-30 23:00:00+03:00
- **ts_parse_fail:** 0
- **rows_after_ts_drop:** 56184
- **duplicate_key_rows:** 0
- **rows_after_dedupe:** 56184
- **duplicate_ts_hour_removed:** 0
- **dropped_columns:** []
- **numeric_na_pct:** {}

### realtime_generation

- **Output:** `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/clean/realtime_generation_hourly.parquet`
- **Rows in:** 56181
- **Rows out:** 56181
- **ts_hour range:** 2020-01-01 00:00:00+03:00 → 2026-05-29 20:00:00+03:00
- **ts_parse_fail:** 0
- **rows_after_ts_drop:** 56181
- **duplicate_key_rows:** 0
- **rows_after_dedupe:** 56181
- **duplicate_ts_hour_removed:** 0
- **dropped_columns:** ['naphta', 'lng']
- **numeric_na_pct:** {}

### real_consumption

- **Output:** `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/clean/real_consumption_hourly.parquet`
- **Rows in:** 56181
- **Rows out:** 56181
- **ts_hour range:** 2020-01-01 00:00:00+03:00 → 2026-05-29 20:00:00+03:00
- **ts_parse_fail:** 0
- **rows_after_ts_drop:** 56181
- **duplicate_key_rows:** 0
- **rows_after_dedupe:** 56181
- **duplicate_ts_hour_removed:** 0
- **dropped_columns:** []
- **numeric_na_pct:** {}

### smf

- **Output:** `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/clean/smf_hourly.parquet`
- **Rows in:** 56177
- **Rows out:** 56177
- **ts_hour range:** 2020-01-01 00:00:00+03:00 → 2026-05-29 16:00:00+03:00
- **ts_parse_fail:** 0
- **rows_after_ts_drop:** 56177
- **duplicate_key_rows:** 0
- **rows_after_dedupe:** 56177
- **duplicate_ts_hour_removed:** 0
- **dropped_columns:** ['hour']
- **numeric_na_pct:** {}

### yal_yat

- **Output:** `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/clean/yal_yat_hourly.parquet`
- **Rows in:** 56208
- **Rows out:** 56208
- **ts_hour range:** 2020-01-01 00:00:00+03:00 → 2026-05-30 23:00:00+03:00
- **ts_parse_fail:** 0
- **rows_after_ts_drop:** 56208
- **duplicate_key_rows:** 0
- **rows_after_dedupe:** 56208
- **duplicate_ts_hour_removed:** 0
- **dropped_columns:** ['upRegulationTwoCoded', 'downRegulationTwoCoded']
- **numeric_na_pct:** {}

### wind

- **Output:** `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/clean/wind_hourly.parquet`
- **Rows in:** 336333
- **Rows out:** 56208
- **ts_hour range:** 2020-01-01 00:00:00+03:00 → 2026-05-30 23:00:00+03:00
- **rows_after_ts_drop:** 336333
- **duplicate_key_rows:** 0
- **rows_after_dedupe:** 336333
- **partial_hours:** 432
- **numeric_na_pct:** {'quarter2_mean': 0.048035866780529464, 'quarter3_mean': 0.048035866780529464, 'quarter4_mean': 0.048035866780529464, 'generation_mean': 0.5604184457728437, 'generation_min': 0.5604184457728437, 'generation_max': 0.5604184457728437, 'forecast_mean': 0.048035866780529464, 'forecast_min': 0.048035866780529464, 'forecast_max': 0.048035866780529464, 'forecast_std': 0.07828067179049246}

### outages

- **Output:** `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/clean/outages_hourly.parquet`
- **Rows in:** 183437
- **Rows out:** 61368
- **ts_hour range:** 2020-01-01 00:00:00+03:00 → 2026-12-31 23:00:00+03:00
- **duplicate_key_rows:** 0
- **rows_after_expand:** 2357390
- **numeric_na_pct:** {'outage_fault_mw_loss_sum': 8.682049276495894, 'outage_fault_mw_loss_max': 8.682049276495894, 'outage_fault_operator_power_sum': 8.682049276495894, 'outage_maint_capacity_sum': 0.12384304523530179, 'outage_maint_operator_power_sum': 0.12384304523530179}

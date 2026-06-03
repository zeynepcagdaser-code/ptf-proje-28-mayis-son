# Dam Fullness Report

Generated: `2026-06-03T14:49:53.133878Z`
Endpoint: `https://seffaflik.epias.com.tr/electricity-service/v1/dams/data/active-fullness`

- Daily rows: `1`
- Hourly rows: `56280`
- Daily range: `2026-06-02 00:00:00` → `2026-06-02 00:00:00`

## Stats (national mean)

- Overall mean: `77.5%`
- Min / Max: `77.5%` / `77.5%`

## Columns

- `dam_fullness_mean`: national mean active fullness % (all dams, forward-filled hourly)
- `dam_fullness_min/max`: cross-dam spread for that day
- `dam_fullness_change_7d/30d`: weekly/monthly trend
- `dam_fullness_roll_7d/30d/90d`: rolling averages
- `dam_fullness_seasonal_dev`: deviation from 90d baseline (seasonal signal)
- `dam_low_hydro_flag`: 1 when national mean < 40% (hydro-scarce regime)
- `dam_high_hydro_flag`: 1 when national mean > 70% (hydro-abundant, low PTF pressure)
- `dam_fullness_<basin>`: per-basin averages (if available)

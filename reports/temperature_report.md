# Temperature Data Report

Generated: `2026-06-03T11:48:01.029065Z`  Rows: `56328`  Range: `2020-01-01 00:00:00` → `2026-06-04 23:00:00`

## Cities & Population Weights

```
  istanbul     lat=41.0082  lon=28.9784  weight=0.3986
  ankara       lat=39.9334  lon=32.8597  weight=0.1455
  izmir        lat=38.4189  lon=27.1287  weight=0.1127
  bursa        lat=40.1885  lon=29.0610  weight=0.0804
  antalya      lat=36.8969  lon=30.7133  weight=0.0676
  adana        lat=37.0000  lon=35.3213  weight=0.0572
  konya        lat=37.8715  lon=32.4846  weight=0.0586
  diyarbakir   lat=37.9144  lon=40.2306  weight=0.0451
  samsun       lat=41.2867  lon=36.3300  weight=0.0343
```

## National Weighted Temperature

```
  mean=15.6  min=-5.4  max=38.7  p5=3.4  p95=29.9
```

Heatwave hours (tr_temp_mean ≥ 35.0°C): `123`

## Missing Data (> 1%)

```
  (all < 1%)
```

## Output Columns

| Column | Description |
|--------|-------------|
| `tr_temp_mean` | Population-weighted national temperature (°C) |
| `tr_apparent_temp_mean` | Pop-weighted feels-like temperature (°C) |
| `tr_humidity_mean` | Pop-weighted relative humidity (%) |
| `tr_cloud_cover_mean` | Pop-weighted cloud cover (%) |
| `tr_radiation_mean` | Pop-weighted shortwave radiation (W/m²) |
| `tr_wind_speed_mean` | Pop-weighted wind speed (m/s) |
| `tr_cooling_degree` | max(0, tr_temp_mean − 22) — AC load proxy |
| `tr_heating_degree` | max(0, 18 − tr_temp_mean) — heating load proxy |
| `tr_heatwave_flag` | 1 if tr_temp_mean ≥ 35 °C |
| `temp_{city}` | Per-city temperature_2m (9 cities) |
| `apparent_temp_{city}` | Per-city feels-like temperature (9 cities) |

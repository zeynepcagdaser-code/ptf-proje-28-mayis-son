## FİBA/FİBS + GRF Feature Sanity & Leakage Check

**Model eğitimi yapılmadı.**

### Master doluluk (ham kolonlar)
- `dam_price_independent_buy_mwh`: missing=0 inf=0 min=8132.6 p01=12659.2 median=22470 p99=33008.6 max=43135.4
- `dam_price_independent_sell_mwh`: missing=0 inf=0 min=4686.5 p01=7157.96 median=12839.8 p99=22041.2 max=28136.5
- `grf_tl_1000sm3`: missing=0 inf=0 min=1204.86 p01=1247.53 median=10597.4 p99=26265.1 max=26297
- `grf_usd_1000sm3`: missing=0 inf=0 min=34.5 p01=166.58 median=349.555 p99=1424.81 max=1444.51
- `grf_eur_mwh`: missing=0 inf=0 min=3.1 p01=13.11 median=29.225 p99=135.65 max=139.69
- `grf_usd_mmbtu`: missing=0 inf=0 min=0.95 p01=4.59 median=9.63 p99=39.25 max=39.79

### Feature parquet doluluk (split bazında)
#### split=train
- `dam_price_independent_buy_mwh`: missing=0 inf=0 min=9505.3 p01=12696.5 median=21939.7 p99=32007.1 max=43135.4 extreme=0
- `dam_price_independent_sell_mwh`: missing=0 inf=0 min=4686.5 p01=6976.44 median=12575.4 p99=20605.7 max=25225.2 extreme=0
- `grf_tl_1000sm3`: missing=0 inf=0 min=1204.86 p01=1233.08 median=9652.08 p99=26280 max=26297 extreme=0
- `grf_usd_1000sm3`: missing=0 inf=0 min=162.85 p01=166.35 median=351.705 p99=1437.4 max=1444.51 extreme=0
- `grf_eur_mwh`: missing=0 inf=0 min=12.86 p01=13.09 median=30.245 p99=135.83 max=139.69 extreme=0
- `grf_usd_mmbtu`: missing=0 inf=0 min=4.49 p01=4.58 median=9.69 p99=39.59 max=39.79 extreme=0
- `fiba_fibs_ratio`: missing=0 inf=0 min=0.561712 p01=0.92689 median=1.73754 p99=3.03521 max=5.8605 extreme=0
- `fiba_fibs_balance`: missing=0 inf=0 min=-7654 p01=-1211.23 median=9274.3 p99=18841.7 max=34656.7 extreme=0
- `fiba_fibs_total`: missing=0 inf=0 min=19171 p01=23000.2 median=34540 p99=49322.9 max=54907.1 extreme=0
- `fiba_fibs_pressure`: missing=0 inf=0 min=-0.280646 p01=-0.037942 median=0.269417 p99=0.504363 max=0.708476 extreme=0
- `grf_tl_lag_1d`: missing=0 inf=0 min=1204.86 p01=1233.08 median=9649.28 p99=26280 max=26297 extreme=0
- `grf_tl_change_7d`: missing=0.000549451 inf=0 min=-6570.23 p01=-2101.41 median=0.58 p99=3164.51 max=7649.72 extreme=0
- `grf_tl_rolling_mean_7d`: missing=0.000526557 inf=0 min=1211.82 p01=1235.08 median=9614.01 p99=26256.9 max=26297 extreme=0
- `gas_cost_pressure`: missing=0 inf=0 min=18.2363 p01=48.7437 median=1362.24 p99=7949.49 max=11104.8 extreme=0
- `thermal_cost_pressure`: missing=0 inf=0 min=281.67 p01=421.031 median=4867.65 p99=18728.1 max=22103.8 extreme=0
- `gas_marginal_pressure`: missing=0 inf=0 min=4.16604 p01=15.2404 median=759.583 p99=5927.89 max=9018.38 extreme=0

#### split=validation
- `dam_price_independent_buy_mwh`: missing=0 inf=0 min=15692.8 p01=17598.7 median=25481 p99=34515.6 max=37256 extreme=0
- `dam_price_independent_sell_mwh`: missing=0 inf=0 min=7200.9 p01=8306.41 median=13420.2 p99=24213.1 max=28136.5 extreme=0
- `grf_tl_1000sm3`: missing=0 inf=0 min=1245.53 p01=12305.6 median=14234.4 p99=14947.8 max=15459.5 extreme=0
- `grf_usd_1000sm3`: missing=0 inf=0 min=34.5 p01=326.58 median=348.63 p99=373.78 max=425.22 extreme=0
- `grf_eur_mwh`: missing=0 inf=0 min=3.1 p01=26.76 median=28.49 p99=33.19 max=38.12 extreme=0
- `grf_usd_mmbtu`: missing=0 inf=0 min=0.95 p01=9 median=9.6 p99=10.3 max=11.71 extreme=0
- `fiba_fibs_ratio`: missing=0 inf=0 min=0.681751 p01=0.876278 median=1.8743 p99=3.20296 max=4.23102 extreme=0
- `fiba_fibs_balance`: missing=0 inf=0 min=-8941.9 p01=-2835.03 median=11556.6 p99=22554.3 max=27497.2 extreme=0
- `fiba_fibs_total`: missing=0 inf=0 min=24896.8 p01=28037.7 median=39573.5 p99=51294.5 max=54207 extreme=0
- `fiba_fibs_pressure`: missing=0 inf=0 min=-0.189237 p01=-0.0659426 median=0.304178 p99=0.524144 max=0.617665 extreme=0
- `grf_tl_lag_1d`: missing=0 inf=0 min=1245.53 p01=12305.6 median=13894.3 p99=14947.8 max=15459.5 extreme=0
- `grf_tl_change_7d`: missing=0 inf=0 min=-11605 p01=-757.42 median=2 p99=1377.39 max=11934.5 extreme=0
- `grf_tl_rolling_mean_7d`: missing=0 inf=0 min=11252.6 p01=11279.9 median=13780.8 p99=14644.5 max=14740.5 extreme=0
- `gas_cost_pressure`: missing=0 inf=0 min=150.622 p01=319.784 median=3124.42 p99=5456.48 max=5972.38 extreme=0
- `thermal_cost_pressure`: missing=0 inf=0 min=698.986 p01=2929.08 median=8051.09 p99=11766.3 max=12537.2 extreme=0
- `gas_marginal_pressure`: missing=0 inf=0 min=30.7292 p01=90.4847 median=1815.84 p99=4297.22 max=4848.7 extreme=0

#### split=test
- `dam_price_independent_buy_mwh`: missing=0 inf=0 min=8132.6 p01=10421.2 median=22583.9 p99=32440 max=34131.9 extreme=0
- `dam_price_independent_sell_mwh`: missing=0 inf=0 min=8825.4 p01=9885.12 median=14732.5 p99=24321.4 max=27862.6 extreme=0
- `grf_tl_1000sm3`: missing=0 inf=0 min=14107 p01=14122 median=14699 p99=17357 max=17384.4 extreme=0
- `grf_usd_1000sm3`: missing=0 inf=0 min=322.05 p01=322.93 median=335.82 p99=390.12 max=390.59 extreme=0
- `grf_eur_mwh`: missing=0 inf=0 min=25.8 p01=25.93 median=27.07 p99=31.7 max=31.78 extreme=0
- `grf_usd_mmbtu`: missing=0 inf=0 min=8.87 p01=8.89 median=9.25 p99=10.75 max=10.76 extreme=0
- `fiba_fibs_ratio`: missing=0 inf=0 min=0.333607 p01=0.506295 median=1.46068 p99=3.06271 max=3.78873 extreme=0
- `fiba_fibs_balance`: missing=0 inf=0 min=-18079.8 p01=-11117.2 median=7020.1 p99=21247.1 max=24951.8 extreme=0
- `fiba_fibs_total`: missing=0 inf=0 min=21493.7 p01=24151.9 median=38048.1 p99=48189.6 max=50432.3 extreme=0
- `fiba_fibs_pressure`: missing=0 inf=0 min=-0.499692 p01=-0.327761 median=0.187216 p99=0.507713 max=0.582353 extreme=0
- `grf_tl_lag_1d`: missing=0 inf=0 min=14107 p01=14122 median=14678 p99=17357 max=17384.4 extreme=0
- `grf_tl_change_7d`: missing=0 inf=0 min=-572 p01=-376.19 median=-5.09 p99=2393.9 max=2665.36 extreme=0
- `grf_tl_rolling_mean_7d`: missing=0 inf=0 min=14142.7 p01=14147.7 median=14641.2 p99=17319.2 max=17338.1 extreme=0
- `gas_cost_pressure`: missing=0 inf=0 min=19.2774 p01=25.2942 median=479.026 p99=5111.63 max=6035.63 extreme=0
- `thermal_cost_pressure`: missing=0 inf=0 min=1737.92 p01=2003.64 median=4471.18 p99=10633 max=11935.9 extreme=0
- `gas_marginal_pressure`: missing=0 inf=0 min=2.32243 p01=3.80104 median=120.579 p99=3762.58 max=5079 extreme=0

### Güvenli bölme kontrolleri (FİBA/FİBS)
- **ratio_sell_zero_rows**: 0
- **ratio_bad_non_nan_when_sell_zero**: 0
- **ratio_inf_when_sell_zero**: 0
- **pressure_total_zero_rows**: 0
- **pressure_bad_non_nan_when_total_zero**: 0
- **pressure_inf_when_total_zero**: 0

### GRF forward-fill kontrolü (master içinde gün bazında)
- `grf_tl_1000sm3`: days_with_multiple_values=0 max_unique_values_in_a_day=1
- `grf_usd_1000sm3`: days_with_multiple_values=0 max_unique_values_in_a_day=1
- `grf_eur_mwh`: days_with_multiple_values=0 max_unique_values_in_a_day=1
- `grf_usd_mmbtu`: days_with_multiple_values=0 max_unique_values_in_a_day=1

### Target korelasyonları (overall)
- `dam_price_independent_buy_mwh`: corr(target_1h)=0.358261 corr(target_24h)=0.301485
- `dam_price_independent_sell_mwh`: corr(target_1h)=-0.119399 corr(target_24h)=-0.0816014
- `grf_tl_1000sm3`: corr(target_1h)=0.776812 corr(target_24h)=0.776492
- `grf_usd_1000sm3`: corr(target_1h)=0.539898 corr(target_24h)=0.541292
- `grf_eur_mwh`: corr(target_1h)=0.53983 corr(target_24h)=0.541271
- `grf_usd_mmbtu`: corr(target_1h)=0.539891 corr(target_24h)=0.541284
- `fiba_fibs_ratio`: corr(target_1h)=0.337881 corr(target_24h)=0.276204
- `fiba_fibs_balance`: corr(target_1h)=0.415789 corr(target_24h)=0.337597
- `fiba_fibs_total`: corr(target_1h)=0.203835 corr(target_24h)=0.181168
- `fiba_fibs_pressure`: corr(target_1h)=0.373096 corr(target_24h)=0.296486
- `grf_tl_lag_1d`: corr(target_1h)=0.777838 corr(target_24h)=0.776381
- `grf_tl_change_7d`: corr(target_1h)=-0.0114185 corr(target_24h)=-0.0170469
- `grf_tl_rolling_mean_7d`: corr(target_1h)=0.779982 corr(target_24h)=0.778842
- `gas_cost_pressure`: corr(target_1h)=0.807064 corr(target_24h)=0.75964
- `thermal_cost_pressure`: corr(target_1h)=0.849531 corr(target_24h)=0.823167
- `gas_marginal_pressure`: corr(target_1h)=0.744898 corr(target_24h)=0.695711

### Leakage değerlendirmesi (manuel)
- **fiba_fibs_same_hour**: FİBA/FİBS DAM orderbook toplamları saatlik; anchor saatinde bilinebilir kabul edildi. Cutoff politikası daha sıkıysa (G-1 belirli saat) aynı-saat kullanımı yeniden değerlendirilmelidir.
- **grf_current_vs_lagged**: GRF günlük ilan/publish zamanına bağlı olarak aynı-gün kullanım leakage riski taşıyabilir. Bu nedenle lag_1d + rolling/change gibi geçmişe dayalı türevler daha güvenli.
- **gas_cost_pressure_current**: gas_cost_pressure = gas_share * grf_tl_1000sm3 ifadesinde grf current ise yayın zamanı riskini taşır; lagged GRF ile türetilmiş versiyonlar daha güvenlidir (gerekirse yeni feature eklenebilir).

### Feature list resolve kontrolü
- **main_regression_new_features_in_list**: ['dam_price_independent_buy_mwh', 'dam_price_independent_sell_mwh', 'fiba_fibs_ratio', 'fiba_fibs_balance', 'fiba_fibs_pressure', 'grf_tl_1000sm3', 'grf_tl_lag_1d', 'grf_tl_change_7d', 'grf_tl_rolling_mean_7d', 'gas_cost_pressure', 'thermal_cost_pressure', 'gas_marginal_pressure']
- **main_regression_list_contains_all**: True
- **low_price_list_contains_all**: True
- **parquet_has_all_new_features**: True
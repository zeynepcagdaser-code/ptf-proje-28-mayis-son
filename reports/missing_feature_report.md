# Missing Feature Report

- Requested: 30
- Present: 13
- Missing: 17
- Missing source data: 19

## Tez / piyasa veri borcu (ham veri yok — pipeline'a eklenmez)

Aşağıdaki gruplar `features.config.THESIS_DATA_DEBT_GROUPS` ile tanımlıdır.
Feature engineering bu kaynaklar gelene kadar üretilmez; yalnızca raporlanır.

| Grup | Hedef feature'lar | Ham veri kaynağı |
|------|-------------------|------------------|
| Fiyattan bağımsız alış/satış oranı | `buy_sell_ratio, order_book_imbalance_proxy` | GÖP/DAM işlem veya emir defteri (alış/satış hacmi ayrımı) |
| BOTAŞ doğal gaz tarifesi | `botas_gas_tariff, botas_tariff_lag_30d` | BOTAŞ / EPDK tarihsel tarife tablosu |
| USD/TL, EUR/TL | `usd_try, eur_try, fx_vol_30d` | TCMB / ECB günlük kur serisi |
| TTF / Brent | `ttf_gas_price, brent_oil_price, ttf_try_proxy` | ICE TTF, Brent; ttf_try_proxy = ttf * usd_try |
| TETAŞ-EÜAŞ tarife | `tetas_euas_tariff, regulated_tariff_index` | EPDK / şirket duyuruları |
| Mesken AG tarife | `residential_ag_tariff` | EPDK perakende tarife |
| Doğal gaz santrali yakıt maliyeti | `gas_plant_fuel_cost, fuel_cost_index, gas_cost_pressure, gas_marginal_pressure` | BOTAŞ tarife + TTF + verimlilik / kgup gaz MW |
| Güneş / gökyüzü açıklığı veya solar radiation proxy | `clearness_index, solar_radiation_proxy, sky_clear_fraction` | Meteoroloji, PVGIS veya bulutluluk (şu an yalnızca solar_peak_hour_flag) |
| YEKDEM / merchant / non-merchant ayrımı | `yekdem_unit_price, merchant_proxy_share, non_merchant_proxy_share, yekdem_revenue_loss_proxy` | YEKDEM birim fiyat, KGUP müst run / sözleşme sınıflandırması |

## İstenen feature'lar (REQUESTED_FEATURES)

| Feature | Present | Source status | Missing sources |
|---------|:-------:|--------------|----------------|
| `low_load_flag` | 1 | ok | - |
| `holiday_low_load_flag` | 1 | ok | - |
| `renewable_pressure` | 1 | ok | - |
| `hydro_pressure` | 0 | missing_source_data | - |
| `res_ges_hes_pressure` | 0 | missing_source_data | - |
| `zero_price_risk_proxy` | 1 | ok | - |
| `low_price_regime_score` | 0 | missing_source_data | - |
| `gas_share` | 1 | ok | - |
| `coal_share` | 1 | ok | - |
| `gas_coal_competition_index` | 1 | ok | - |
| `thermal_price_setting_share` | 1 | ok | - |
| `gas_marginal_proxy` | 0 | missing_source_data | - |
| `merchant_proxy_share` | 0 | missing_source_data | - |
| `non_merchant_proxy_share` | 0 | missing_source_data | - |
| `ttf_gas_price` | 0 | missing_source_data | - |
| `brent_oil_price` | 0 | missing_source_data | - |
| `usd_try` | 0 | missing_source_data | - |
| `eur_try` | 0 | missing_source_data | - |
| `ttf_try_proxy` | 0 | missing_source_data | - |
| `fuel_cost_index` | 0 | missing_source_data | - |
| `gas_cost_pressure` | 1 | missing_source_data | - |
| `gas_marginal_pressure` | 1 | missing_source_data | - |
| `price_cap` | 1 | partial_unknown | - |
| `ptf_to_cap_ratio` | 1 | ok | - |
| `smf_to_cap_ratio` | 1 | ok | - |
| `spread_risk_flag` | 0 | missing_source_data | - |
| `imbalance_cost_proxy` | 0 | missing_source_data | - |
| `yekdem_unit_price` | 0 | missing_source_data | - |
| `kgup_excess_generation` | 0 | missing_source_data | kgup_excess_generation |
| `yekdem_revenue_loss_proxy` | 0 | missing_source_data | - |

## Model kovaları — parquet'te mevcut / eksik

- MAIN_REGRESSION: 73/73 mevcut (0 eksik)
- LOW_PRICE_CLASSIFIER: 57/57 mevcut (0 eksik)
- RISK_DASHBOARD: 32/32 mevcut (0 eksik)

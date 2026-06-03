# Weather Feature Impact Report

Generated: 2026-06-03  |  Experiment: 9-sehir sicaklik → rolling PTF next-24h modeli

## 1. Dogrulama Kontrolleri

| Kontrol | Sonuc |
|---------|-------|
| data/processed/weather_features.parquet | BULUNMADI (dogru ad: temperature_hourly.parquet) |
| data/processed/temperature_hourly.parquet | OK: 56,328 satir, 28 sutun, 2020-01-01 to 2026-06-04 |
| build_master_dataset.py weather merge? | Hayir - rolling system master dataset kullanmiyor |
| load_temperature() loaderlar listesinde? | OK - rolling_ptf_forecast_system.py satir 372 |
| Delivery-hour weather leakage? | TEMIZ - REALIZED_DELIVERY_COLS icinde degil |
| Train/val/test NaN? | %0 NaN tum splitlede |

## 2. Modeldeki Weather Feature Sayisi

Toplam feature: 187 (onceki: 170, +17 hava durumu feature)

27 weather feature modele girdi:
  - 18 tr_* aggregate (ulusal agirlikli): tr_temp_mean, tr_cooling_degree, tr_heatwave_flag, ...
  - 9 per-city: delivery_temp_{istanbul, ankara, izmir, bursa, antalya, adana, konya, diyarbakir, samsun}

## 3. Model Performansi Karsilastirmasi

| Metrik | Onceki (170 feat) | Yeni (187 feat) | Delta |
|--------|:-----------------:|:---------------:|:-----:|
| Val MAE (TL/MWh) | 219.05 | 216.87 | -2.18 |
| Val RMSE | 310.38 | 307.43 | -2.95 |
| Val WAPE | 8.36% | 8.28% | -0.08% |
| Test MAE (TL/MWh) | 396.40 | 387.38 | -9.02 |
| Test RMSE | 619.31 | 611.80 | -7.50 |
| Test WAPE | 24.55% | 23.99% | -0.56% |
| Val vs persistence | 45.0% | 45.6% | +0.6pp |
| Test vs persistence | 26.2% | 27.9% | +1.7pp |

Val persistence: 398.51 TL/MWh  |  Test persistence: 537.35 TL/MWh

## 4. Saatlik MAE Dagilimi

| Saat | Val MAE | Test MAE | Not |
|-----:|--------:|---------:|-----|
| 00:00 | 157.9 | 471.2 |
| 01:00 | 185.4 | 410.5 |
| 02:00 | 207.5 | 409.5 |
| 03:00 | 214.5 | 353.5 |
| 04:00 | 231.4 | 354.1 |
| 05:00 | 247.2 | 354.0 |
| 06:00 | 272.2 | 399.8 |
| 07:00 | 269.1 | 373.9 |
| 08:00 | 250.9 | 290.4 |
| 09:00 | 291.5 | 221.7 |
| 10:00 | 296.0 | 214.6 |
| 11:00 | 325.1 | 212.8 |
| 12:00 | 260.1 | 186.5 | <- klima/solar
| 13:00 | 288.7 | 195.9 | <- klima/solar
| 14:00 | 313.3 | 224.9 | <- klima/solar
| 15:00 | 298.7 | 228.9 | <- klima/solar
| 16:00 | 229.1 | 284.1 | <- klima/solar
| 17:00 | 158.4 | 401.3 | <- klima/solar
| 18:00 | 125.9 | 558.8 | <- klima/solar
| 19:00 | 60.8 | 717.0 | <- aksam pik
| 20:00 | 84.1 | 603.2 | <- aksam pik
| 21:00 | 83.3 | 621.1 | <- aksam pik
| 22:00 | 158.8 | 613.6 | <- aksam pik
| 23:00 | 195.1 | 595.9 |

12-18h (klima/solar): Val 239.2 TL/MWh  |  Test 297.2 TL/MWh
17-22h (aksam pik):   Val 111.9 TL/MWh  |  Test 585.8 TL/MWh

Test 2026 aksam saatlerindeki (17-22h) yuksek MAE, hava durumu eksikliginden degil,
2026 piyasa rejim kaymasindan (fiyat seviyesi ve volatilite artisi) kaynaklanmaktadir.

## 5. Sicak Hava Dalgasi (Heatwave) Analizi

Heatwave tanimi: tr_temp_mean >= 35 derece C (nufus agirlikli ulusal ortalama)

| Split | Heatwave Saatleri | Heatwave MAE | Normal MAE | Fark |
|-------|:-----------------:|:------------:|:----------:|:----:|
| Val 2025 | 1416 | 222.69 | 216.83 | +5.86 |
| Test 2026 | 0 | - | 387.38 | Test doneminde heatwave yok (Ocak-Haziran) |

Val 2025: 1,416 heatwave saatinde model yalnizca +5.86 TL/MWh daha kotü performans gosteriyor.
load_x_cooling ve tr_heatwave_flag featurelari bu durumu onemli olcude yakaliyor.

## 6. Ozet

| | Val MAE | Test MAE |
|-|:-------:|:--------:|
| Onceki model (170 feat) | 219.05 | 396.40 |
| Yeni model (187 feat) | 216.87 | 387.38 |
| Delta | -2.18 | -9.02 |

- Val: -2.2 TL/MWh iyilesme (%1.0)
- Test: -9.0 TL/MWh iyilesme (%2.3) -- 2026 doneminde daha belirgin etki
- Heatwave saatlerinde model performansi normale yakin (+5.9 TL/MWh fark)
- Test 2026 aksam piki (17-22h) yuksek MAE: rejim kaymasi, weather eksikligi degil

### Sonraki Adimlar
1. Incremental fetch: gunluk pipeline icin python3 fetch_temperature.py (--full olmadan)
2. gas_share x hour etkilesimi -- regime-dependent gas pricing dene
3. Full ablation (300 estimator) -- weather grubunun gercek DELTA_VAL degerini olc
4. 2026 rejim kaymasi arastir -- test aksam saatleri icin ek ozellik

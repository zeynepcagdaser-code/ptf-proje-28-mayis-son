# Leave-One-Group-Out (LOGO) Ablation Raporu

**Oluşturulma:** `2026-06-02T23:01:55.023583Z`  
**Mod:** Hızlı (quick)  
**Baseline Val MAE:** `225.05` TL/MWh  
**Baseline Test MAE:** `396.65` TL/MWh  
**Toplam feature:** `194`

---

## Özet Tablo

ΔMAE = Baseline − Ablated. **Pozitif** = grup kaldırılınca MAE arttı → grup faydalı.

| Grup | N | ΔVAL MAE | ΔTEST MAE | Kritik Saatler |
|------|--:|---------:|----------:|----------------|
| 🟠 **Baseline_persistence** | 6 | **+91.3** | +86.3 | s12(Δ+205) · s10(Δ+204) · s13(Δ+180) |
| 🟠 **DAM_orderbook** | 13 | **+41.4** | -12.4 | s8(Δ+96) · s10(Δ+90) · s16(Δ+83) |
| 🟡 **PTF_history** | 38 | **+12.2** | -20.4 | s8(Δ+46) · s9(Δ+28) · s3(Δ+26) |
| 🟡 **Ramp_dynamics** | 8 | **+12.1** | +17.1 | s8(Δ+38) · s9(Δ+32) · s10(Δ+30) |
| 🟡 **Load_demand** | 14 | **+7.4** | -6.2 | s8(Δ+33) · s10(Δ+28) · s9(Δ+27) |
| 🟡 **Lagged_realized** | 20 | **+6.8** | +5.1 | s5(Δ+24) · s6(Δ+22) · s0(Δ+20) |
| 🟡 **Calendar_time** | 31 | **+3.8** | -0.0 | s8(Δ+26) · s6(Δ+22) · s0(Δ+14) |
| 🟡 **International_commodities** | 11 | **+0.5** | -3.6 | s6(Δ+17) · s10(Δ+16) · s17(Δ-13) |
| ⚪ **GRF_gas_price** | 14 | **-0.2** | +1.0 | s10(Δ+15) · s17(Δ-14) · s18(Δ-12) |
| ⚪ **Exchange_rates** | 11 | **-2.4** | -3.2 | s1(Δ-21) · s23(Δ-15) · s14(Δ+12) |
| ⚪ **KGUP_generation_plan** | 14 | **-2.8** | +10.3 | s18(Δ-19) · s21(Δ-19) · s8(Δ+16) |
| ⚪ **Temperature** | 8 | **-4.4** | +1.9 | s21(Δ-25) · s19(Δ-19) · s17(Δ-17) |
| ⚪ **Composite_interactions** | 6 | **-7.5** | -1.5 | s21(Δ-18) · s17(Δ-18) · s16(Δ-16) |

---

## Grup Detayları

### Baseline_persistence

**Açıklama:** Persistence baselines: dünün aynı saati, 2 gün önce, hafta önce  
**İngilizce:** D-1 / D-2 / D-7 persistence prices and their momentum differences  
**Feature sayısı:** 6  
**Val MAE ablated:** 316.34 (baseline: 225.05) → ΔMAE = **+91.3** (✅ faydalı)  
**Test MAE ablated:** 483.0 (baseline: 396.65) → ΔMAE = **+86.3**

**Saate göre ΔMAE (validation):**

| Saat | ΔMAE | Yorum |
|------|-----:|-------|

**Features:**

- `baseline_d1_ptf`
- `anchor_baseline_d2_ptf`
- `anchor_baseline_week_ptf`
- `anchor_baseline_recent_3d_mean`
- `anchor_baseline_d1_d2_momentum`
- `anchor_baseline_d1_week_momentum`

### DAM_orderbook

**Açıklama:** DAM emir defteri: alış/satış hacim, fiyatsız emirler, blok emirler  
**İngilizce:** DAM bid/offer volumes, price-independent buy/sell, block buy volume, bid-sell balance/ratio  
**Feature sayısı:** 13  
**Val MAE ablated:** 266.44 (baseline: 225.05) → ΔMAE = **+41.4** (✅ faydalı)  
**Test MAE ablated:** 384.23 (baseline: 396.65) → ΔMAE = **-12.4**

**Saate göre ΔMAE (validation):**

| Saat | ΔMAE | Yorum |
|------|-----:|-------|

**Features:**

- `anchor_dam_matched_volume_lag_24`
- `anchor_dam_matched_volume_lag_168`
- `delivery_dam_bid_volume`
- `delivery_dam_sell_offer_volume`
- `delivery_dam_price_independent_buy`
- `delivery_dam_price_independent_sell`
- `delivery_dam_block_buy_volume`
- `delivery_price_independent_balance`
- `delivery_price_independent_pressure`
- `delivery_dam_bid_sell_ratio`
- `delivery_dam_bid_sell_balance`
- `delivery_block_to_matched_ratio`
- `delivery_night_block_pressure`

### PTF_history

**Açıklama:** Anchor saatteki PTF geçmişi: lag, rolling stats, momentum  
**İngilizce:** PTF price lags, rolling mean/std/min/max (6h→168h), zero/spike ratios, D1/week momentum  
**Feature sayısı:** 38  
**Val MAE ablated:** 237.26 (baseline: 225.05) → ΔMAE = **+12.2** (✅ faydalı)  
**Test MAE ablated:** 376.24 (baseline: 396.65) → ΔMAE = **-20.4**

**Saate göre ΔMAE (validation):**

| Saat | ΔMAE | Yorum |
|------|-----:|-------|

**Features:**

- `anchor_ptf`
- `anchor_ptf_lag_1`
- `anchor_ptf_lag_2`
- `anchor_ptf_lag_3`
- `anchor_ptf_lag_4`
- `anchor_ptf_lag_6`
- `anchor_ptf_lag_12`
- `anchor_ptf_lag_24`
- `anchor_ptf_lag_25`
- `anchor_ptf_lag_26`
- `anchor_ptf_lag_48`
- `anchor_ptf_lag_72`
- `anchor_ptf_lag_168`
- `anchor_ptf_roll_mean_6`
- `anchor_ptf_roll_std_6`
- `anchor_ptf_roll_min_6`
- `anchor_ptf_roll_max_6`
- `anchor_ptf_roll_mean_12`
- `anchor_ptf_roll_std_12`
- `anchor_ptf_roll_min_12`
- _...+18 more_

### Ramp_dynamics

**Açıklama:** Saatlik değişim (delta) ve rampa baskısı  
**İngilizce:** 1-hour deltas for load/KGUP/renewable, ramp tightness, ramp flags (morning/evening/night)  
**Feature sayısı:** 8  
**Val MAE ablated:** 237.19 (baseline: 225.05) → ΔMAE = **+12.1** (✅ faydalı)  
**Test MAE ablated:** 413.74 (baseline: 396.65) → ΔMAE = **+17.1**

**Saate göre ΔMAE (validation):**

| Saat | ΔMAE | Yorum |
|------|-----:|-------|

**Features:**

- `delivery_kgup_total_delta_1h`
- `delivery_kgup_gas_delta_1h`
- `delivery_kgup_renewable_delta_1h`
- `delivery_ramp_tightness`
- `delivery_morning_ramp_flag`
- `delivery_evening_ramp_flag`
- `delivery_night_block_flag`
- `delivery_block_buy_delta_1h`

### Load_demand

**Açıklama:** Yük tahmini ve net yük türevleri  
**İngilizce:** TEİAŞ load forecast, net load after wind/solar/renewable, imbalance vs KGUP, ramp deltas  
**Feature sayısı:** 14  
**Val MAE ablated:** 232.46 (baseline: 225.05) → ΔMAE = **+7.4** (✅ faydalı)  
**Test MAE ablated:** 390.44 (baseline: 396.65) → ΔMAE = **-6.2**

**Saate göre ΔMAE (validation):**

| Saat | ΔMAE | Yorum |
|------|-----:|-------|

**Features:**

- `delivery_load_forecast`
- `delivery_net_load_after_wind_solar`
- `delivery_net_load_after_renewable`
- `delivery_load_minus_kgup_total`
- `delivery_wind_load_share`
- `delivery_solar_load_share`
- `delivery_renewable_load_share`
- `delivery_load_forecast_delta_1h`
- `delivery_net_load_renewable_delta_1h`
- `delivery_load_x_cooling`
- `delivery_load_x_heating`
- `delivery_load_x_renewable`
- `delivery_load_x_gas`
- `delivery_netload_x_thermal`

### Lagged_realized

**Açıklama:** Gecikmeli gerçekleşen veriler: SMF, gerçek tüketim, üretim, IDM, YAL/YAT  
**İngilizce:** Realized post-settlement data lagged 24h/168h: SMF, real consumption, generation mix, IDM price, YAL/YAT regulation  
**Feature sayısı:** 20  
**Val MAE ablated:** 231.86 (baseline: 225.05) → ΔMAE = **+6.8** (✅ faydalı)  
**Test MAE ablated:** 401.79 (baseline: 396.65) → ΔMAE = **+5.1**

**Saate göre ΔMAE (validation):**

| Saat | ΔMAE | Yorum |
|------|-----:|-------|

**Features:**

- `anchor_idm_price_lag_24`
- `anchor_idm_price_lag_168`
- `anchor_dam_idm_spread_lag_24`
- `anchor_dam_idm_spread_lag_168`
- `anchor_smf_lag_24`
- `anchor_smf_lag_168`
- `anchor_real_consumption_lag_24`
- `anchor_real_consumption_lag_168`
- `anchor_yal_yat_net_lag_24`
- `anchor_yal_yat_net_lag_168`
- `anchor_gen_total_lag_24`
- `anchor_gen_total_lag_168`
- `anchor_gen_gas_lag_24`
- `anchor_gen_gas_lag_168`
- `anchor_gen_wind_lag_24`
- `anchor_gen_wind_lag_168`
- `anchor_gen_solar_lag_24`
- `anchor_gen_solar_lag_168`
- `anchor_smf_ptf_spread_lag_24`
- `anchor_smf_ptf_spread_lag_168`

### Calendar_time

**Açıklama:** Takvim ve zaman özellikleri: saat, gün, ay, tatil, Ramazan  
**İngilizce:** Hour/DOW/month (linear + sin/cos), weekend flag, public holiday flags, Ramadan proxy  
**Feature sayısı:** 31  
**Val MAE ablated:** 228.86 (baseline: 225.05) → ΔMAE = **+3.8** (✅ faydalı)  
**Test MAE ablated:** 396.63 (baseline: 396.65) → ΔMAE = **-0.0**

**Saate göre ΔMAE (validation):**

| Saat | ΔMAE | Yorum |
|------|-----:|-------|

**Features:**

- `horizon`
- `anchor_hour`
- `anchor_dow`
- `anchor_month`
- `anchor_hour_sin`
- `anchor_hour_cos`
- `anchor_dow_sin`
- `anchor_dow_cos`
- `anchor_month_sin`
- `anchor_month_cos`
- `anchor_is_weekend`
- `anchor_is_holiday`
- `anchor_is_holiday_or_weekend`
- `anchor_is_pre_holiday`
- `anchor_is_post_holiday`
- `anchor_ramadan_season_proxy`
- `delivery_hour`
- `delivery_dow`
- `delivery_month`
- `delivery_hour_sin`
- _...+11 more_

### International_commodities

**Açıklama:** Uluslararası emtia fiyatları: Brent, TTF, kömür, Henry Hub  
**İngilizce:** Brent crude, TTF European gas, API2 coal, Henry Hub, 7d change / 30d rolling mean  
**Feature sayısı:** 11  
**Val MAE ablated:** 225.51 (baseline: 225.05) → ΔMAE = **+0.5** (✅ faydalı)  
**Test MAE ablated:** 393.04 (baseline: 396.65) → ΔMAE = **-3.6**

**Saate göre ΔMAE (validation):**

| Saat | ΔMAE | Yorum |
|------|-----:|-------|

**Features:**

- `delivery_brent_usd`
- `delivery_ttf_eur_mwh`
- `delivery_ttf_try_mwh`
- `delivery_coal_api2_usd`
- `delivery_brent_try`
- `delivery_henry_hub_usd`
- `delivery_brent_usd_change_7d`
- `delivery_ttf_eur_mwh_change_7d`
- `delivery_brent_usd_roll_mean_30d`
- `delivery_ttf_eur_mwh_roll_mean_30d`
- `delivery_brent_ttf_try_ratio`

### GRF_gas_price

**Açıklama:** GRF (Günlük Referans Fiyatı): doğalgaz maliyeti  
**İngilizce:** Turkish natural gas daily reference price (TL/1000Sm³ + USD/MMBtu), lags, 7d/30d trends  
**Feature sayısı:** 14  
**Val MAE ablated:** 224.84 (baseline: 225.05) → ΔMAE = **-0.2** (⚠️ zararlı/nötr)  
**Test MAE ablated:** 397.61 (baseline: 396.65) → ΔMAE = **+1.0**

**Saate göre ΔMAE (validation):**

| Saat | ΔMAE | Yorum |
|------|-----:|-------|

**Features:**

- `anchor_grf_tl_lag_24`
- `anchor_grf_usd_mmbtu_lag_24`
- `delivery_grf_tl_1000sm3`
- `delivery_grf_usd_1000sm3`
- `delivery_grf_eur_mwh`
- `delivery_grf_usd_mmbtu`
- `delivery_grf_tl_lag_24`
- `delivery_grf_tl_change_7d`
- `delivery_grf_tl_pct_change_7d`
- `delivery_grf_tl_roll_mean_7d`
- `delivery_grf_tl_roll_mean_30d`
- `delivery_grf_usd_mmbtu_lag_24`
- `delivery_grf_usd_mmbtu_change_7d`
- `delivery_ttf_vs_grf_premium`

### Exchange_rates

**Açıklama:** Döviz kurları: USD/TRY, EUR/TRY  
**İngilizce:** TCMB USD/TRY and EUR/TRY exchange rates, 1d lags, 7-day change/rolling mean  
**Feature sayısı:** 11  
**Val MAE ablated:** 222.69 (baseline: 225.05) → ΔMAE = **-2.4** (⚠️ zararlı/nötr)  
**Test MAE ablated:** 393.43 (baseline: 396.65) → ΔMAE = **-3.2**

**Saate göre ΔMAE (validation):**

| Saat | ΔMAE | Yorum |
|------|-----:|-------|

**Features:**

- `delivery_usd_try_buy`
- `delivery_eur_try_buy`
- `delivery_eur_usd_cross_buy`
- `delivery_usd_try_buy_lag_1d`
- `delivery_eur_try_buy_lag_1d`
- `delivery_usd_try_buy_change_7d`
- `delivery_eur_try_buy_change_7d`
- `delivery_usd_try_buy_pct_change_7d`
- `delivery_eur_try_buy_pct_change_7d`
- `delivery_usd_try_buy_roll_mean_7d`
- `delivery_eur_try_buy_roll_mean_7d`

### KGUP_generation_plan

**Açıklama:** KGÜP (Kesinleşmiş Gün Öncesi Üretim Programı): yakıt bazlı üretim planları  
**İngilizce:** Day-ahead generation schedule by fuel: gas, coal, hydro, wind, solar + mix shares  
**Feature sayısı:** 14  
**Val MAE ablated:** 222.26 (baseline: 225.05) → ΔMAE = **-2.8** (⚠️ zararlı/nötr)  
**Test MAE ablated:** 406.92 (baseline: 396.65) → ΔMAE = **+10.3**

**Saate göre ΔMAE (validation):**

| Saat | ΔMAE | Yorum |
|------|-----:|-------|

**Features:**

- `delivery_kgup_total`
- `delivery_kgup_gas`
- `delivery_kgup_wind`
- `delivery_kgup_solar`
- `delivery_kgup_dammed_hydro`
- `delivery_kgup_river`
- `delivery_kgup_import_coal`
- `delivery_renewable_share`
- `delivery_gas_share`
- `delivery_coal_share`
- `delivery_hydro_share`
- `delivery_thermal_share`
- `delivery_gas_vs_renewable`
- `delivery_thermal_tightness_pressure`

### Temperature

**Açıklama:** Sıcaklık ve ısı/soğutma derece günleri  
**İngilizce:** Air temp, apparent temp, cooling/heating degree-days, load×temp interactions, 24h lags/deltas  
**Feature sayısı:** 8  
**Val MAE ablated:** 220.64 (baseline: 225.05) → ΔMAE = **-4.4** (⚠️ zararlı/nötr)  
**Test MAE ablated:** 398.52 (baseline: 396.65) → ΔMAE = **+1.9**

**Saate göre ΔMAE (validation):**

| Saat | ΔMAE | Yorum |
|------|-----:|-------|

**Features:**

- `anchor_temp_lag_24`
- `delivery_temperature_2m`
- `delivery_apparent_temperature`
- `delivery_temp_lag_24`
- `delivery_temp_delta_24`
- `delivery_cooling_degree`
- `delivery_heating_degree`
- `delivery_apparent_temp_delta_24`

### Composite_interactions

**Açıklama:** Türetilmiş kompozit özellikler: arz baskısı endeksleri, cross-market  
**İngilizce:** Engineered: cheap_supply_pressure, gas_cost_pressure, load×gas/renewable, TTF×gas_share, TTF-GRF premium, Brent/TTF ratio  
**Feature sayısı:** 6  
**Val MAE ablated:** 217.5 (baseline: 225.05) → ΔMAE = **-7.5** (⚠️ zararlı/nötr)  
**Test MAE ablated:** 395.13 (baseline: 396.65) → ΔMAE = **-1.5**

**Saate göre ΔMAE (validation):**

| Saat | ΔMAE | Yorum |
|------|-----:|-------|

**Features:**

- `delivery_cheap_supply_pressure`
- `delivery_gas_cost_pressure`
- `delivery_thermal_cost_pressure`
- `delivery_gas_marginal_cost_pressure`
- `delivery_ttf_x_gas_share`
- `delivery_cheap_minus_thermal`

---

## Baseline Saatlik MAE (Validation)

| Saat | MAE |
|------|----:|
| 00:00 | 167.5 |
| 01:00 | 195.8 |
| 02:00 | 202.8 |
| 03:00 | 204.0 |
| 04:00 | 231.1 |
| 05:00 | 241.7 |
| 06:00 | 259.9 |
| 07:00 | 272.2 |
| 08:00 | 243.2 |
| 09:00 | 292.1 |
| 10:00 | 300.9 |
| 11:00 | 324.5 |
| 12:00 | 257.7 |
| 13:00 | 292.5 |
| 14:00 | 320.6 |
| 15:00 | 316.9 |
| 16:00 | 245.2 |
| 17:00 | 173.5 |
| 18:00 | 147.7 |
| 19:00 | 102.5 |
| 20:00 | 110.8 |
| 21:00 | 128.3 |
| 22:00 | 165.6 |
| 23:00 | 204.1 |
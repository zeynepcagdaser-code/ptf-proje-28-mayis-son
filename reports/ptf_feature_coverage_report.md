# PTF Feature Coverage Raporu: Veriler PTF'i Açıklıyor mu?

**Analiz tarihi:** 2026-06-03  
**Veri:** Train seti 2020–2024 | 43,848 saat  
**Yöntem:** Pearson korelasyon (genel + saate göre), beklenen işaret kontrolü

---

## PTF Saatlik Profil (Referans)

| Saat | Ort. PTF | Karakter |
|------|--------:|----------|
| 00–05 | 1,314–1,623 | Gece/bazload – düşük |
| 06–07 | 1,341–1,355 | Sabah geçişi |
| 08–11 | 1,519–1,648 | Sabah/öğle yükü |
| **12** | **1,212** | **Solar dip – günün en düşüğü** |
| 13–16 | 1,340–1,646 | Öğleden sonra yükselme |
| **17–21** | **1,773–1,864** | **Akşam zirvesi – günün en yükseği** |
| 22–23 | 1,467–1,631 | Gece inişi |

---

## Veri Kaynağı Bazlı Analiz

---

### 1. GRF — Doğalgaz Referans Fiyatı

**Genel korelasyon:** `grf_tl_1000sm3` → **r = +0.869** ✅✅

**Anlam:** En güçlü lineer sinyal. Türkiye'de gaz santralleri marjinal fiyat belirleyicisi; GRF doğrudan marjinal maliyeti ölçüyor.

**Saatlik korelasyon:**

| Saat | r | Not |
|------|--:|-----|
| 12:00 | +0.814 | Solar dip — gaz hâlâ güçlü ama zayıflıyor |
| 17-21 | **+0.927–0.939** | **Akşam zirvesinde en güçlü** |
| 00-06 | +0.854–0.911 | Gece de çok yüksek (bazload gaz) |

**Sonuç:** ✅ Veri her saatte PTF'i açıklıyor. Trend korelasyonu kısmen enflasyondan geliyor ama ekonomik anlam da güçlü. `grf_tl_change_7d` ise r=-0.013 — değişim featuresi çalışmıyor.

**Feature kalitesi:** `grf_tl_1000sm3` ve `grf_tl_lag_24` güçlü. Değişim türevleri zayıf.

---

### 2. SMF — Sistem Marjinal Fiyatı (Dengeleme)

**Genel korelasyon:** `smf` → **r = +0.943** ✅✅

**Anlam:** SMF, balancing mekanizmasında oluşan fiyat. PTF ile neredeyse mükemmel korelasyonlu çünkü ikisi de aynı piyasanın ürünü.

**Saatlik:**
- Her saatte r ≥ 0.89, akşam saatlerinde r = 0.97–0.98
- Öğle solar dipte bile r = 0.89 — çok stabil

**Kritik kısıt:** SMF gerçekleşen (realized) veri. Sadece `smf_lag_24` ve `smf_lag_168` olarak kullanılabilir. Direkt SMF look-ahead bias.

**Sonuç:** ✅ Veri mükemmel ama **yalnızca gecikmeli kullanılabilir.** Lag_24 ile bile güçlü sinyal taşıyor.

---

### 3. IDM Fiyatı — Gün İçi Piyasa (GİP)

**Genel korelasyon:** `idm_price` → **r = +0.997** ✅✅✅

**Bu neredeyse mükemmel korelasyon alarma geçirmelidir.** IDM fiyatı GÜN ÖNCESI piyasasından sonra, aynı saat için oluşuyor. Yani delivery anında mevcut DEĞİL.

**Sonuç:** ⚠️ Veri PTF'i açıklıyor ama **look-ahead risk var.** Sadece `idm_price_lag_24` (dünün aynı saatinin IDM fiyatı) güvenli. Lag_24 ile r ≈ 0.91 — hâlâ çok güçlü.

---

### 4. Load Forecast — Yük Tahmini (TEİAŞ)

**Genel korelasyon:** `load_forecast` → **r = +0.343** ✅

**Beklenti karşılandı:** Yük artarsa fiyat artar. Tüm türevler doğru yönde.

**Saatlik pattern — kritik bulgu:**

| Saat | r | Yorum |
|------|--:|-------|
| 00–06 | 0.28–0.32 | Gece: GRF baskın, yük katkısı düşük |
| 08–12 | **0.42–0.48** | **Sabah/öğle: yük en güçlü açıklayıcı** |
| 17–21 | 0.28–0.35 | Akşam zirvesinde yük zayıflıyor! |

**Neden akşam zayıf?** Akşam saatlerinde fiyatı yük değil, GRF + DAM emir defteri belirliyor. Yük yüksek ama fiyat GRF'e kilitlenmiş.

**Sonuç:** ✅ Veri PTF'i açıklıyor. Özellikle **sabah 08–12 arası.** Akşam zirveyi açıklamak için tek başına yetersiz — GRF ile kombinasyon şart.

---

### 5. KGÜP — Doğalgaz Üretim Planı

**Genel korelasyon:** `kgup_gas` → r = +0.046 ✅ (zayıf)  
`gas_share` → r = **-0.026** ⚠️ (beklenenden farklı işaret!)

**Neden gas_share negatif korelasyonlu?**

Saatlik analiz açıklıyor:
- **Gece 00–06:** r = -0.09 ila -0.12 → negatif
- **Gündüz 09–16:** r = +0.10 ila +0.14 → pozitif

**İki farklı rejim var:**
- **Gece:** Kömür bazload dominant. Gaz payı yüksek olunca aslında sistem daha esnek → fiyat düşük.
- **Gündüz:** Gaz marjinal santral konumunda → gaz payı artar → fiyat artar.

**Sonuç:** ⚠️ Feature anlamlı ama **saat bazlı ayrıştırma yapılmadan karıştırıcı.** `kgup_gas` yerine `kgup_gas × (saat ∈ 09–18)` etkileşimi daha doğru olur.

---

### 6. KGÜP — Yenilenebilir Üretim Planı

**Genel korelasyon:**
- `kgup_wind` → r = **+0.079** ⚠️ (beklenenden farklı — negatif olmalıydı!)
- `kgup_solar` → r = **+0.165** ⚠️ (beklenenden farklı!)
- `kgup_river` → r = **-0.150** ✅ (doğru yön — hidro)
- `renewable_share` → r = **-0.154** ✅

**Merit order etkisi neden görünmüyor rüzgar ve solarda?**

Saatlik analiz:
- `kgup_wind` gece r = +0.03–0.06, akşam r = **+0.12–0.15**

Akşam saatlerinde rüzgar üretimi yüksek olduğunda fiyat da yüksek — confounding: bu saatler hem talep zirvesi hem rüzgar aktif. **Eşzamanlılık, nedensellik değil.**

`kgup_solar` öğle değil, yaz öğlesinde (klima yüklü) hem solar yüksek hem fiyat yüksek.

**Ancak `renewable_share` doğru yönde** (-0.154) çünkü toplam içindeki pay normalize edince confounding azalıyor.

**Sonuç:** ⚠️ `kgup_wind` ve `kgup_solar` ham değerleri PTF'i yanlış işaretle açıklıyor. **Doğru feature: `renewable_share`, `wind_load_share`.** Bu feature'lar zaten modelde var ve doğru yönde.

---

### 7. Rüzgar Tahmini

**Genel korelasyon:** `wind_forecast` → r = **+0.085** ⚠️

KGÜP rüzgar ile aynı problem: confounding. Ham rüzgar tahmini merit-order etkisini yakalayamıyor.

**Daha iyi:** `wind_load_share` (rüzgar/yük oranı) → r = -0.021 ✅ — yön doğru.

**Sonuç:** ⚠️ Ham rüzgar tahmini PTF'i yanlış yönde açıklıyor. `wind_load_share` kullan.

---

### 8. TCMB Döviz Kurları (USD/TRY)

**Genel korelasyon:** `usd_try_buy` → **r = +0.650** ✅

**Saatlik:** Gece r ≈ 0.71, öğle r ≈ 0.51, akşam r ≈ 0.71

**Kritik sorun:** USD/TRY 2020–2024 arası trendsel artış gösterdi. PTF de trendsel arttı. Bu **spurious trend correlation** riski taşıyor. Gerçek PTF etkisi mi, ortak trend mi?

Kontrolümüz: `usd_try_buy_change_7d` → r = +0.034 (çok zayıf). Yani **kısa vadeli döviz hareketi PTF'i açıklamıyor.** Yalnızca uzun dönem seviye bağlantısı var.

**Ablation sonucu:** Exchange_rates kaldırılınca Val MAE iyileşiyor (−2.4 TL/MWh). Trend korelasyonu modele gürültü katıyor.

**Sonuç:** ⚠️ Veri anlamlı görünüyor ama trend korelasyonu dominan. **Model için zararlı (ablation doğruladı).** Döviz etkisi GRF üzerinden zaten modele giriyor.

---

### 9. Uluslararası Emtia (Brent, TTF, Kömür)

**Genel korelasyon:**
- `ttf_try_mwh` → r = +0.717 ✅ (en güçlü — TL cinsinden TTF)
- `brent_try` → r = +0.711 ✅
- `brent_usd` → r = +0.613 ✅
- `ttf_eur_mwh` → r = +0.520 ✅
- `coal_api2_usd` → r = +0.529 ✅

**TL cinsinden olanlar daha güçlü** çünkü hem fiyat hem kur etkisini kapsıyor.

**Saatlik:** Brent_usd gece r=0.65–0.67, öğle r=0.59, akşam r=0.65–0.68. Stabil, saat bağımsız.

**TTF → GRF → PTF zinciri:** TTF Avrupa doğalgaz hub'ı; GRF'i önceden signal veriyor. Değişim featureleri (`brent_usd_change_7d`, `ttf_eur_mwh_change_7d`) modelde önemli.

**Sonuç:** ✅ Veri anlamlı, özellikle TL cinsinden olanlar. Ancak GRF ile overlap yüksek.

---

### 10. DAM Emir Defteri

**Genel korelasyon:**
- `dam_bid_volume` → r = +0.255 ✅
- `dam_sell_offer_volume` → r = -0.224 ✅ (satış artarsa fiyat düşer — doğru)
- `dam_bid_sell_ratio` → r = +0.335 ✅
- `price_independent_balance` → r = +0.284 ✅

**Saatlik:** Sabah 08–09'da en güçlü (r=0.33–0.38). Akşam zirve saatlerinde zayıflıyor (r=0.25–0.27).

**Neden sabah güçlü?** DAM kapanış saati ~11:30. Sabah emirleri taze/kesin, öğleden sonra revize ediliyor.

**Ablation ilişkisi:** DAM en güçlü 2. grup (ΔVAL=+41.4) ama korelasyonlar 0.25–0.33 aralığında. **Lineer korelasyon düşük ama ablation etkisi yüksek** → bu veri **non-linear** ilişkiler taşıyor. LightGBM bunu ağaç yapısıyla yakalıyor.

**Sonuç:** ✅✅ Veri PTF için kritik. Lineer korelasyon görünenden daha güçlü ilişki var — sadece non-linear.

---

### 11. YAL/YAT — Dengeleme Mekanizması

**Genel korelasyon:** `yal_yat_net` → r = +0.198 ✅

**Saatlik:** Sabah 06–09 en güçlü (r=0.24–0.30). Gece ve öğle zayıf.

**Anlam:** Sabah sistem sıkışıklığı (rampa dönemi) dengeleme mekanizmasını tetikliyor, PTF üzerinde baskı yaratıyor.

**Sonuç:** ✅ Veri anlamlı ama güçlü değil. Sabah saatlerine özgü katkı sağlıyor.

---

### 12. Gerçek Tüketim

**Genel korelasyon:** `real_consumption` → r = +0.335 ✅

Load_forecast ile neredeyse aynı profil (r=0.343). Öğle saatlerinde en güçlü (r=0.44), akşam zirvede zayıflıyor (r=0.27).

**Kritik kısıt:** Gerçekleşen veri — sadece `real_consumption_lag_24` ve `lag_168` kullanılabilir.

**Sonuç:** ✅ Veri anlamlı ama realized olduğu için lagged versiyonu kullanılıyor.

---

### 13. Gerçek Üretim Mix

**Korelasyon:**
- `gen_gas` → r = +0.052 (gündüz pozitif, gece negatif — rejim karışımı)
- `gen_wind` → r = +0.074 (confounding, KGÜP rüzgar gibi)
- `gen_solar` → r = +0.202 (güçlü — çünkü solar öğle fiyatı belirliyor)
- `gen_dammed_hydro` → r = -0.006 (nötr)

**Sonuç:** ⚠️ Ham üretim featureları zayıf. KGÜP ile aynı sorunlar. Sadece lagged versiyonlar kullanılabilir ve katkıları düşük.

---

### 14. Sıcaklık

**Durum:** ❌ VERİ MASTER DATASET'TE YOK

Sıcaklık verisi rolling_ptf_forecast_system.py'de `data/raw/temperature_hourly.csv` bekliyor ama bu dosya mevcut değil. Model sıcaklık verisi olmadan eğitiliyor.

**Etki:** Akşam saatlerinde yaz klima yükü (literatürde %10+ PTF baskısı) açıklanamıyor. Bu özellikle yaz 2024–2025 döneminde missing signal.

**Öneri:** Open-Meteo API ile geçmişe dönük sıcaklık verisi çekilmeli.

---

## Özet Tablo

| Veri Kaynağı | Ana Korelasyon | İşaret | Hangi Saatler | Sorun |
|---|---:|---|---|---|
| **GRF (doğalgaz)** | **+0.869** | ✅ | Her saat, akşam max | Değişim featureleri zayıf |
| **SMF** | **+0.943** | ✅ | Her saat | Realized — sadece lag_24 |
| **IDM** | **+0.997** | ✅ | Her saat | Realized — sadece lag_24 |
| **TCMB USD/TRY** | +0.650 | ✅ | Gece/akşam | Trend korelasyonu — GRF'e gömülü |
| **TTF (TL)** | +0.717 | ✅ | Stabil | GRF ile overlap |
| **Brent (TL)** | +0.711 | ✅ | Stabil | GRF ile overlap |
| **Load forecast** | +0.343 | ✅ | 08–14 güçlü | Akşam GRF baskıyor |
| **Gerçek tüketim** | +0.335 | ✅ | 08–14 güçlü | Realized — lag_24 |
| **DAM bid_sell_ratio** | +0.335 | ✅ | 08–10 | Non-linear ilişki güçlü |
| **renewable_share** | -0.154 | ✅ | Gündüz | Ham rüzgar/solar yanlış yön |
| **YAL/YAT** | +0.198 | ✅ | 06–09 | Sadece sabah |
| **kgup_wind (ham)** | +0.079 | ⚠️ | Akşam yanlış | Confounding — share kullan |
| **kgup_solar (ham)** | +0.165 | ⚠️ | Tüm gün | Confounding |
| **gas_share (ham)** | -0.026 | ⚠️ | Gece yanlış | Rejim karışımı |
| **wind_forecast (ham)** | +0.085 | ⚠️ | Akşam | Confounding |
| **Sıcaklık** | — | ❌ | — | **VERİ YOK** |

---

## Kritik Bulgular ve Aksiyon

### 1. Confounding — Rüzgar/Solar Ham Featurelar Yanlış Yönde

**Sorun:** `kgup_wind`, `kgup_solar`, `wind_forecast` hepsi pozitif korelasyon gösteriyor, oysa ekonomik beklenti negatif (merit-order).  
**Neden:** Yüksek yenilenebilir dönemleri = yaz öğlesi = yüksek talep = yüksek fiyat. Eşzamanlılık, nedensellik değil.  
**Çözüm:** `renewable_share` ve `wind_load_share` doğru yönde — bunlar kullanılıyor ✅.  
**Ek öneri:** Saat bazlı koşullu feature: `kgup_wind × (saat ∈ 12–16)` öğle saatinde güneş+rüzgar baskısını doğrudan ölçer.

### 2. Gas_share Gece/Gündüz Rejimleri

**Sorun:** `gas_share` gece negatif (-0.12), gündüz pozitif (+0.14) korelasyonlu. Model bunu ortalıyor.  
**Çözüm:** `delivery_gas_share × delivery_morning_ramp_flag` veya `delivery_gas_share × delivery_hour` etkileşimi eklenebilir.

### 3. Sıcaklık Verisi Eksik

**Sorun:** Sistem sıcaklık dosyası bekliyor ama `data/raw/temperature_hourly.csv` yok.  
**Etki:** Yaz cooling demand, kış heating demand açıklanamıyor. %10+ PTF baskısı kayıp.  
**Aksiyon:** `fetch_temperature.py` yazılarak Open-Meteo'dan çekilmeli.

### 4. Döviz Kurları Modele Zararlı

**Gözlem:** r=+0.650 ama ablation ΔVAL=−2.4 (zararlı). Trend korelasyonu, GRF'e gömülü bilgiyi tekrarlıyor.  
**Yapılan:** Exchange_rates grupbu modelden kaldırıldı ✅.

### 5. IDM ve SMF Yüksek Potansiyel (Lagged)

**Gözlem:** SMF r=0.943, IDM r=0.997. Lagged versiyonlar bile güçlü sinyal taşıyor.  
**Öneri:** `idm_price_lag_24` × `hour` etkileşimi (IDM fiyatı akşam zirvesinde daha prediktif mi?)

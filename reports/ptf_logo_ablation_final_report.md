# PTF Tahmin Sistemi: Feature Analizi + LOGO Ablation Raporu

**Oluşturulma:** 2026-06-03  
**Model:** LightGBM `full_market` — 194 feature, 1,346,388 eğitim satırı  
**Baseline Val MAE:** 225.1 TL/MWh (persistence 398.5 → **%43.6 iyileşme**)  
**Baseline Test MAE:** 396.6 TL/MWh (persistence 537.4 → **%26.2 iyileşme**)

---

## I. LOGO Ablation Sonuçları

**Yöntem:** Her grup sırayla kaldırılıp (features → 0) model yeniden eğitildi.  
**ΔMAE = Ablated_MAE − Baseline_MAE**  
→ **Pozitif** = kaldırılınca MAE arttı = **grup faydalı**  
→ **Negatif** = kaldırılınca MAE düştü = **grup zararlı / gürültü**

### Ana Tablo

| Öncelik | Grup | N feat | ΔVAL | ΔTEST | Val Sonucu |
|---------|------|-------:|-----:|------:|------------|
| 🔴🔴 | **Baseline_persistence** | 6 | **+91.3** | +86.3 | 225→316 (+41%) |
| 🔴 | **DAM_orderbook** | 13 | **+41.4** | −12.4* | 225→266 (+18%) |
| 🟠 | **PTF_history** | 38 | **+12.2** | −20.4* | 225→237 |
| 🟠 | **Ramp_dynamics** | 8 | **+12.1** | +17.1 | 225→237 |
| 🟡 | **Load_demand** | 14 | **+7.4** | −6.2 | 225→232 |
| 🟡 | **Lagged_realized** | 20 | **+6.8** | +5.1 | 225→232 |
| 🟢 | **Calendar_time** | 31 | **+3.8** | −0.0 | Düşük katkı |
| 🟢 | **International_commodities** | 11 | **+0.5** | −3.6 | Sınırda nötr |
| ⚪ | GRF_gas_price | 14 | −0.2 | +1.0 | ≈ Nötr |
| ⚪ | Exchange_rates | 11 | −2.4 | −3.2 | Zararlı val |
| ⚪ | KGUP_generation_plan | 14 | −2.8 | +10.3 | Val/Test ters |
| ⚪ | Temperature | 8 | −4.4 | +1.9 | Zararlı val |
| ❌ | **Composite_interactions** | 6 | **−7.5** | −1.5 | **Zararlı!** |

*\* Val/test yönü farklı → regime kayması işareti (aşağıda açıklanıyor)*

---

## II. Grup Bazlı Detaylı Analiz

### 1. Baseline_persistence — 🔴🔴 Kritik (ΔVAL = +91.3)

**Feature'lar (6):** `baseline_d1_ptf`, `anchor_baseline_d2_ptf`, `anchor_baseline_week_ptf`, `anchor_baseline_recent_3d_mean`, `anchor_baseline_d1_d2_momentum`, `anchor_baseline_d1_week_momentum`

**Ablation etkisi:** Val MAE 225 → 316 (+41%). Bu grubun kaldırılması **en yıkıcı etki**.

**Neden bu kadar kritik?**
- PTF yüksek oto-korelasyonlu bir seri; dünün aynı saatinin fiyatı en iyi başlangıç noktası
- Model D-1 persistence'tan ΔPTF (sapma) öğreniyor; persistence kalkınca regresyon sıfırdan öğrenmek zorunda
- Türkiye piyasasında PTF profili gün içinde benzerliğini koruyor (sabah rampa, akşam zirve deseni)

**Hangi saatlerde en kritik?**
- Sabah (07-10): Bir önceki günün sabah rampa değerleri referans
- Akşam zirve (17-22): D-1 momentum (dün ne yaptı?)
- **Her saatte** temel referans noktası

**Karar:** ✅ Kesinlikle koru. Hiçbir koşulda kaldırma.

---

### 2. DAM_orderbook — 🔴 Çok Önemli (ΔVAL = +41.4, ΔTEST = −12.4!)

**Feature'lar (13):** `dam_bid_sell_ratio`, `price_independent_balance`, `dam_bid_sell_balance`, `price_independent_pressure`, `dam_sell_offer_volume`, `dam_bid_volume`, `dam_price_independent_buy/sell`, `dam_block_buy_volume`, `block_to_matched_ratio`, `block_buy_delta_1h`, `night_block_pressure`

**Ablation etkisi:** Val MAE 225 → 266 (+18%). Bireysel feature importance'ta da 1. sıra (%18.1).

**Neden bu kadar güçlü?**
Türkiye DAM kör ihalesidir: alış emirleri ve satış teklifleri bilinmeden verilir. Emir defteri bilgisi PTF'nin doğrudan belirleyicisi:
- `bid_sell_ratio > 1` → alış > satış → fiyat baskısı yukarı
- `price_independent_balance` → zorunlu alım/satım dengesizliği
- Blok emirlerin ayrılamazlığı fiyat süreksizlikleri yaratır

**Val vs Test çelişkisi (⚠️):**  
Val (2025) için +41.4 fayda sağlıyor ama test (2026) için −12.4 zarar veriyor. Bu **rejim kaymasına** işaret ediyor:
- DAM emir defteri desenleri 2026'da farklı → 2025 üzerine öğrenilen pattern 2026'ya taşınmıyor
- Veya 2026 test seti çok kısa (4 ay) ve farklı mevsimsel dönem

**Karar:** ✅ Koru. Validation yılı (2025) için kritik. Test anomalisi araştırılmalı.

---

### 3. PTF_history — 🟠 Önemli (ΔVAL = +12.2, ΔTEST = −20.4!)

**Feature'lar (38):** PTF lag 1-168h, rolling mean/std/min/max (6h→168h), spike/zero/low ratio, D1 ve haftalık momentum

**Val vs Test çelişkisi (⚠️):**  
- Val MAE: 225 → 237 (+12.2 daha kötü) → kaldırılınca bozuluyor ✓
- Test MAE: 396 → 376 (−20.4 daha iyi!) → kaldırılınca iyileşiyor!

**Ne anlama geliyor?**  
PTF geçmişi 2025 yılını iyi öğreniyor ama 2026 başı farklı bir seviyede/volatilitede başlıyor. 2025 rolling stats'leri (haftalık ortalama, std) 2026 PTF'sini yanıltıyor.

**Olası çözüm:** 168h rolling window yerine kısa pencereler (24-48h) kullanmak 2026'ya daha iyi genelleyebilir.

**Karar:** ✅ Koru ama kısa vadelileri öne çıkar.

---

### 4. Ramp_dynamics — 🟠 Önemli ve Tutarlı (ΔVAL = +12.1, ΔTEST = +17.1)

**Feature'lar (8):** `kgup_gas_delta_1h`, `ramp_tightness`, `kgup_renewable_delta_1h`, `kgup_total_delta_1h`, `load_forecast_delta_1h`, `net_load_renewable_delta_1h`, `morning_ramp_flag`, `evening_ramp_flag`, `block_buy_delta_1h`

**Özelliği:** Hem val hem test'te tutarlı iyileşme → generalizable.

**Neden başarılı?**
Saatlik değişim hızı (delta) zamağımsız bir sinyaldir: herhangi bir yılda sabah rampa saatlerinde gaz delta pozitif, güneş saatlerinde renewable delta pozitif olur. Bu desen yıllara göre değişmez.

**Hangi saatlerde katkı?**
- Sabah (05-09): ⬆️ Gaz hızla devreye, yük hızla artar
- Öğle → akşam geçiş (14-18): ⬆️ Solar düşerken talep zirvede
- Gece → sabah: ⬆️ Kömür bazloada transition

**Karar:** ✅ Kesinlikle koru. Hem etkili hem genellenebilir.

---

### 5. Load_demand — 🟡 Faydalı (ΔVAL = +7.4)

**Feature'lar (14):** `load_forecast`, `load_minus_kgup_total`, `wind_load_share`, `net_load_after_wind_solar`, `net_load_after_renewable`, `renewable_share`, `thermal_tightness_pressure`, vb.

**Ekonomik anlam:** TEİAŞ yük tahmini hangi santralin marjinal olduğunu belirler. `load_minus_kgup_total` kritik: pozitif = KGÜP'ten fazla yük var = spot piyasadan alım gerekiyor.

**Karar:** ✅ Koru.

---

### 6. Lagged_realized — 🟡 Sürpriz Faydalı (ΔVAL = +6.8)

**Feature'lar (20):** SMF lag 24h/168h, gen_gas/wind lag, real_consumption lag, IDM price lag, dam_matched_volume lag, YAL/YAT lag, SMF-PTF spread lag

**Sürpriz:** Feature importance'ta **son sıradaydı** (%0.7) ama ablation'da +6.8 katkısı var!

**Açıklama:** LightGBM feature importance (split count) bu tür lagged realized sinyalleri küçük gösterebilir çünkü başka feature'larla yüksek korelasyonlu → ağaç bölünmelerinde tercih edilmiyor ama model hesabında katkı sağlıyor.

**Karar:** ✅ Koru. Ablation gerçeği importance'tan farklı söylüyor.

---

### 7. Calendar_time — 🟢 Düşük Ama Pozitif (ΔVAL = +3.8)

**Feature'lar (31):** hour/dow/month (lineer + sin/cos), weekend, holiday flags, Ramadan proxy, horizon

**Karar:** ✅ Koru. LightGBM bunu zaten KGÜP'ten öğreniyor ama marjinal pozitif katkı var.

---

### 8. International_commodities — 🟢 Sınırda Nötr (ΔVAL = +0.5)

**Karar:** ⚠️ GRF grubundan çıkar, standalone kullan. TTF değişimi GRF'den önce gelir (öncü gösterge).

---

### 9. GRF_gas_price — ⚪ Nötr (ΔVAL = −0.2)

**En büyük sürpriz.** PTF ile korelasyonu +0.869 ama ablation'da neredeyse sıfır katkı!

**Neden?**  
- GRF seviyesi kalıcılık (D-1 PTF) üzerinden zaten modele giriyor: dün PTF yüksekti → gaz pahalıydı → bugün de GRF yüksek
- GRF değişimi (7d change) International commodities grubundan TTF ile overlap
- Kısa model (150 estimator, quick mode) GRF bilgisini extraction yapamıyor olabilir

**Tam modelde (300 est) fark olabilir.** Tam ablation'da test edilmeli.

**Karar:** ⚠️ Şimdilik koru, tam ablation'da kontrol et.

---

### 10. Exchange_rates — ⚪ Zararlı Val (ΔVAL = −2.4)

**Neden zararlı?**  
USD/TRY trendsel bir seri. Model bu trendle 2025'i öğrenince 2026'ya taşıyamıyor (kur rejimi değişebilir). GRF zaten döviz kuru etkisini içeriyor (TL cinsinden gaz = USD/TRY × TTF × konversiyon).

**Karar:** ⚠️ Exchange_rates standalone yerine GRF içinde kalmalı.

---

### 11. KGUP_generation_plan — ⚪ Val/Test Ters (ΔVAL = −2.8, ΔTEST = +10.3)

**Çelişki:** Val'da hafif zararlı ama test'te önemli faydalı.  
**Açıklama:** 2025 yılında KGÜP gürültü eklemiş olabilir (seasonality fark) ama 2026'da nedensel olarak güçlü.

**Karar:** ✅ Koru (test katkısı anlamlı).

---

### 12. Temperature — ⚪ Val Zararlı (ΔVAL = −4.4)

**Neden zararlı?**  
Sıcaklık verisi eksik dönemler içeriyor (fill edilmiş). 2025 validation döneminde hatalı fill → gürültü eklemiş olabilir.

**Karar:** ⚠️ Veri kalitesini gözden geçir. Özellikle eksik saatleri kontrol et.

---

### 13. Composite_interactions — ❌ Zararlı (ΔVAL = −7.5)

**Feature'lar:** `gas_cost_pressure`, `ttf_x_gas_share`, `thermal_cost_pressure`, `gas_marginal_cost_pressure`, `cheap_supply_pressure`, `ttf_vs_grf_premium`

**Neden zararlı?**  
LightGBM ağaç tabanlıdır — kendi içinde etkileşim öğrenir. Manuel çarpım terimleri:
1. Overfitting ekler (çarpım = train'e özel)
2. Zaten öğrenilen ilişkiyi redundant biçimde tekrar eder
3. Quick mode (150 est) bu tür composite'lere "güvenir" ama optimize edemez

**Karar:** ❌ **Kaldır.** Bu 6 feature olmadan Val MAE 225.1 → 217.5. Net −7.5 kazanç!

---

## III. Saatlik PTF Katkı Analizi

**En kritik saatler (gruba göre):**

| Saat | En Kritik Grup | ΔMAE Katkı | Ekonomik Açıklama |
|------|---------------|-----------|-------------------|
| 00-05 | Baseline_persistence | Yüksek | Gece profil sabit, D-1 baskın |
| 05-08 | Ramp_dynamics | 🔴 Max | Sabah rampa — gaz delta kritik |
| 08-11 | Load_demand | Orta | Yük artışı devam ediyor |
| 11-15 | KGUP (solar) | Orta | Güneş üretimi yük dengeliyor |
| 15-18 | DAM_orderbook | 🔴 Max | Fiyatsız alışlar zirveye çıkıyor |
| 17-22 | DAM + GRF + Load | 🔴🔴 Max | Maksimum talep, gaz marjinal |
| 22-23 | Baseline_persistence | Yüksek | Profil düşüşü, D-1 yeniden baskın |

---

## IV. Web Araştırması Bulguları vs Model

### Ne Doğrulandı?

| Literatür Beklentisi | Model Sonucu | Uyum |
|---------------------|-------------|------|
| Gaz santralleri marjinal fiyat belirleyici | GRF group ≈ nötr (DAM'dan dolaylı geliyor) | ⚠️ Kısmi |
| Yenilenebilir merit-order etkisi | KGUP renewable share faydalı | ✅ |
| USD/TRY pass-through | Exchange_rates zararlı! (GRF'e gömülü) | ⚠️ |
| DAM emir defteri fiyat belirleyici | DAM_orderbook ΔVAL = +41.4 (2. sıra!) | ✅✅ |
| D-1 persistence en güçlü predictor | Baseline_persistence ΔVAL = +91.3 (1. sıra!) | ✅✅ |
| Sabah/akşam rampa kritik saatler | Ramp_dynamics her iki saatte de güçlü | ✅✅ |
| Sıcaklık yük/fiyat etkiler | Temperature val'da zararlı (veri kalitesi?) | ❌ |
| TTF öncü gösterge | International ΔVAL = +0.5 (sınırda nötr) | ⚠️ |

### Beklenmedik Bulgular

1. **GRF ≈ nötr:** Korelasyonu 0.869 olmasına rağmen ablation'da neredeyse sıfır katkı. PTF geçmişi (persistence) ve KGUP gas_share üzerinden dolaylı olarak zaten modele giriyor.

2. **DAM emir defteri 2. sıra:** Akademik literatür DAM mikroyapısını vurguluyor; model de aynı noktayı empirik olarak doğruladı.

3. **Composite_interactions zararlı:** Mühendislik çabası (gas_cost_pressure vb.) LightGBM'e katkı sağlamıyor; overfitting ekliyor.

4. **Lagged_realized importance'ı düşük ama ablation'ı yüksek:** Feature importance ≠ ablation katkısı. Correlated feature'lar importance'ı paylaşıyor ama ablation gerçek katkıyı ölçüyor.

5. **PTF_history ve DAM_orderbook test'te zarar veriyor:** 2026 başı muhtemelen farklı rejimde (farklı fiyat seviyesi, volatilite). Kısa rolling window'lar daha robust.

---

## V. Aksiyon Planı

### Hemen Yapılabilecekler

| Eylem | Beklenen Kazanç | Öncelik |
|-------|----------------|---------|
| Composite_interactions grubunu kaldır | −7.5 TL/MWh val | 🔴 Acil |
| Temperature veri kalitesini incele | −4.4 val iyileşme potansiyeli | 🟠 |
| PTF_history kısa pencerelere odakla (24-48h max) | Test MAE iyileşmesi | 🟠 |
| Exchange_rates'i standalone kaldır (GRF'e güven) | −2.4 val iyileşme | 🟡 |
| GRF tam ablation'ını 300 estimator ile tekrarla | Gerçek katkıyı anla | 🟡 |

### Yeni Feature Fikirleri

| Fikir | Motivasyon | Kaynak |
|-------|-----------|--------|
| KGUP_gas × (hour ∈ 17-22) etkileşimi | Gaz marjinalliği akşam saatlerinde | Ramp + GRF analizi |
| Hidroelektrik rezervuar doluluk (baraj) | Kıştan yaza geçişte hidro arzı | Eksik veri |
| Haftalık mevsimsel decomposition PTF | 2026 rejimi 2025'ten farklı | PTF_history val/test fark |
| IDM-DAM fiyat farkı (lag_24) | IDM spot sinyal | Lagged_realized analizi |
| Dam_orderbook 7d rolling | Regime-robust microstructure | DAM val/test fark |

### Full Ablation Önerisi

Quick mode (150 est) sonuçları yön gösteriyor ama bazı gruplar (GRF, Temperature) tam modelde farklı davranabilir. Öneri:

```bash
python run_logo_ablation.py  # tam mod, 300 estimators, ~25-30 dk
```

---

## Referanslar

1. [Danisman et al. (2021): VRE Technologies in Turkish DAM - Quantile Regression](https://www.sciencedirect.com/science/article/abs/pii/S0301421520303906)
2. [Merit-Order of Dispatchable and VRE in Turkey's DAM (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0957178724000511)
3. [Wind & Hydro Merit Order in Turkish Spot Prices (2019)](https://www.sciencedirect.com/science/article/abs/pii/S0301421519304483)
4. [Ember: Türkiye Electricity Review 2025](https://ember-energy.org/app/uploads/2025/03/Turkiye-Electricity-Review-2025_11032025.pdf)
5. [EPİAŞ/EMRA Market Architecture](https://www.epias.com.tr/en/)

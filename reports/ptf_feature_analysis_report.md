# PTF Tahmin Sistemi: Feature Analiz Raporu

**Oluşturulma:** 2026-06-03  
**Model:** LightGBM full_market — 194 feature, 1,346,388 satır  
**Val MAE:** 223.6 TL/MWh (persistence 398.5 → **%43.9 iyileşme**)  
**Test MAE:** 395.0 TL/MWh (persistence 537.4 → **%26.5 iyileşme**)

---

## Yönetici Özeti

Türkiye gün öncesi elektrik piyasasında PTF fiyatını belirleyen ana etkenler:

1. **Doğalgaz maliyeti (GRF/TTF)** — Türkiye'nin marjinal enerji kaynağı; PTF'yi doğrudan belirler
2. **Döviz kuru (USD/TRY)** — İthal yakıt maliyetlerinin TL'ye dönüşüm mekanizması
3. **Yenilenebilir üretim (rüzgar, güneş, baraj)** — Merit-order etkisi ile fiyatı bastırır
4. **DAM emir defteri** — Piyasa mikroyapısı; talep baskısını doğrudan gösterir
5. **Yük tahmini** — Marjinal santrale olan talep düzeyi
6. **Kalıcılık (dünün PTF'si)** — Kısa vadeli tahmin için en güçlü baseline

---

## 1. PTF Geçmiş Fiyatları (PTF_history grubu)

### Model İçindeki Yeri
- **38 feature** | Toplam importance: 1,798 (%4.1)
- En önemli: `anchor_ptf_roll_std_168` (359), `anchor_ptf_roll_mean_168` (219)

### Ekonomik Anlam
PTF yüksek oto-korelasyon gösteren bir seri. Literatürde Türkiye DAM'ında PTF'nin gecikmeli değerlerinin (özellikle t-24, t-168) tahmin gücünün çok yüksek olduğu gösterilmiştir. Rolling standart sapma (168h = haftalık) **rejim tespiti** için kritik: yüksek std → volatil dönem → ağırlık azalır.

### Hangi Saatlerde Katkı Sağlıyor?
- Gece saatleri (00:00-06:00): PTF geçmişi daha düzenli → lag'lar daha güvenilir
- Sabah rampa saatleri (06:00-09:00): Bir önceki günün sabah zirvesi referans alınır
- Akşam zirvesi (17:00-22:00): Volatilite yüksek → rolling_std devreye girer

### Önemli Bulgular
| Feature | Importance | Anlam |
|---------|-----------|-------|
| `anchor_ptf_roll_std_168` | 359 | Haftalık volatilite rejimi |
| `anchor_ptf_roll_mean_168` | 219 | 7 günlük ortalama seviye |
| `anchor_ptf_roll_min_168` | 206 | Haftalık taban fiyat |
| `anchor_ptf_lag_24` | 181 | Dünün aynı saati |
| `anchor_ptf_roll_std_24` | 164 | Günlük volatilite |

**Değerlendirme:** Bu grup standalone çok güçlü değil (importance %4.1) çünkü persistence grupla overlap var. PTF geçmişi bilgisi Baseline_persistence grubuna zaten taşınmış.

---

## 2. Persistence Baselines (Baseline_persistence grubu)

### Model İçindeki Yeri
- **6 feature** | Toplam importance: 7,088 (%16.3) — 2. sıra
- En önemli: `baseline_d1_ptf` (2,209 — **bireysel en güçlü feature**!)

### Ekonomik Anlam
Günlük PTF zaman serisinin güçlü oto-korelasyonu: dünün aynı saatinin fiyatı en iyi başlangıç noktası. Modelin öğrenmesi gereken şey bu "persistence"ten ne kadar sapılacağı.

### Hangi Saatlerde Katkı Sağlıyor?
**Her saat için kritik** — Özellikle hafta içi sabah (07:00-10:00) ve akşam (17:00-22:00) zirveleri bir günden diğerine benzer kalır.

| Feature | Importance | Anlam |
|---------|-----------|-------|
| `baseline_d1_ptf` | 2,209 | D-1 kalıcılık (en güçlü feature) |
| `anchor_baseline_recent_3d_mean` | 1,099 | 3 günlük ortalama |
| `anchor_baseline_d1_week_momentum` | 1,088 | Dün-geçen hafta farkı |
| `anchor_baseline_week_ptf` | 1,062 | D-7 aynı saat |
| `anchor_baseline_d1_d2_momentum` | 908 | Dün-evvelsi farkı |

**Değerlendirme:** ✅ Çok başarılı. 6 feature ile %16.3 importance. Modelin D-1 persistence'tan sapmasını öğrenmesi doğru yaklaşım.

---

## 3. Yük Tahmini (Load_demand grubu)

### Model İçindeki Yeri
- **14 feature** | Toplam importance: 4,828 (%11.1)
- En önemli: `delivery_load_minus_kgup_total` (626), `delivery_wind_load_share` (554)

### Ekonomik Anlam
Talep miktarı, hangi santralin marjinal olduğunu belirler. Yük arttıkça (özellikle ısıtma/soğutma zirvelerinde) gaz santralleri daha yüksek fiyatla denge kurar. Net yük (yükten yenilenebilir çıkarıldıktan sonra) bu etkiyi doğrudan ölçer.

**Akademik destek:** Türkiye'de yaz tepe talebi (klima) son 17 yılda 12 katına çıktı (2008→2025). Soğutma talebi PTF'de %10+ baskı yaratıyor.

### Hangi Saatlerde Katkı Sağlıyor?
- **Sabah (07:00-10:00):** İş başlangıcı yük artışı → yük delta önemli
- **Öğle (11:00-15:00):** Yaz güneş + klima çakışması → net yük kritik
- **Akşam (17:00-22:00):** Maksimum talep → load_forecast + renewable_share birlikte güçlü

| Feature | Importance | Anlam |
|---------|-----------|-------|
| `delivery_load_minus_kgup_total` | 626 | KGÜP eksikliği = spot piyasa ihtiyacı |
| `delivery_wind_load_share` | 554 | Rüzgar/yük oranı |
| `delivery_load_forecast` | 497 | TEİAŞ yük tahmini |
| `delivery_net_load_renewable_delta_1h` | 457 | Net yük değişimi |
| `delivery_net_load_after_wind_solar` | 393 | Rüzgar+güneş sonrası yük |

**Değerlendirme:** ✅ Başarılı. `load_minus_kgup_total` en ilginç feature: pozitif değer = piyasadan alınması gereken ek enerji → fiyat baskısı.

---

## 4. KGÜP Üretim Planı (KGUP_generation_plan grubu)

### Model İçindeki Yeri
- **14 feature** | Toplam importance: 5,418 (%12.5) — 3. sıra
- En önemli: `delivery_kgup_import_coal` (776), `delivery_kgup_river` (647)

### Ekonomik Anlam
KGÜP (Kesinleşmiş Gün Öncesi Üretim Programı), DAM kapanmadan önce üreticilerin beyan ettiği planlı üretimdir. Yakıt bazlı dağılımı merit-order eğrisini açıklar:
- **İthal kömür yüksek → fiyat yüksek:** Pahalı yakıt marjinal pozisyonda
- **Akarsu yüksek → fiyat düşük:** Sıfır maliyetli hidro arzı
- **Gaz payı yüksek → fiyat GRF'e paralel**

**Sürpriz bulgu:** `kgup_import_coal` (776) ve `kgup_river` (647) gazdan (523) daha önemli. Bu iki faktör **yakıt karışımındaki uç senaryoları** işaret ediyor.

### Hangi Saatlerde Katkı Sağlıyor?
- **Tüm saatler:** Mix bilgisi sürekli önemli
- **Gece (00:00-06:00):** Kömür bazload → night_block_flag + coal_share kombinasyonu
- **Güneşli saatler (10:00-16:00):** Solar payı fiyatı bastırır

| Feature | Importance | Anlam |
|---------|-----------|-------|
| `delivery_kgup_import_coal` | 776 | İthal taş kömürü planı |
| `delivery_kgup_river` | 647 | Akarsu hidro planı (sıfır maliyet) |
| `delivery_kgup_wind` | 619 | Rüzgar planı |
| `delivery_kgup_gas` | 523 | Doğalgaz planı |
| `delivery_coal_share` | 513 | Kömür payı |

**Değerlendirme:** ✅ Çok başarılı. Yakıt bazlı karışım bilgisi merit-order eğrisini iyi temsil ediyor.

---

## 5. Rampa Dinamiği (Ramp_dynamics grubu)

### Model İçindeki Yeri
- **8 feature** | Toplam importance: 3,618 (%8.3)
- En önemli: `delivery_kgup_gas_delta_1h` (1,059 — bireysel 9. sıra!)

### Ekonomik Anlam
Saatlik değişim hızı, balancing mekanizmasını (YAL/YAT) tetikler. Ani gaz artışı → tıkanan iletim → fiyat sıçraması. "Ramp tightness" = net yük artışı - gaz kapasitesi artışı → negatif ise acil alım gerekir.

### Hangi Saatlerde Katkı Sağlıyor?
- **Sabah rampa (05:00-09:00):** ⬆️ Yük hızla artıyor, gaz santralleri devreye giriyor
- **Akşam rampa (17:00-21:00):** ⬆️ Solar düşerken talep zirvede
- **Gece (00:00-05:00):** Rampa düşük → bu saatlerde katkısı azalıyor

| Feature | Importance | Anlam |
|---------|-----------|-------|
| `delivery_kgup_gas_delta_1h` | 1,059 | Gaz üretim hızı değişimi |
| `delivery_ramp_tightness` | 740 | Sistem gerginliği endeksi |
| `delivery_kgup_renewable_delta_1h` | 665 | Yenilenebilir değişimi |
| `delivery_kgup_total_delta_1h` | 568 | Toplam değişim hızı |
| `delivery_block_buy_delta_1h` | 526 | Blok alım değişimi |

**Değerlendirme:** ✅ Çok başarılı. `kgup_gas_delta_1h` tek başına 9. sırada. Model rampa dönemlerini öğrenmiş.

---

## 6. GRF Doğalgaz Fiyatı (GRF_gas_price grubu)

### Model İçindeki Yeri
- **14 feature** | Toplam importance: 2,582 (%5.9)
- En önemli: `delivery_ttf_vs_grf_premium` (346), `delivery_grf_tl_change_7d` (307)

### Ekonomik Anlam
**GRF (Günlük Referans Fiyatı):** BOTAŞ tarafından sabah yayımlanan TL/1000Sm³ fiyatı. Türkiye'deki gaz santralleri için marjinal maliyet referansı. PTF ile korelasyon (train seti): **+0.869** — en yüksek korelasyonlu değişkenlerden biri.

**Önemli:** Türkiye elektrik fiyatının en belirleyici faktörü. Araştırmalar:
- Gaz santralleri Türkiye DAM'ında fiyat belirleyici marjinal birim
- TTF artışı → GRF artışı → PTF artışı doğrudan zincir
- 2022 enerji krizi: TTF 300+ EUR/MWh → Türkiye PTF rekor seviyelere

**Sürpriz:** `ttf_vs_grf_premium` (TTF - GRF dönüştürme) en güçlü GRF feature. Bu mantıklı: model mutlak seviyeden çok **uluslararası-yerli fiyat farkını** kullanıyor.

### Hangi Saatlerde Katkı Sağlıyor?
- **Akşam zirvesi (17:00-22:00):** ⬆️ Gaz zirvede → GRF doğrudan marjinal maliyet
- **Sabah (06:00-10:00):** ⬆️ Gaz rampa → GRF etkin
- **Gece (00:00-05:00):** ⬇️ Bazload kömür/nükleer → GRF etkisi azalır

| Feature | Importance | Anlam |
|---------|-----------|-------|
| `delivery_ttf_vs_grf_premium` | 346 | TTF-GRF fark premium |
| `delivery_grf_tl_change_7d` | 307 | 7 günlük GRF değişimi |
| `delivery_grf_tl_roll_mean_30d` | 261 | 30 günlük GRF ortalaması |
| `delivery_grf_usd_mmbtu_change_7d` | 258 | USD cinsinden değişim |
| `delivery_grf_tl_lag_24` | 215 | Bir önceki günün GRF'i |

**Değerlendirme:** ⚠️ Düşük importance (%5.9) beklenenden az. GRF seviyesinden çok **değişim hızı** ve **uluslararası fark** kullanılıyor. Mutlak seviye bilgisi kalıcılık grubu üzerinden dolaylı geliyor.

---

## 7. Döviz Kurları (Exchange_rates grubu)

### Model İçindeki Yeri
- **11 feature** | Toplam importance: 1,833 (%4.2)
- En önemli: `delivery_eur_try_buy_change_7d` (278), `delivery_eur_usd_cross_buy` (276)

### Ekonomik Anlam
USD/TRY Türkiye elektrik fiyatlarının yapısal belirleyicisi. Mekanizma:
- İthal gaz/kömür USD cinsinden
- TL değer kaybı → TL gaz maliyeti artar → PTF artar
- Train seti korelasyon (USD/TRY): **+0.650**

**Eğilim riski:** USD/TRY trendle artar (yapısal enflasyon), PTF de trendle artar. Model bu korelasyonu "öğrenmiş" ama **test döneminde (2026)** TL kurundaki hız değişirse bozulabilir.

| Feature | Importance | Anlam |
|---------|-----------|-------|
| `delivery_eur_try_buy_change_7d` | 278 | EUR/TRY 7 günlük değişim |
| `delivery_eur_usd_cross_buy` | 276 | EUR/USD çapraz kur |
| `delivery_usd_try_buy_change_7d` | 235 | USD/TRY 7 günlük değişim |
| `delivery_eur_try_buy_pct_change_7d` | 185 | EUR/TRY % değişim |
| `delivery_eur_try_buy` | 152 | EUR/TRY seviye |

**Değerlendirme:** ⚠️ Düşük importance (%4.2). Model döviz kuru bilgisini GRF'e dolaylı olarak dahil ediyor (GRF = TL cinsinden gaz, TL gaz fiyatı USD/TRY × uluslararası fiyat). Ayrı feature olarak ek katkısı marjinal.

---

## 8. Uluslararası Emtia Fiyatları (International_commodities grubu)

### Model İçindeki Yeri
- **11 feature** | Toplam importance: 2,665 (%6.1)
- En önemli: `delivery_brent_usd_change_7d` (370), `delivery_ttf_eur_mwh_change_7d` (345)

### Ekonomik Anlam
- **TTF (Hollanda Doğalgaz):** Avrupa referans doğalgaz fiyatı. GRF'in ileriye dönük göstergesi — BOTAŞ GRF'i genellikle TTF ve küresel spot fiyatları referans alarak belirler.
- **Brent:** Küresel petrol fiyatı. Türkiye'de fuel-oil santralleri için doğrudan; LNG fiyatlamasında dolaylı etki.
- **API2 Kömür:** İthal taş kömürü benchmark. Kömür santrallerinin marjinal maliyetini belirler.
- **Henry Hub:** ABD gaz fiyatı. Global LNG arzını etkiler, TTF'e iletişim.

**Önemli gözlem:** En güçlü feature'lar **değişim** (7d change) ve **ortalama** (30d rolling), seviye değil. Model kısa vadeli momentum ve trend değişimini öğreniyor.

| Feature | Importance | Anlam |
|---------|-----------|-------|
| `delivery_brent_usd_change_7d` | 370 | Brent 7 günlük değişim |
| `delivery_ttf_eur_mwh_change_7d` | 345 | TTF 7 günlük değişim |
| `delivery_brent_usd_roll_mean_30d` | 322 | Brent 30 günlük ortalama |
| `delivery_henry_hub_usd` | 312 | Henry Hub seviye |
| `delivery_brent_try` | 306 | Brent TL cinsinden |

**Değerlendirme:** ✅ Düşük ama anlamlı. GRF grubuyla birlikte kullanıldığında güçlü: TTF değişimi GRF değişiminden önce gelir (öncü gösterge).

---

## 9. Sıcaklık (Temperature grubu)

### Model İçindeki Yeri
- **8 feature** | Toplam importance: 1,919 (%4.4)
- En önemli: `delivery_temp_delta_24` (444), `delivery_temp_lag_24` (438)

### Ekonomik Anlam
Türkiye'de sıcaklığın yük üzerindeki etkisi dramatik artış gösteriyor (2008-2025 arası yaz tepe talebi 12 kat büyümüş). Isıtma/soğutma derece günleri yük talebini → PTF'yi etkiler.

**Sürpriz:** Mutlak sıcaklık (262) değil, **değişim** (444) daha önemli. Model "dün 25°C idi bugün 30°C → klima yük artışı" dinamiğini öğrenmiş.

| Feature | Importance | Anlam |
|---------|-----------|-------|
| `delivery_temp_delta_24` | 444 | 24 saatlik sıcaklık değişimi |
| `delivery_temp_lag_24` | 438 | Dünün sıcaklığı |
| `delivery_apparent_temp_delta_24` | 406 | Hissettiren sıcaklık değişimi |
| `delivery_apparent_temperature` | 267 | Hissettiren sıcaklık |
| `delivery_temperature_2m` | 262 | Mutlak sıcaklık |

**Değerlendirme:** ✅ Makul katkı. Sıcaklık verisinin yük ile etkileşim terimi (`load_x_cooling`, `load_x_heating`) düşük importance → model sıcaklığı yük dolaylı olarak öğreniyor.

---

## 10. DAM Emir Defteri (DAM_orderbook grubu)

### Model İçindeki Yeri
- **13 feature** | Toplam importance: 7,871 (%18.1) — **1. SIRA!**
- En önemli: `delivery_dam_bid_sell_ratio` (1,237), `delivery_price_independent_balance` (1,181)

### Ekonomik Anlam
DAM emir defteri PTF'nin doğrudan deterministik belirleyicisi. Türkiye DAM kör ihalesi: alış emirleri ve satış teklifleri örtüştüğünde PTF oluşur. Piyasa mikroyapısı literatürü:
- Blok emirlerin ayrılamazlığı → fiyat süreksizlikleri
- Fiyatsız emirler (price-independent): talep ne olursa olsun alınacak
- Alış/satış dengesizliği → clearing fiyatını doğrudan iter

**Bu grubun 1. sıraya çıkması kritik bir bulgu:** Model teknik analiz yerine **piyasa mikroyapısını** öğreniyor.

### Hangi Saatlerde Katkı Sağlıyor?
- **Tüm saatler:** Emir defteri her saat için ayrı → sürekli katkı
- **Gece (00:00-06:00):** Night_block_pressure → blok alımları gece saatlerini domine eder
- **Akşam zirvesi (17:00-22:00):** Price-independent alımlar zirveye çıkar

| Feature | Importance | Anlam |
|---------|-----------|-------|
| `delivery_dam_bid_sell_ratio` | 1,237 | Alış/satış hacim oranı |
| `delivery_price_independent_balance` | 1,181 | Fiyatsız alış-satış farkı |
| `delivery_dam_bid_sell_balance` | 1,171 | Alış-satış hacim farkı |
| `delivery_price_independent_pressure` | 1,163 | Fiyatsız emir baskısı |
| `delivery_dam_sell_offer_volume` | 568 | Satış teklif hacmi |

**Değerlendirme:** ✅✅ Harika. DAM emir defteri verisini dahil etmek modeli önemli ölçüde güçlendirdi. Piyasa mikroyapısı PTF açıklama gücü en yüksek kaynak.

---

## 11. Takvim ve Zaman (Calendar_time grubu)

### Model İçindeki Yeri
- **31 feature** | Toplam importance: 1,906 (%4.4)
- En önemli: `delivery_hour` (441), `delivery_dow` (330)

### Ekonomik Anlam
PTF'nin gün içi profili (saat 00:00-23:00) ve hafta içi/hafta sonu farkı güçlü. Türkiye'de:
- Gece (00:00-06:00): Base-load → düşük fiyat
- Sabah rampa (06:00-09:00): Hızlı artış
- Öğle çöküşü (12:00-14:00): Solar üretim
- Akşam zirvesi (17:00-22:00): Maksimum PTF (~2,700 TL/MWh @ 18:00)
- Hafta sonu: Sanayi talebi düşük → fiyat düşük

| Feature | Importance | Anlam |
|---------|-----------|-------|
| `delivery_hour` | 441 | Saatlik profil (lineer) |
| `delivery_dow` | 330 | Haftanın günü |
| `delivery_hour_cos` | 321 | Saat (döngüsel kodlama) |
| `delivery_hour_sin` | 261 | Saat (döngüsel) |
| `delivery_dow_sin` | 93 | Gün (döngüsel) |

**Değerlendirme:** ⚠️ Düşük. LightGBM kalıcılık, KGÜP ve yük verisinden saatlik deseni zaten öğreniyor. Saf takvim özelliklerinin marjinal katkısı düşük — doğru davranış.

---

## 12. Gecikmeli Gerçekleşen Veriler (Lagged_realized grubu)

### Model İçindeki Yeri
- **20 feature** | Toplam importance: 304 (%0.7) — **Son sıra**

### Ekonomik Anlam
Gerçekleşen (realized) veriler: SMF (dengeleme mekanizması fiyatı), gerçek tüketim, gerçek üretim, IDM ağırlıklı ortalama fiyatı, YAL/YAT. Bunlar delivery anında mevcut değil, 24h-168h gecikmeli olarak kullanılıyor.

**Neden bu kadar düşük?**
- 24h lag: Bir önceki günün gerçekleşen verisi — persistence zaten daha doğrudan bu bilgiyi sunuyor
- 168h lag: Hafta öncesi gerçekleşen — çok eskimiş bilgi
- IDM fiyatı lag_24: DAM fiyatıyla yüksek korelasyon → üst kümeleşme

| Feature | Importance | Anlam |
|---------|-----------|-------|
| `anchor_gen_wind_lag_24` | 79 | Dünün rüzgar üretimi |
| `anchor_gen_gas_lag_24` | 70 | Dünün gaz üretimi |
| `anchor_gen_wind_lag_168` | 50 | Geçen hafta rüzgar |
| `anchor_smf_lag_24` | 23 | Dünün dengeleme fiyatı |
| `anchor_idm_price_lag_24` | (Composite) | IDM gecikmeli |

**Değerlendirme:** ❌ Düşük katkı. Bu grup **LOGO ablation'da test edilmeli** — belki zararlı bile olabilir (gürültü).

---

## 13. Kompozit Etkileşim Özellikleri (Composite_interactions grubu)

### Model İçindeki Yeri
- **6 feature** | Toplam importance: 1,570 (%3.6)
- En önemli: `delivery_gas_cost_pressure` (404), `delivery_ttf_x_gas_share` (369)

### Ekonomik Anlam
Türetilmiş endeksler: birden fazla değişkenin çarpımı veya farkı. Örneğin:
- `gas_cost_pressure = gas_share × grf_tl`: Ne kadar gazdan üretiliyorsa GRF o kadar marjinal maliyeti etkiler
- `ttf_x_gas_share`: TTF fiyatı × gaz payı = global gaz piyasasının etkisi
- `cheap_supply_pressure`: Yenilenebilir/hidro/rüzgar/güneş kombinasyonunun arz baskısı

| Feature | Importance | Anlam |
|---------|-----------|-------|
| `delivery_gas_cost_pressure` | 404 | Gaz maliyet baskısı endeksi |
| `delivery_ttf_x_gas_share` | 369 | TTF × gaz payı |
| `delivery_thermal_cost_pressure` | 361 | Termik maliyet baskısı |
| `delivery_gas_marginal_cost_pressure` | 301 | Gaz marjinal maliyet |
| `delivery_cheap_supply_pressure` | 89 | Ucuz arz baskısı |

**Değerlendirme:** ⚠️ Orta katkı. Etkileşim terimleri LightGBM'de zaten ağaç bölünmesiyle öğreniliyor, dolayısıyla bu özellikler LightGBM'e çok marjinal katkı sağlıyor. Linear model için daha değerli olurdu.

---

## Saat Bazlı PTF Analizi (Teorik)

Türkiye PTF'nin saatlik profili ve her grubun katkısı:

| Saat | Tipik PTF Seviyesi | En Kritik Grup | Açıklama |
|------|-------------------|----------------|----------|
| 00-05 | Düşük (gece) | DAM_orderbook | Blok alım emirleri bazload belirler |
| 06-09 | Hızlı artış | Ramp_dynamics | Sabah rampa — gaz devreye |
| 10-14 | Orta (solar) | KGUP + Temperature | Güneş üretim zirvesi fiyatı bastırır |
| 15-17 | Artış | Load_demand | Klima zirvesi başlıyor |
| 17-22 | **Maksimum** | GRF + Load + DAM | Gaz marginal + maks talep + maks emir |
| 22-23 | Hızlı düşüş | Baseline_persistence | Profil bilgisi dominant |

---

## Web Araştırması Bulguları (Özet)

Türk elektrik piyasası literatürü aşağıdaki ana bulguları destekliyor:

### Merit Order Etkisi
- Rüzgar ve akarsu hidro elektriğin 2012-2017 döneminde sistem marjinal fiyatını **istatistiksel olarak anlamlı düşürdüğü** kanıtlanmış (Karaçor et al., 2019; Danisman et al., 2021)
- 2024'te yenilenebilir enerji toptan elektrik fiyatlarını **%45 düşürdü**
- 2025'te yenilenebilir sayesinde hane faturalarında ortalama **%9.1** tasarruf

### Gaz-Elektrik Bağlantısı
- Gaz santralleri Türkiye'nin marjinal fiyat belirleyicisi
- TTF → GRF → PTF aktarım zinciri
- 2022 enerji krizi (TTF 300+ EUR/MWh) Türkiye PTF'yi rekor seviyelere taşıdı

### Piyasa Mikroyapısı
- DAM kör ihalesi; blok emirlerin ayrılamazlığı fiyat süreksizliği yaratır
- Günde ~15,000 emir, ~800 blok emir işleniyor
- Fiyatsız emirler (MFSP alışlar) talep ne olursa olsun piyasayı doğrudan etkiler

### Saat Referans Noktaları
- Türkiye'de en yüksek PTF: **18:00** (~2,700 TL/MWh)
- Kış zirvesi: 18:00-23:00 | Yaz zirvesi: 13:00-18:00
- En düşük: Gece 02:00-05:00

---

## Kaynakça

1. [Variable Renewable Energy in Turkey DAM: Quantile Regression](https://www.sciencedirect.com/science/article/abs/pii/S0301421520303906)
2. [Merit-Order of Dispatchable and VRE in Turkey's DAM](https://www.sciencedirect.com/science/article/abs/pii/S0957178724000511)
3. [Wind & Hydro Merit Order Effect in Turkish Spot Prices](https://www.sciencedirect.com/science/article/abs/pii/S0301421519304483)
4. [Ember: Türkiye Electricity Review 2025](https://ember-energy.org/app/uploads/2025/03/Turkiye-Electricity-Review-2025_11032025.pdf)
5. [EPİAŞ/EMRA: Market Architecture Overview](https://www.epias.com.tr/en/)
6. [IEA: Natural Gas Price Volatility](https://www.iea.org/commentaries/what-drives-natural-gas-price-volatility-in-europe-and-beyond)

---

*Not: Leave-One-Group-Out ablation sonuçları ayrı raporda: `logo_ablation_results.md`*

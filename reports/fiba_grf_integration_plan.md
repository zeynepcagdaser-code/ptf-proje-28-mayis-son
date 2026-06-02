# FİBA/FİBS + GRF (DRP) Entegrasyon Planı (yalnızca plan)

Bu rapor **kod değiştirmeden**, **veri çekmeden** ve **model eğitmeden** hazırlanmıştır. Amaç; eksik piyasa verisi entegrasyonunu, repo’daki mevcut ingestion/clean/master/features mimarisine ilk aşamada bağlayacak net adımları tanımlamaktır.

## Mevcut altyapı özeti (repo)

### EPİAŞ fetch/auth kalıbı
- Repo’da EPİAŞ kaynakları için ortak desen: **CAS TGT** ile login (`https://giris.epias.com.tr/cas/v1/tickets`) + ilgili endpoint’e **POST**.
- Örnekler:
  - Final PTF/MCP: `update_dataset.py` → `.../electricity-service/v1/markets/dam/data/mcp` → `data/ptf_dataset.csv`
  - Yük tahmini: `fetch_load_forecast.py` → `.../consumption/data/load-estimation-plan` → `data/load_forecast.csv`
  - SMF: `fetch_smf.py` → `.../markets/bpm/data/system-marginal-price` → `data/smf.csv`

### Cleaning katmanı
- `run_cleaning.py` → `cleaning/pipeline.py` → `cleaning/config.py: HOURLY_CSV_SOURCES`
- Raw CSV → `data/clean/*_hourly.parquet` (hourly ts standardı: `ts_hour`, `Europe/Istanbul`).

### Master merge katmanı
- `build_master.py` → `master/build_master.py` → `master/schema.py: JOIN_DATASETS`
- `data/clean/*.parquet` dosyaları PTF spine (`data/clean/ptf_hourly.parquet`) üzerine **left join** ile eklenir.
- Son master: `data/master/master_hourly_v1.parquet` (son raporda 77 kolon; FİBA/FİBS/GRF yok).

### Feature engineering katmanı
- `build_features.py` master’ı okuyup `features/engineering.py` ile leakage-safe feature üretir.
- Ana model ve low-price classifier feature sözleşmeleri `features/config.py` listeleri ile yönetilir.

---

## Entegrasyon hedefleri (ilk aşama)

Bu aşamada sadece 3 veri:

1) **GÖP Fiyattan Bağımsız Alış Teklifi (FİBA)**  
URL: https://seffaflik.epias.com.tr/electricity/electricity-markets/day-ahead-market-dam/dam-price-independent-bid-order  
Hedef master kolonu: `dam_price_independent_buy_mwh`

2) **GÖP Fiyattan Bağımsız Satış Teklifi (FİBS)**  
URL: https://seffaflik.epias.com.tr/electricity/electricity-markets/day-ahead-market-dam/dam-price-independent-sales-order  
Hedef master kolonu: `dam_price_independent_sell_mwh`

3) **Doğal Gaz GRF / Günlük Referans Fiyatı (DRP)**  
URL: https://seffaflik.epias.com.tr/natural-gas/natural-gas-markets/spot-gas-markets-sgp/price/daily-reference-price-drp  
Hedef master kolonları:
- `grf_tl_1000sm3`
- `grf_usd_1000sm3`
- `grf_eur_mwh`
- `grf_usd_mmbtu`

---

## Repo’da mevcut durum (tespit)

### 1) Endpoint / fetch desteği
- **FİBA/FİBS:** Repo’da “price-independent order” için fetch script’i yok. `fetch_*.py` ve `update_dataset.py` içinde ilgili endpoint bulunmuyor.
- **GRF/DRP:** Repo’da “natural gas / daily reference price” için fetch script’i yok. `fetch_*.py` içinde natural-gas endpoint yok.

### 2) Master’da kolon var mı?
`data/master/master_hourly_v1.parquet` şemasında şu kolonlar **yok**:
- `dam_price_independent_buy_mwh`
- `dam_price_independent_sell_mwh`
- `grf_*` (4 kolonun tamamı)

---

## Önerilen ingestion + cleaning + master merge tasarımı

### A. FİBA / FİBS (DAM price-independent buy/sell)

#### Beklenen zaman granülaritesi
- EPİAŞ DAM emir/teklif tablosu genelde **teslimat günü+saati** bazında (hourly) raporlanır.
- Master’ın join anahtarı `ts_hour` olduğu için hedef çıktı hourly olmalıdır.

#### Önerilen raw çıktı dosyaları
- `data/dam_price_independent_buy.csv`
- `data/dam_price_independent_sell.csv`

#### Önerilen temiz (clean) parquet
Cleaning pipeline ile uyumlu olacak şekilde:
- `data/clean/dam_price_independent_buy_hourly.parquet` → kolonlar: `ts_hour`, `price_independent_buy_mwh`
- `data/clean/dam_price_independent_sell_hourly.parquet` → kolonlar: `ts_hour`, `price_independent_sell_mwh`

#### Master merge (master/schema.py)
`JOIN_DATASETS` listesine iki `DatasetSpec` eklenmesi planlanır:
- `name="dam_price_independent_buy"`, `filename="dam_price_independent_buy_hourly.parquet"`, `prefix="dam_"`, `default_availability="planned"`
- `name="dam_price_independent_sell"`, `filename="dam_price_independent_sell_hourly.parquet"`, `prefix="dam_"`, `default_availability="planned"`

Bu sayede master kolonları:
- `dam_price_independent_buy_mwh`
- `dam_price_independent_sell_mwh`

#### Endpoint keşfi (manuel gereken nokta)
Verilen URL’ler web UI sayfalarıdır; repo’da bu sayfaların arkasındaki **API path** henüz tanımlı değil.
Plan:
- Tarayıcı “Network” incelemesiyle UI’nın çağırdığı endpoint bulunur.
- Büyük olasılıkla `https://seffaflik.epias.com.tr/electricity-service/v1/markets/dam/data/...` altında bir POST endpoint olacaktır (repo’daki diğer DAM endpoint’leri ile aynı aile).
- Bulunan endpoint `fetch_*` kalıbına uygun şekilde (CAS TGT + POST + json payload) script’e alınır.

#### Leakage riski
- Bu veriler DAM teklif bilgisi olduğu için, teslimat saatine göre **D-1**’de bilinir. Anchor `ts_hour=t` için “same_hour_ok” kabul edilebilir.
- Yine de sıkı DAM cutoff için: veri zaman damgası/versiyonlaması varsa yalnızca **kapalı periyot** snapshot’ı kullanılmalıdır. İlk ingestion aşamasında “planned” etiketiyle master’a eklenmesi, feature tarafında ekstra “lag” zorunluluğu koymadan da leakage yaratmaz; çünkü zaten hedef `target_{h}` gelecektir ve teklif büyüklüğü anchor’dan önce bilinir.

---

### B. GRF / DRP (doğal gaz günlük referans fiyatı)

#### Beklenen zaman granülaritesi
- GRF “daily” (günlük) bir seri.
- Master saatlik (`ts_hour`) olduğundan iki seçenek vardır:
1) **Daily → hourly expand:** Her gün için 24 saate aynı GRF değeri yazılır (join kolay).
2) Master’a daily join eklemek yerine feature katmanında daily tabloyu ayrı okuyup map etmek.

Repo mimarisine en uyumlu seçenek: **daily → hourly expand** ile `data/clean/grf_hourly.parquet` üretmek.

#### Önerilen raw çıktı dosyası
- `data/grf_daily_reference_price.csv` (veya `data/grf_drp.csv`)

#### Önerilen clean parquet
- `data/clean/grf_hourly.parquet`
  - `ts_hour` (günlük değer 24 saat boyunca yayılır)
  - `tl_1000sm3`, `usd_1000sm3`, `eur_mwh`, `usd_mmbtu`

#### Master merge (master/schema.py)
`JOIN_DATASETS` listesine bir `DatasetSpec` eklenmesi planlanır:
- `name="grf"`, `filename="grf_hourly.parquet"`, `prefix="grf_"`, `default_availability="realized"` (veri yayımlanma anı geçmişe dönük olabilir)

Master kolonları:
- `grf_tl_1000sm3`
- `grf_usd_1000sm3`
- `grf_eur_mwh`
- `grf_usd_mmbtu`

#### Endpoint keşfi (manuel gereken nokta)
Verilen URL doğal gaz transparanlık UI’ıdır. Repo’da natural gas service endpoint’leri yok.
Plan:
- Network incelemesi ile UI’nın çağırdığı servis bulunur (muhtemelen `.../natural-gas-service/v1/...` ailesi veya export endpoint’i).
- Eğer CAS auth gerekiyorsa (çoğu EPİAŞ service gibi), mevcut login kalıbı (`LOGIN_URL`) yeniden kullanılır.

#### Leakage riski
- GRF günlük yayımlanma saatine bağlı olarak “same-hour” feature olarak almak riskli olabilir.
- Bu nedenle model feature’ları mutlaka **lagged** ve **past-only** olmalıdır (aşağıda).

---

## Üretilecek feature setleri (plan)

### 1) FİBA/FİBS türevleri (hourly)
Master’da iki raw kolon geldikten sonra `features/engineering.py` içinde üretilecek (tamamı leakage-safe, same-hour volume):
- `fiba_fibs_ratio = dam_price_independent_buy_mwh / dam_price_independent_sell_mwh`
- `fiba_fibs_balance = dam_price_independent_buy_mwh - dam_price_independent_sell_mwh`
- `fiba_fibs_total = dam_price_independent_buy_mwh + dam_price_independent_sell_mwh`
- `fiba_fibs_pressure = fiba_fibs_balance / fiba_fibs_total` (0 bölme kontrolü ile)

Önerilen listelere ekleme:
- `MAIN_REGRESSION_FEATURES`: `fiba_fibs_pressure`, `fiba_fibs_balance`, `fiba_fibs_total` (ratio opsiyonel)
- `LOW_PRICE_CLASSIFIER_FEATURES`: `fiba_fibs_pressure`, `fiba_fibs_balance` (düşük fiyat rejimini yakalamak için)

### 2) GRF türevleri (daily kaynak → hourly)
Master’da `grf_tl_1000sm3` geldikten sonra `features/engineering.py` içinde:
Lag/rolling (past-only):
- `grf_tl_lag_1d` (t-24)
- `grf_tl_change_7d` (t-24 - (t-24-7d) gibi tanım; günlük seride 7 gün fark)
- `grf_tl_rolling_mean_7d` (past-only, en az 7 gün)

Maliyet baskısı (anchor’daki shares * lagged GRF):
- `gas_cost_pressure = gas_share * grf_tl_lag_1d`
- `thermal_cost_pressure = thermal_price_setting_share * grf_tl_lag_1d`
- `gas_marginal_pressure = gas_share * thermal_price_setting_share * grf_tl_lag_1d`

Önerilen listelere ekleme:
- `MAIN_REGRESSION_FEATURES`: `gas_cost_pressure`, `thermal_cost_pressure`, `gas_marginal_pressure`, `grf_tl_lag_1d`, `grf_tl_change_7d`, `grf_tl_rolling_mean_7d`
- `LOW_PRICE_CLASSIFIER_FEATURES`: `gas_cost_pressure` (opsiyonel), `grf_tl_lag_1d`

Not: GRF’nin aynı gün yayımlanma saatine bağlı risk nedeniyle ana modele **ham** `grf_tl_1000sm3` (same-hour) konulmaz; yalnızca lag/rolling kullanılır.

---

## Eksik / manuel gereken noktalar (ilk entegrasyon blokajları)

1) **UI URL → API endpoint** dönüşümü:  
FİBA/FİBS ve GRF için repo’da endpoint yok. Network incelemesi şart.

2) **Zaman alanları ve birimler**:
- DAM bağımsız teklif tablolarında saat alanı `hour`/`time` olabilir; `clean_hourly_csv` ile uyumlu CSV şeması tasarlanmalı.
- GRF birimleri: TL/1000Sm3 vb. Dönüşüm yapılmayacaksa ham kolon isimleri netleştirilmeli.

3) **Daily → hourly expand**:
Cleaning pipeline “hourly CSV” bekliyor. GRF için ya:
- ayrı “daily cleaner” eklenmeli, ya da fetch script’i doğrudan hourly expand edip CSV yazmalı.

4) **Availability etiketleri**:
- FİBA/FİBS: `planned` (D-1’de bilinir).
- GRF: yayın zamanı belirsiz; master’da `realized` tutulup feature tarafında lag zorunlu kılınmalı.

---

## Önerilen ilk implementasyon sırası (uygulama değil, plan)

1) Endpoint keşfi (3 UI sayfası için) ve JSON payload alanlarının (startDate/endDate vb.) çıkarılması.  
2) 3 yeni fetch script’i:
   - `fetch_dam_price_independent_buy.py`
   - `fetch_dam_price_independent_sell.py`
   - `fetch_grf_drp.py`
3) Cleaning entegrasyonu:
   - FİBA/FİBS: `cleaning/config.py: HOURLY_CSV_SOURCES` içine 2 kaynak ekleme
   - GRF: daily→hourly expand stratejisine göre cleaning pipeline’a ekleme
4) Master merge (`master/schema.py: JOIN_DATASETS`) + prefix standardizasyonu (`dam_`, `grf_`)
5) Feature engineering ekleri (`features/engineering.py`) + listelere ekleme (`features/config.py`)
6) Audit: `audit_feature_inventory.py` ile master/feature coverage doğrulama (eğitim yok).


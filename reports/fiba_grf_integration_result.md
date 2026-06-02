## FİBA/FİBS + GRF Entegrasyon Sonucu (Aşama-1)

**Model eğitimi yapılmadı.** (Bu aşamada yalnızca veri çekimi + master merge + feature engineering + sequence dataset build komutları çalıştırıldı.)

### Bulunan endpoint’ler
- **GÖP Fiyattan Bağımsız Alış Teklifi (FİBA)**: `POST /electricity-service/v1/markets/dam/data/price-independent-bid`
- **GÖP Fiyattan Bağımsız Satış Teklifi (FİBS)**: `POST /electricity-service/v1/markets/dam/data/price-independent-offer`
- **GRF / Günlük Referans Fiyatı (DRP)**: `POST /natural-gas-service/v1/markets/sgp/data/daily-reference-price`

Kaynak: EPİAŞ teknik dokümantasyon (electricity-service TR 5.129/5.130, natural-gas-service TR 5.10).

### Çekilen veri özeti
- **FİBA (buy)**:
  - **Ham**: `data/external/epias_market/fiba_fibs/dam_price_independent_buy_raw.csv`
  - **Processed**: `data/processed/dam_price_independent_buy.parquet`
  - **Satır**: **56,208** (saatlik)
- **FİBS (sell)**:
  - **Ham**: `data/external/epias_market/fiba_fibs/dam_price_independent_sell_raw.csv`
  - **Processed**: `data/processed/dam_price_independent_sell.parquet`
  - **Satır**: **56,208** (saatlik)
- **GRF (daily)**:
  - **Ham**: `data/external/epias_market/grf/grf_daily_reference_price_raw.csv`
  - **Processed**: `data/processed/grf_daily_reference_price.parquet`
  - **Satır**: **2,342** (günlük, `ts_day`)

**Tarih aralığı (master spine)**: `2020-01-01 00:00+03:00` → `2026-05-30 23:00+03:00`

**API limit**: Gözlenmedi. İstekler 365-günlük chunk’lar halinde atıldı (tüm chunk’larda HTTP 200).

### Master merge sonucu
Komut: `python3 master/build_master.py`

Master’a eklenen kolonlar:
- **FİBA/FİBS**: `dam_price_independent_buy_mwh`, `dam_price_independent_sell_mwh`
- **GRF**: `grf_tl_1000sm3`, `grf_usd_1000sm3`, `grf_eur_mwh`, `grf_usd_mmbtu`

GRF join politikası:
- **Günlük GRF**, `ts_day := floor(ts_hour,'D')` anahtarıyla join edildi ve **hourly spine boyunca forward-fill** uygulandı.

### Feature engineering sonucu
Komut: `python3 build_features.py`

FİBA/FİBS engineered:
- `fiba_fibs_ratio = buy/sell`
- `fiba_fibs_balance = buy-sell`
- `fiba_fibs_total = buy+sell`
- `fiba_fibs_pressure = balance/total`

GRF engineered (leakage-safe tasarım):
- `grf_tl_lag_1d` (t-24h)
- `grf_tl_change_7d` (lag_1d - lag_8d)
- `grf_tl_rolling_mean_7d` (lag_1d üzerinden rolling)
- `gas_cost_pressure = gas_share * grf_tl_1000sm3`
- `thermal_cost_pressure = thermal_price_setting_share * grf_tl_1000sm3`
- `gas_marginal_pressure = gas_share * thermal_price_setting_share * grf_tl_1000sm3`

**Güvenli bölme (denominator 0)**:
- `fiba_fibs_ratio`: `sell==0` ise **NaN**
- `fiba_fibs_pressure`: `total==0` ise **NaN**

### Feature listeleri ve sayılar
Komut: `python3 audit_feature_inventory.py`

- **Feature parquet toplam feature sayısı**: **130**
- **MAIN_REGRESSION_FEATURES**: **52/52 mevcut**, eksik **0**
- **LOW_PRICE_CLASSIFIER_FEATURES**: **45/45 mevcut**, eksik **0**
- **RISK_DASHBOARD_FEATURES**: **21/21 mevcut**, eksik **0**

### Leakage riski
- Audit çıktısı: **Leakage riski high olan feature: boş liste**
- Not: `grf_tl_1000sm3` aynı-gün kullanılacağı için yayın zamanı/kesinleşme açısından **manuel gözden geçirme** gerektirir; bu yüzden GRF için lag/rolling/change feature’lar özellikle eklendi.

### Sequence dataset build sonucu
- **main_regression**:
  - Çıktı: `data/model/`
  - Feature sayısı: **52**
  - Shape:
    - train `X=(43489,168,52)`, `y=(43489,24)`
    - validation `X=(8570,168,52)`, `y=(8570,24)`
    - test `X=(3387,168,52)`, `y=(3387,24)`
  - Raporlar: `reports/sequence_report_main_regression_latest.md/json`
- **low_price_classifier**:
  - Çıktı: `data/model_low_price/`
  - Feature sayısı: **45**
  - Shape:
    - train `X=(43513,168,45)`, `y=(43513,24)`
    - validation `X=(8570,168,45)`, `y=(8570,24)`
    - test `X=(3387,168,45)`, `y=(3387,24)`
  - Raporlar: `reports/sequence_report_low_price_classifier_latest.md/json`

### Çalıştırılan komutlar (sadece veri/feature)
- `python3 fetch_dam_price_independent_buy.py`
- `python3 fetch_dam_price_independent_sell.py`
- `python3 fetch_grf_daily_reference_price.py`
- `python3 master/build_master.py`
- `python3 build_features.py`
- `python3 audit_feature_inventory.py`
- `python3 run_sequence.py`
- `python3 run_sequence.py --feature-profile low_price_classifier`

### Çalıştırılmayan komutlar
- `train_lstm.py`, `train_d2_ptf_forecaster.py`, `train_spike_classifier.py`, `train_low_price_classifier.py`, `run_main_tabular_hybrid_baseline.py`
- **Model eğiten hiçbir script çalıştırılmadı.**

---

Detaylı makine-okunur çıktı: `reports/fiba_grf_integration_result.json`


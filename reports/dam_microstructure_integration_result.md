## DAM microstructure entegrasyon sonucu (Aşama-2)

**Model eğitimi yapılmadı.**

### Endpoint’ler
- **dam_bid_volume**: `POST /electricity-service/v1/markets/dam/data/submitted-bid-order-volume`
- **dam_sell_offer_volume**: `POST /electricity-service/v1/markets/dam/data/submitted-sales-order-volume`
- **dam_matched_volume**: `POST /electricity-service/v1/markets/dam/data/clearing-quantity`
- **dam_block_buy_volume**: `POST /electricity-service/v1/markets/dam/data/amount-of-block-buying`

### Tarih aralığı
- master: 2020-01-01 00:00:00+03:00 → 2026-05-30 23:00:00+03:00

### Master’a eklenen kolonlar
- `dam_bid_volume_mwh`
- `dam_sell_offer_volume_mwh`
- `dam_matched_buy_mwh`
- `dam_matched_sell_mwh`
- `dam_block_matched_buy_mwh`
- `dam_block_unmatched_buy_mwh`

### Feature/sequence sayıları
- Feature parquet total: **169**
- MAIN_REGRESSION_FEATURES: **73**
- LOW_PRICE_CLASSIFIER_FEATURES: **57**
- RISK_DASHBOARD_FEATURES: **32**
- Sequence main feature count: **73**
- Sequence low-price feature count: **57**

### Leakage risk
- high leakage risk features: `[]`

### Mod notu
- current DAM mikro-yapı feature’ları: post_dam_publication_mode için kullanılabilir.
- strict_forecast_mode için lagged alternatifler üretildi ve listelere eklendi.

Detay sanity raporu: `reports/dam_microstructure_feature_sanity.md/json`

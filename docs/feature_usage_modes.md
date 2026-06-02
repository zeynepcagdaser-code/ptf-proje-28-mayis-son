## Feature kullanım modları: FİBA/FİBS + GRF zamanlama notu

Bu projede aynı feature seti iki farklı operasyonel kullanım modunda çalıştırılabilir.

### A) post_dam_publication_mode
- **Amaç**: GÖP (DAM) verileri yayınlandıktan sonra analiz/tahmin.
- **Kural**: Current (aynı saat) FİBA/FİBS kullanılabilir.
- **Kapsam (örnek kolonlar)**:
  - `dam_price_independent_buy_mwh`
  - `dam_price_independent_sell_mwh`
  - `fiba_fibs_ratio`
  - `fiba_fibs_balance`
  - `fiba_fibs_pressure`

### B) strict_forecast_mode
- **Amaç**: GÖP sonucu/ilgili veri yayınlanmadan önce “strict” tahmin.
- **Kural**: Current FİBA/FİBS kullanılmaz; **lagged** alternatifleri tercih edilir.
- **Kapsam (örnek kolonlar)**:
  - `dam_price_independent_buy_lag_24`
  - `dam_price_independent_sell_lag_24`
  - `fiba_fibs_ratio_lag_24`
  - `fiba_fibs_balance_lag_24`
  - `fiba_fibs_pressure_lag_24`
  - `fiba_fibs_ratio_lag_168`
  - `fiba_fibs_pressure_lag_168`

### GRF (Günlük Referans Fiyatı) zamanlama notu
- GRF yayın/ilan zamanının operasyonel saat-kesiti belirsiz olabildiği için, **ana modelde** current `grf_tl_1000sm3` yerine lag/türev feature’lar **önceliklidir**:
  - `grf_tl_lag_1d`, `grf_tl_change_7d`, `grf_tl_rolling_mean_7d`
  - `gas_cost_pressure_lag_1d`, `thermal_cost_pressure_lag_1d`, `gas_marginal_pressure_lag_1d`
- Dashboard/analiz tarafında current GRF ve current maliyet baskıları izleme amacıyla tutulabilir.


## Thesis-ready model summary

### Used data
- `data/model`: main_regression sequence dataset (168h input window, 24h horizons)
- Feature set: **73** features (FİBA/FİBS + GRF + DAM microstructure entegre)
- Splits: train/validation/test with `anchor_*.csv` timestamps

### Model
- Backend: **lightgbm**
- Multi-horizon: 24 ayrı model (h1..h24)
- Tabularization: last timestep + 24h mean/std/min/max + 168h mean/std/min/max + trend(last-mean24)

### Why LightGBM/XGBoost
- Ağaç tabanlı boosting modelleri tabular feature set’lerde güçlü ve hızlı baseline sağlar.
- Deep learning’e göre daha hızlı iterasyon, daha az operasyonel karmaşıklık.

### Validation tuning
- Small grid search (learning_rate, num_leaves/max_depth, n_estimators) on validation.
- Primary metric: **MAPE(actual>100)**, secondary metric: **MAE**.
- Selected params: `{'learning_rate': 0.1, 'num_leaves': 31, 'n_estimators': 500}`

### Persistence baseline comparison (test, flattened h1-h24)
- Persistence: dünkü aynı saat PTF (`anchor_ts + h - 24`), eksikse `ptf_lag_24` fallback.
- Persistence MAE: **538.93**
- Model MAE: **757.00**
- Persistence MAPE(actual>100): **68.41%**
- Model MAPE(actual>100): **169.65%**
- Model RMSE: **950.35**
- Model WAPE: **48.07%**

### 24-hour forecasting approach
- Her horizon için ayrı model eğitilir; bu sayede horizon’a özgü hata profili yakalanır.


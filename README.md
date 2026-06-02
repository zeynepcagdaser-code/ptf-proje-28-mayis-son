# ptf-proje-28-mayis-son

## Veri temizleme

Ham CSV dosyalarını saatlik parquet formatına dönüştürmek için:

```bash
pip install -r requirements.txt
python run_cleaning.py
```

- Girdi: `data/*.csv`
- Çıktı: `data/clean/*_hourly.parquet`
- Rapor: `reports/cleaning_report_latest.json` ve `.md`

## Master dataset

Temiz parquet dosyalarını PTF spine üzerinde birleştirmek için:

```bash
python build_master.py
```

- Girdi: `data/clean/*_hourly.parquet`
- Çıktı: `data/master/master_hourly_v1.parquet`
- Rapor: `reports/master_report_latest.json` ve `.md`

## LSTM feature dataset (tabular)

```bash
python build_features.py
```

- Girdi: `data/master/master_hourly_v1.parquet`
- Çıktı: `data/features/lstm_next24_v1.parquet`
- Rapor: `reports/features_report_latest.json` ve `.md`
- Format: anchor `ts_hour=t`, features at `t`, targets `target_1h..target_24h` = PTF at `t+1..t+24`

## Sequence tensors (LSTM-ready)

```bash
python run_sequence.py
```

- Girdi: `data/features/lstm_next24_v1.parquet`
- Çıktı: `data/model/X_{train,val,test}.npy`, `y_*.npy`, scaler `.pkl`, metadata JSON
- Rapor: `reports/sequence_report_latest.json` ve `.md`
- Anchor metadata: `data/model/anchor_{train,val,test}.csv`

### ML artifacts (manuel CI)

Ham CSV güncellemesinden sonra parquet/feature/sequence üretmek için GitHub Actions:

- Workflow: `.github/workflows/build_ml_artifacts.yml`
- Tetik: yalnızca `workflow_dispatch` (saatlik cron’u yavaşlatmaz)
- `.npy` dosyaları commit edilmez (yerelde `python run_sequence.py` ile üretilir)

## PTF üretim pipeline

Aşağıdaki script ve workflow, iki aşamalı PTF tahmin zincirini üretim için hazırlar:

- `python scripts/two_stage_ptf_pipeline.py --run-all --quantile-alpha 0.5`
- `python scripts/two_stage_multihorizon_pipeline.py --run-all --quantile-alpha 0.5`
- `python scripts/hourly_retrain_ptf_regressor.py`
- `python scripts/optuna_tune_two_stage.py --horizon 1 --objective regression --trials 50`

Workflow:

- `.github/workflows/retrain_ptf_pipeline.yml`
- Tetik: `schedule` (saatlik), `push` `main`, `workflow_dispatch`
- Güncelleme: `data/predictions/` içeriği değişirse otomatik commit/push eder

## LSTM baseline eğitimi (PyTorch)

```bash
pip install -r requirements.txt
python run_sequence.py   # .npy yoksa önce üret
python train_lstm.py
```

Çıktılar: `models/lstm_baseline.pt`, `reports/lstm_baseline_metrics.*`, `reports/figures/`, `data/predictions/lstm_test_predictions.csv`

## Güvenlik / risk azaltma yardımcıları

- Atomik Parquet yazımı için: `src/utils/safe_io.py` içindeki `atomic_parquet_write(df, path)` fonksiyonunu kullanın; bu, önce geçici dosyaya yazar sonra hedef dosyayla atomik olarak değiştirir.
- Zaman damgası normalizasyonu: `src/utils/tz_utils.py` içindeki `normalize_to_ts_hour(df, col='date')` fonksiyonu `ts_hour` sütununu `Europe/Istanbul` zaman dilimine göre oluşturur (naive vs tz-aware uyuşmazlıklarını azaltmak için).
- Veri klasörünüzü taramak için basit bir araç: `scripts/check_timezones.py` — örnekleme yapıp hangi sütunların tz-aware olduğunu raporlar.

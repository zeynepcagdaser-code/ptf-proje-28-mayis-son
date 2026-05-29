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
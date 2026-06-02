# Low-Price Tabular Dataset Report

- **Generated (UTC):** 2026-06-01T20:33:58.252553+00:00
- **Sequence dir:** `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/model_low_price`
- **Output dir:** `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/model_low_price_tabular`
- **Base features:** 43 → **tabular features:** 430 (expected 430)
- **Low-price threshold:** 50.0 TL/MWh

## Splits

### train

- Rows: 43513
- Sequence X shape: `[43513, 168, 43]`
- Tabular feature count: 430
- y_low shape: `[43513, 24]`
- Low class rate (any horizon): 0.0428
- Low class rate (mean over horizons): 0.0109
- Zero class rate (any horizon): 0.0075
- Zero class rate (mean over horizons): 0.0012
- X: `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/model_low_price_tabular/X_train.parquet`
- y_low: `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/model_low_price_tabular/y_low_train.parquet`

### validation

- Rows: 8570
- Sequence X shape: `[8570, 168, 43]`
- Tabular feature count: 430
- y_low shape: `[8570, 24]`
- Low class rate (any horizon): 0.0560
- Low class rate (mean over horizons): 0.0104
- Zero class rate (any horizon): 0.0277
- Zero class rate (mean over horizons): 0.0049
- X: `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/model_low_price_tabular/X_val.parquet`
- y_low: `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/model_low_price_tabular/y_low_val.parquet`

### test

- Rows: 3387
- Sequence X shape: `[3387, 168, 43]`
- Tabular feature count: 430
- y_low shape: `[3387, 24]`
- Low class rate (any horizon): 0.3676
- Low class rate (mean over horizons): 0.1067
- Zero class rate (any horizon): 0.3596
- Zero class rate (mean over horizons): 0.0971
- X: `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/model_low_price_tabular/X_test.parquet`
- y_low: `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/model_low_price_tabular/y_low_test.parquet`
- y_zero (test only): `/Users/salihcagdaser/Desktop/ptf-proje-28-may-s/data/model_low_price_tabular/y_zero_test.parquet`


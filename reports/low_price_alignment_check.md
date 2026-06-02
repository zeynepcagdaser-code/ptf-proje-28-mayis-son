# Low-Price Classifier Alignment Check

- **Generated (UTC):** 2026-06-01T20:43:14.332601+00:00

## Mapping conventions

- Row grain: one row per (split, sample_index, horizon)
- `low_prob` at `horizon=h` ↔ `is_low_{h}h` / `is_low_actual`
- `sample_index`: per-split 0..N-1, not global

## Contradiction analysis

Overall diagnostics pool train+validation+test. Train rows (~78% of stacked rows) show extreme overfit (median p_low|low≈1, recall@0.02=1.0). Test-only median p_low|actual_low is near zero, so recall stays low.

| Split | median p_low \| low | recall@0.02 | ROC-AUC |
|-------|------------------:|------------:|--------:|
| train | 0.9999999956887036 | 1.0000 | 1.0 |
| validation | 2.60187138703813e-06 | 0.1091 | 0.9349236031896812 |
| test | 5.293130356369978e-06 | 0.1447 | 0.8527077264636997 |

**Alignment verdict:** PASS
**Horizon offset:** False

## Test parquet vs CSV labels

- Total `is_low` mismatches: 0

## Test horizon snapshot (h=1, h=2, h=11)

| h | low_n | med p\|low | med p\|not low | recall@0.02 | recall@0.05 | ROC-AUC |
|--:|------:|-------------:|---------------:|------------:|------------:|--------:|
| 1 | 355 | 0.1028106116820893 | 9.699580639333806e-10 | 0.5380 | 0.5155 | 0.8935235423092645 |
| 2 | 355 | 0.313944362178286 | 4.358910362643957e-10 | 0.6366 | 0.6056 | 0.9565099037496748 |
| 11 | 361 | 1.450970220360906e-09 | 3.554220845416738e-10 | 0.0000 | 0.0000 | 0.6083252623157016 |

## Planned (not trained)

### Zero-specific classifier

{
  "status": "planned_not_trained",
  "target": "is_zero_{h}h from inverse-scaled y (PTF == 0)",
  "model": "separate horizon-wise classifier (LightGBM/HGB)",
  "threshold_policy": "recall-first; target zero recall >= 0.70 on validation",
  "features": "LOW_PRICE_CLASSIFIER_FEATURES + zero regime history",
  "note": "Decouple from <=50 TL low-price head; optimize zero capture"
}

### Any-horizon classifier

{
  "status": "planned_not_trained",
  "target_any_low": "any(is_low_1h..is_low_24h) per anchor row",
  "target_any_zero": "any(is_zero_1h..is_zero_24h) per anchor row",
  "grain": "one row per anchor hour (not 24x stacked)",
  "use_case": "hourly regime alert independent of horizon offset",
  "threshold_policy": "recall-first on validation"
}

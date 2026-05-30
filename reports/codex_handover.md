# Codex handover — PTF forecasting project

Generated: 2026-05-30T12:41:14.267798+00:00

This package exports reasoning, experiments, leakage lessons, and file index for continuation in Codex **without context loss**.

---

## 1. Executive summary

**Problem:** Forecast Turkish electricity DAM PTF (TL/MWh) for next 24 hours from hourly anchors.

**Current best (frozen checkpoint):** Validation-weighted ensemble (advanced tree + microstructure), **mean h1–h4 MAE ≈ 443.87** on LSTM-aligned test (3281 anchors). Beats advanced tree alone (~453.09) and persistence (~544).

**Next direction:** Interim MCP (K.PTF) baseline → predict final MCP correction. Data pipeline started; historical fetch incomplete.

**Do not re-run without reading:** leakage notes on `regressor_online`, test oracle ensembles, and interim fetch API limits.

---

## 2. Project evolution

See `reports/experiment_timeline.md` for full chronology.

**Arc:** Persistence → LSTM fail → residual LSTM → trees win → microstructure → validation ensemble → interim MCP pivot.

---

## 3. Major discoveries

### Persistence is extremely strong
- 24h same-clock lag MAE ~545 across horizons
- Any model must be judged against this, not only vs older LSTM

### Direct LSTM failed
- Normalized sequence on 75 features → test MAE ~1085
- Lesson: need residual/baseline structure or tabular trees

### Advanced tree is best single method
- Hour-specific LightGBM, persistence+residual targets, recency weights, optional classifiers
- h1–h4 ~453 MAE (aligned test)

### Validation leakage was real
- Using `regressor_online` (train+val fit) on validation for ensemble weighting gave ~12 MAE — meaningless
- **Fix:** weight search uses train-only `regressor`; test uses saved prediction CSVs from online models

### Ensemble gains are real but modest
- ~9 TL/MWh vs advanced tree with honest validation weights
- Fixed 0.7/0.3 on test is slightly better (~440) but not the official checkpoint

### Interim MCP is not in current features
- `update_dataset.py` pulls **final** MCP only (`/mcp` → `price`)
- K.PTF requires **`/interim-mcp`** → `marketTradePrice`

---

## 4. Current best checkpoint (h1–h4)

| Model | Mean MAE h1–h4 |
|-------|----------------|
| Persistence | ~544.14 |
| Residual LSTM | ~538.59 |
| Advanced tree | **453.09** |
| Microstructure | 461.76 |
| **Val-weighted ensemble** | **443.87** |

**Validation weights (advanced / micro):** h1=0.6/0.4, h2=0.7/0.3, h3=0.9/0.1, h4=1.0/0.0

**Why trusted:**
- Weights from validation only (`evaluate_h1h4_validation_weighted_ensemble.py`)
- Audit pass: splits, persistence shift, prediction alignment (`audit_final_h1h4_pipeline.py`)
- Same test anchors as LSTM (`data/model/anchor_test.csv`)

**Artifacts:**
- Predictions: `data/predictions/h1h4_validation_weighted_ensemble_predictions.csv`
- Metrics: `reports/final_h1h4_summary.json`

---

## 5. Current limitations

1. **Still ~444 MAE** — far from persistence in absolute terms for ensemble; tree already ~453 vs ~544 persistence (trees win but gap to "perfect" remains)
2. **Persistence uses finalized lag** — may be unfairly strong if market reacts to interim prices
3. **h5–h24** not improved by ensemble extension
4. **Market microstructure** partially captured; SMF/spread/outage features help but not enough
5. **Zero/capped prices** hurt metrics; classifiers help but not solved
6. **Finalized MCP forecasting** conflates level + revision process

---

## 6. New research direction — Interim → Final correction

**Hypothesis:** At end of day D, K.PTF for D (and possibly D+1 after DAM) is observable. Final MCP for D+1 differs by a **correction** predictable from microstructure + interim dynamics.

**Definitions (at anchor t, horizon h):**
- `interim_baseline_h` = K.PTF at delivery hour t+h
- `target_h` = finalized MCP at t+h
- `target_residual_h` = target_h − interim_baseline_h

**Pipeline scripts:**
1. `fetch_interim_mcp.py` → `data/raw/interim_mcp.csv`
2. `build_interim_residual_dataset.py` → `data/features/interim_residual_next24_v1.parquet`
3. (Future) Train LGBM/LSTM on residual targets — **not done**

**Fetch status:** See Section 7 and `reports/unfinalized_ptf_data_audit.md`.

---

## 7. Interim MCP status

| Item | Status |
|------|--------|
| Endpoint found | `POST .../markets/dam/data/interim-mcp` |
| Auth (CAS/TGT) | Works (same as `update_dataset.py`) |
| Response field | `marketTradePrice` |
| **API quirk** | **Only 24 rows per call** (one day); must loop daily |
| Rate limit | HTTP 429 if <~0.5s between calls; retry after 65s |
| Historical fetch | **Incomplete** (~2160 rows; ['2020-01-01T00:00:00+03:00', '2020-03-30T23:00:00+03:00']) |
| `interim_residual_next24_v1.parquet` | **Not built** (needs full interim history) |

**Why game-changing (if fetch completes):**
- Aligns baseline with **pre-finalization** market info
- Target becomes **revision** not full price level
- May correlate better with SMF/imbalance microstructure already in features

---

## 8. Reproducibility — run order

1. `python update_dataset.py` — finalized PTF (existing)
2. `python run_cleaning.py`
3. `python build_master.py`
4. `python build_features.py`
5. `python run_sequence.py` — LSTM tensors (optional for tree-only work)
6. `python build_tree_features.py`
7. `python build_microstructure_features.py`
8. `python train_tree_advanced.py` — long; models gitignored
9. `python train_microstructure_h1h4.py`
10. `python evaluate_h1h4_validation_weighted_ensemble.py` — **checkpoint metrics**
11. `python fetch_interim_mcp.py` — **daily loop; hours to complete**
12. `python build_interim_residual_dataset.py` — after interim CSV complete

**Seeds:** LightGBM/XGB/sklearn `random_state=42`; LSTM `SEED=42` in `train_lstm*.py`

**Config export:** `reports/reproducibility/model_configs.json`

---

## 9. Critical file index (read first)

| Priority | Path | Why |
|----------|------|-----|
| 1 | `reports/codex_handover.md` | This document |
| 2 | `reports/final_h1h4_summary.json` | Official h1–h4 checkpoint numbers |
| 3 | `reports/experiment_timeline.md` | What was tried and why |
| 4 | `reports/unfinalized_ptf_data_audit.md` | Interim MCP data discovery |
| 5 | `evaluate_h1h4_validation_weighted_ensemble.py` | Best ensemble logic + leakage fix |
| 6 | `tree_advanced/pipeline.py` | Advanced tree train/predict |
| 7 | `features/config.py` | Splits, leakage checklist |
| 8 | `features/interim_residual.py` | New interim baseline design |
| 9 | `fetch_interim_mcp.py` | Interim fetch (daily API loop) |
| 10 | `audit_final_h1h4_pipeline.py` | Pipeline audit template |

---

## 10. Key paths (models, data, reports)

### Parquets
- `data/master/master_hourly_v1.parquet`
- `data/features/lstm_next24_v1.parquet`
- `data/features/lstm_tree_micro_v1.parquet`
- `data/features/lstm_microstructure_next24_v1.parquet`
- `data/features/lstm_residual_next24_v1.parquet`
- `data/features/interim_residual_next24_v1.parquet` — **pending**

### Models (local; mostly gitignored)
- `models/tree_advanced/` (~1.7 GB)
- `models/microstructure_h1h4/`
- `models/lstm_residual.pt`, `models/lstm_baseline.pt`

### Predictions
- `data/predictions/h1h4_validation_weighted_ensemble_predictions.csv`
- `data/predictions/tree_test_predictions.csv`

### Reports
- All `reports/*_metrics.json` for experiments
- `reports/repository_audit.md`, `reports/repository_tree.txt`

---

## 11. Git handover

See `reports/repository_audit.md` for untracked files, safe commit list, and `.gitignore` suggestions.

**No files were deleted in this export.**

---

## 12. Related JSON

Machine-readable bundle: `reports/codex_handover.json`

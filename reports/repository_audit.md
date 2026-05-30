# Repository audit

Generated: 2026-05-30T12:41:14.267798+00:00

## Active scripts (production path)

| Stage | Script | Role |
|-------|--------|------|
| Raw PTF | `update_dataset.py` | Final MCP → `data/ptf_dataset.csv` |
| Interim PTF | `fetch_interim_mcp.py` | K.PTF → `data/raw/interim_mcp.csv` (in progress) |
| Cleaning | `run_cleaning.py` | Raw CSV → `data/clean/*.parquet` |
| Master | `build_master.py` | Clean → `data/master/master_hourly_v1.parquet` |
| Features | `build_features.py` | LSTM tabular `lstm_next24_v1.parquet` |
| Residual features | `build_residual_features.py` | Persistence residual targets |
| Tree features | `build_tree_features.py` | `lstm_tree_micro_v1.parquet` |
| Micro features | `build_microstructure_features.py` | `lstm_microstructure_next24_v1.parquet` |
| Interim dataset | `build_interim_residual_dataset.py` | `interim_residual_next24_v1.parquet` (blocked until full interim CSV) |
| LSTM train | `train_lstm.py`, `train_lstm_residual.py` | Sequence models |
| Tree train | `train_tree_advanced.py` | Hour×horizon LightGBM stack |
| Short expert | `train_short_horizon_expert.py` | h1–h4 specialist |
| Micro train | `train_microstructure_h1h4.py`, `train_microstructure_h5h12.py` | Per-horizon LGBM |
| Eval ensemble | `evaluate_h1h4_*.py`, `evaluate_h5h12_*.py` | Post-hoc blends |
| Audits | `audit_*.py`, `export_codex_handover.py` | No training |

## Trusted models (do not delete)

| Path | Use |
|------|-----|
| `models/tree_advanced/` | Best single-model family (~453 MAE h1–h4) |
| `models/microstructure_h1h4/` | Ensemble component |
| `models/short_horizon_expert/` | Reference |
| `models/lstm_residual.pt` | Residual LSTM (beats persistence, loses to tree) |
| `models/lstm_baseline.pt` | Failed direct baseline |

## Leakage: found and fixed

| Issue | Status | Fix |
|-------|--------|-----|
| `regressor_online` on validation for weight search | **Fixed** | Use train-only `regressor` for validation ensemble weights |
| `regressor_online` in-sample val MAE ~12 | Documented | Misleading; not used for weight pick |
| Test oracle ensemble weights | Documented | Reference only, never primary |
| Fixed 0.7/0.3 test blend in reports | Documented | Not primary metric |
| Finalized MCP same-hour in interim features | **Designed out** | `features/interim_residual.py` uses lagged interim only in X |
| Feature matrix same-hour gen/cons | Review | `features/config.py` LEAKAGE_CHECKLIST |

## Failed / deprecated experiments

| Experiment | Outcome | Notes |
|------------|---------|-------|
| Direct LSTM | **Failed** | Test MAE ~1085 vs persistence ~546 |
| Tree horizon (simple) | Weak | Superseded by advanced tree |
| Advanced tree h1–h4 feature tune (broken) | **Failed run** | Column_0 importance mapping bug |
| Microstructure alone h1–h4 | Worse than tree | ~462 vs ~453 |
| h5–h12 val-weighted ensemble | **No gain** | All weights → 1.0 (advanced only) |
| h5–h12 micro alone | Mixed | Lower mean MAE than tree on test but not selected on val |

## Duplicate / overlapping scripts

- `build_tree_features.py` (root) → wraps `features/build_tree_features.py`
- `build_residual_features.py` / `build_features.py` — same pattern
- `tree_test_predictions.csv` mirrors `tree_advanced_test_predictions.csv`

## Git status summary

```
 M .gitignore
 M README.md
 M features/engineering.py
 M features/report.py
 M requirements.txt
 M sequence/report.py
?? audit_final_h1h4_pipeline.py
?? audit_unfinalized_ptf_data.py
?? baseline_persistence.py
?? build_interim_residual_dataset.py
?? build_microstructure_features.py
?? build_residual_features.py
?? build_tree_features.py
?? data/features/lstm_microstructure_next24_v1.parquet
?? data/features/lstm_residual_next24_v1.parquet
?? data/features/lstm_tree_micro_v1.parquet
?? data/model_residual/
?? data/predictions/
?? data/raw/
?? evaluate_h1h4_ensemble.py
?? evaluate_h1h4_validation_weighted_ensemble.py
?? evaluate_h5h12_validation_weighted_ensemble.py
?? evaluate_lstm_residual_predictions.py
?? export_codex_handover.py
?? features/build_residual_features.py
?? features/build_tree_features.py
?? features/interim_residual.py
?? features/microstructure.py
?? features/microstructure_v2.py
?? features/residual_config.py
?? fetch_interim_mcp.py
?? reports/advanced_tree_h1h4_feature_selection.json
?? reports/advanced_tree_h1h4_feature_selection.md
?? reports/final_h1h4_summary.json
?? reports/final_h1h4_summary.md
?? reports/h1h4_ensemble_metrics.json
?? reports/h1h4_ensemble_metrics.md
?? reports/h1h4_validation_weighted_ensemble_metrics.json
?? reports/h1h4_validation_weighted_ensemble_metrics.md
?? reports/h5h12_validation_weighted_ensemble_metrics.json
?? reports/h5h12_validation_weighted_ensemble_metrics.md
?? reports/lstm_baseline_metrics.json
?? reports/lstm_baseline_metrics.md
?? reports/lstm_residual_metrics.json
?? reports/lstm_residual_metrics.md
?? reports/microstructure_feature_report.json
?? reports/microstructure_feature_report.md
?? reports/microstructure_h1h4_metrics.json
?? reports/microstructure_h1h4_metrics.md
?? reports/persistence_metrics.json
?? reports/persistence_metrics.md
?? reports/reproducibility/
?? reports/residual_features_report_latest.json
?? reports/residual_features_report_latest.md
?? reports/residual_sequence_report_latest.json
?? reports/residual_sequence_report_latest.md
?? reports/short_horizon_expert_metrics.json
?? reports/short_horizon_expert_metrics.md
?? reports/tree_advanced_metrics.json
?? reports/tree_advanced_metrics.md
?? reports/tree_baseline_metrics.json
?? reports/tree_baseline_metrics.md
?? reports/tree_features_report_latest.json
?? reports/tree_features_report_latest.md
?? reports/unfinalized_ptf_data_audit.json
?? reports/unfinalized_ptf_data_audit.md
?? run_sequence_residual.py
?? sequence/residual_config.py
?? sequence/residual_pipeline.py
?? train_lstm.py
?? train_lstm_residual.py
?? train_microstructure_h1h4.py
?? train_microstructure_h5h12.py
?? train_short_horizon_expert.py
?? train_tree_advanced.py
?? train_tree_horizon.py
?? tree_advanced/
?? tune_advanced_tree_h1h4_features.py

```

Untracked files: **72** (see `codex_handover.json` for categories)

## .gitignore recommendations (do not delete locally)

Already ignored: `models/tree_advanced/`, `*.npy`, `reports/figures/*.png`, timestamped reports.

**Suggest adding to .gitignore:**
- `data/raw/interim_mcp.csv` (large, regenerable)
- `data/predictions/*.csv` (regenerable eval outputs)
- `data/features/*.parquet` (regenerable; large)
- `models/microstructure_h5h12/` (already listed)
- `reports/advanced_tree_h1h4_feature_selection.*` (failed experiment)

**Safe to commit (documentation):**
- All `audit_*.py`, `evaluate_*.py`, `export_codex_handover.py`
- `reports/codex_handover.*`, `reports/repository_audit.md`, `reports/experiment_timeline.md`
- `reports/final_h1h4_summary.*`, `reports/*_metrics.json` (non-timestamped)
- `features/interim_residual.py`, `fetch_interim_mcp.py`

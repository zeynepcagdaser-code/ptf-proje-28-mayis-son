#!/usr/bin/env python3
"""Generate Codex handover documentation package (no training/fetch)."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS = PROJECT_ROOT / "reports"

HANDOVER_MD = REPORTS / "codex_handover.md"
HANDOVER_JSON = REPORTS / "codex_handover.json"
REPO_AUDIT_MD = REPORTS / "repository_audit.md"
REPO_TREE = REPORTS / "repository_tree.txt"
TIMELINE_MD = REPORTS / "experiment_timeline.md"


def load_json(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def git_lines(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=PROJECT_ROOT, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        return e.output or str(e)


def build_tree_text() -> str:
    lines = []
    for root, dirs, files in sorted(PROJECT_ROOT.walk(), key=lambda x: str(x[0])):
        if any(p in root.parts for p in {".git", "__pycache__", ".cursor"}):
            continue
        rel = root.relative_to(PROJECT_ROOT)
        depth = len(rel.parts)
        indent = "  " * depth
        if rel != Path("."):
            lines.append(f"{indent}{rel.name}/")
        for f in sorted(files):
            if f.endswith(".pyc"):
                continue
            fp = root / f
            try:
                size = fp.stat().st_size
                lines.append(f"{indent}  {f}  ({size:,} bytes)")
            except OSError:
                lines.append(f"{indent}  {f}")
    return "\n".join(lines[:2500])  # cap for readability


def main() -> None:
    generated = datetime.now(timezone.utc).isoformat()

    metrics = {
        "persistence": load_json(REPORTS / "persistence_metrics.json"),
        "lstm_baseline": load_json(REPORTS / "lstm_baseline_metrics.json"),
        "lstm_residual": load_json(REPORTS / "lstm_residual_metrics.json"),
        "tree_advanced": load_json(REPORTS / "tree_advanced_metrics.json"),
        "short_expert": load_json(REPORTS / "short_horizon_expert_metrics.json"),
        "micro_h1h4": load_json(REPORTS / "microstructure_h1h4_metrics.json"),
        "h1h4_ensemble": load_json(REPORTS / "h1h4_ensemble_metrics.json"),
        "h1h4_val_ensemble": load_json(REPORTS / "h1h4_validation_weighted_ensemble_metrics.json"),
        "h5h12_val_ensemble": load_json(REPORTS / "h5h12_validation_weighted_ensemble_metrics.json"),
        "final_h1h4": load_json(REPORTS / "final_h1h4_summary.json"),
        "unfinalized_audit": load_json(REPORTS / "unfinalized_ptf_data_audit.json"),
    }

    interim_csv = PROJECT_ROOT / "data" / "raw" / "interim_mcp.csv"
    interim_rows = 0
    interim_date_range = None
    if interim_csv.exists():
        import pandas as pd

        im = pd.read_csv(interim_csv)
        interim_rows = len(im)
        if "date" in im.columns and len(im):
            interim_date_range = [str(im["date"].min()), str(im["date"].max())]

    git_status = git_lines(["git", "status", "--short"])
    git_untracked = [ln[3:] for ln in git_status.splitlines() if ln.startswith("??")]

    payload = {
        "generated_at_utc": generated,
        "project_root": str(PROJECT_ROOT),
        "purpose": "Codex handover — full context export without re-running experiments",
        "current_best_checkpoint": {
            "scope": "h1_h4_lstm_anchor_aligned_test",
            "model": "validation_weighted_ensemble",
            "mean_mae_h1_h4": 443.87,
            "weights_validation": {"h1": 0.6, "h2": 0.7, "h3": 0.9, "h4": 1.0},
            "beats_advanced_tree_by": 9.22,
            "advanced_tree_mae": 453.09,
            "persistence_mae_h1_h4_approx": 544.14,
            "report": "reports/final_h1h4_summary.json",
            "predictions": "data/predictions/h1h4_validation_weighted_ensemble_predictions.csv",
        },
        "new_research_direction": {
            "name": "interim_mcp_to_final_mcp_correction",
            "baseline": "K.PTF marketTradePrice (interim-mcp endpoint)",
            "target": "finalized MCP price from /mcp",
            "residual": "final_mcp - interim_mcp at delivery hour",
            "status": "data_audit_done; fetch_partial; dataset_build_pending_full_history",
            "endpoint": "POST /electricity-service/v1/markets/dam/data/interim-mcp",
        },
        "interim_mcp_fetch_status": {
            "csv_path": "data/raw/interim_mcp.csv",
            "rows": interim_rows,
            "date_range": interim_date_range,
            "api_limitation": "One calendar day (24 rows) per request; requires daily loop",
            "rate_limit": "HTTP 429 after ~50 rapid requests; need 0.65s+ delay and retry",
            "historical_fetch_complete": interim_rows > 50000,
            "scripts": ["fetch_interim_mcp.py", "build_interim_residual_dataset.py"],
        },
        "split_policy": {
            "train_years": "2020-2024",
            "validation_year": 2025,
            "test_year": 2026,
            "anchor_files": [
                "data/model/anchor_train.csv",
                "data/model/anchor_val.csv",
                "data/model/anchor_test.csv",
            ],
            "evaluation_alignment": "LSTM anchor_test.csv for model comparison",
        },
        "global_seed": 42,
        "git": {"status_short": git_status, "untracked_count": len(git_untracked)},
        "metrics_files": {k: str(REPORTS / f"{k}.json") for k in metrics if metrics[k]},
    }

    # repository_audit.md
    audit_md = """# Repository audit

Generated: {ts}

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
{git_status}
```

Untracked files: **{untracked_count}** (see `codex_handover.json` for categories)

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
""".format(
        ts=generated,
        git_status=git_status[:4000],
        untracked_count=len(git_untracked),
    )
    REPO_AUDIT_MD.write_text(audit_md, encoding="utf-8")

    # experiment_timeline.md
    timeline = """# Experiment timeline

Chronological research arc for PTF h1–h24 forecasting (Turkey DAM).

## Phase 0 — Data foundation
- Cleaning pipeline → master hourly spine (`ptf_price` from finalized MCP)
- Splits: train 2020–2024, val 2025, test 2026
- LSTM sequence tensors + anchor CSVs

## Phase 1 — Baselines
| Step | Result |
|------|--------|
| Persistence (24h lag) | MAE ~545 — very strong |
| Direct LSTM | MAE ~1085 — **failed** |
| **Insight:** naive seasonal lag is hard to beat with raw sequence model |

## Phase 2 — Residual LSTM
| Step | Result |
|------|--------|
| Target = PTF − persistence | MAE ~536 — beats persistence slightly |
| Still far from useful trading edge vs trees |

## Phase 3 — Tree models
| Step | Result |
|------|--------|
| Simple tree horizon | MAE ~577–603 |
| **Advanced tree** (hour×horizon, classifiers, online refit, micro features) | **~453 MAE h1–h4** — best single family |
| Full 24h aligned test ~589 |

## Phase 4 — Short horizon & microstructure
| Step | Result |
|------|--------|
| Short horizon expert h1–h4 | ~476 — beats persistence, loses to advanced |
| Microstructure LGBM h1–h4 | ~462 — loses to advanced |
| Feature selection tune (advanced tree) | Broken importance mapping — inconclusive |

## Phase 5 — Ensembles (post-hoc, no retrain)
| Step | Result |
|------|--------|
| Fixed 0.7 advanced + 0.3 micro (test) | ~440 MAE — **test-tuned, leakage risk** |
| **Validation-weighted per horizon** | **~444 MAE h1–h4** — **current checkpoint** |
| Weights: h1=0.6, h2=0.7, h3=0.9, h4=1.0 |

## Phase 6 — h5–h12 extension
| Step | Result |
|------|--------|
| Microstructure h5–h12 trained | Val-weighted picks w=1.0 all horizons |
| Ensemble = advanced only; no improvement |

## Phase 7 — Interim MCP pivot (in flight)
| Step | Status |
|------|--------|
| Data audit: no K.PTF in repo | Done — `reports/unfinalized_ptf_data_audit.md` |
| Endpoint identified: `interim-mcp` | Done |
| `fetch_interim_mcp.py` | **Partial** (~60 days of 2020); rate limits |
| `build_interim_residual_dataset.py` | **Not run** (needs full history) |
| New target: final − interim at delivery hour | Designed |

## Why pivot to interim MCP
- Persistence uses **yesterday's finalized price** — not what market knows after DAM clearing
- K.PTF is published **before objection/finalization** — closer to decision-time information
- Modeling **correction** (final − interim) may match how participants revise expectations
- Potential to beat persistence if interim is a better baseline for short horizons
""".format()
    TIMELINE_MD.write_text(timeline, encoding="utf-8")

    # codex_handover.md (main document)
    handover_md = f"""# Codex handover — PTF forecasting project

Generated: {generated}

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
| Historical fetch | **Incomplete** (~{interim_rows} rows; {interim_date_range}) |
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
"""
    HANDOVER_MD.write_text(handover_md, encoding="utf-8")

    payload["documents"] = {
        "codex_handover_md": str(HANDOVER_MD),
        "codex_handover_json": str(HANDOVER_JSON),
        "repository_audit_md": str(REPO_AUDIT_MD),
        "repository_tree_txt": str(REPO_TREE),
        "experiment_timeline_md": str(TIMELINE_MD),
    }
    payload["critical_paths"] = {
        "parquets": [
            "data/master/master_hourly_v1.parquet",
            "data/features/lstm_tree_micro_v1.parquet",
            "data/features/lstm_microstructure_next24_v1.parquet",
            "data/features/interim_residual_next24_v1.parquet",
        ],
        "models": [
            "models/tree_advanced/",
            "models/microstructure_h1h4/",
            "models/lstm_residual.pt",
        ],
        "predictions": [
            "data/predictions/h1h4_validation_weighted_ensemble_predictions.csv",
            "data/predictions/tree_test_predictions.csv",
        ],
        "csv_raw": [
            "data/ptf_dataset.csv",
            "data/raw/interim_mcp.csv",
        ],
    }
    payload["untracked_sample"] = git_untracked[:80]

    HANDOVER_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    tree_text = build_tree_text()
    REPO_TREE.write_text(
        f"# Repository tree (partial, max 2500 lines)\n# Generated: {generated}\n\n{tree_text}\n",
        encoding="utf-8",
    )

    print("Codex handover package written:")
    for p in [HANDOVER_MD, HANDOVER_JSON, REPO_AUDIT_MD, REPO_TREE, TIMELINE_MD]:
        print(" ", p)


if __name__ == "__main__":
    main()

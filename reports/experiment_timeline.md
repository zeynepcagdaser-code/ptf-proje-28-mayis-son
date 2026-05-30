# Experiment timeline

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

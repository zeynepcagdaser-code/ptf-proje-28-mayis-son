# Interim MCP Point-in-Time Audit

Generated: 2026-05-30 16:30 Europe/Istanbul

## Executive Diagnosis

The current historical `interim-mcp` dataset is not safe for correction-model training. Across the available aligned range, `marketTradePrice` is exactly equal to finalized PTF `price`.

This does not prove that K.PTF has no economic value. It proves that the historical API output currently available in `data/raw/interim_mcp.csv` does not demonstrate point-in-time behavior. Used as a model baseline, it would behave as an oracle.

## Inputs

| Item | Value |
|---|---|
| Interim CSV | `data/raw/interim_mcp.csv` |
| Finalized CSV | `data/ptf_dataset.csv` |
| Interim endpoint | `POST /v1/markets/dam/data/interim-mcp` |
| Finalized endpoint | `POST /v1/markets/dam/data/mcp` |
| Status endpoint | `GET /v1/markets/dam/data/interim-mcp-published-status` |

Official EPİAŞ client metadata describes `interim-mcp` as K.PTF: the temporary hourly price formed during the objection process. The path mapping is `markets/dam/data/interim-mcp`; the status endpoint is `markets/dam/data/interim-mcp-published-status`.

## Coverage

| Metric | Value |
|---|---:|
| Interim rows | 21,792 |
| Finalized rows | 56,232 |
| Aligned rows | 21,792 |
| Coverage start | 2020-01-01 00:00 +03:00 |
| Coverage end | 2022-06-26 23:00 +03:00 |
| Unique days | 908 |
| Missing hours in aligned range | 0 |
| Duplicate interim rows | 0 |
| Null rows in key columns | 0 |

## Oracle Leakage Tests

| Test | Result |
|---|---:|
| Correlation, interim vs finalized | 1.000000 |
| Exact match rows | 21,792 / 21,792 |
| Exact match rate | 100.00% |
| Mean absolute correction | 0.00 TL/MWh |
| Max absolute correction | 0.00 TL/MWh |
| Nonzero rows at epsilon 1e-9 | 0 |

The equality holds under both timezone-aware timestamp joins and naive `date/hour` joins. This rules out the main join logic as the cause.

Shift sanity checks also rule out a timezone displacement: only the zero-hour shift produces exact equality. A +1 hour shift gives MAE about 53.03 TL/MWh; a -1 hour shift gives MAE about 53.06 TL/MWh; a +24 hour shift gives MAE about 86.48 TL/MWh.

Year-level equality also holds:

| Year | Rows | Mean absolute diff | Max absolute diff |
|---:|---:|---:|---:|
| 2020 | 8,784 | 0.00 | 0.00 |
| 2021 | 8,760 | 0.00 | 0.00 |
| 2022 | 4,248 | 0.00 | 0.00 |

## Live Endpoint Checks

Short live API checks showed the same behavior for representative days:

| Day | Interim rows | Final rows | First-hour interim | First-hour final | Interim avg | Final avg |
|---|---:|---:|---:|---:|---:|---:|
| 2020-01-01 | 24 | 24 | 311.65 | 311.65 | 284.06 | 284.06 |
| 2022-06-26 | 24 | 24 | 2240.00 | 2240.00 | 1869.27 | 1869.27 |
| 2026-05-30 | 24 | 24 | 171.00 | 171.00 | 142.53 | 142.53 |
| 2026-05-31 | 24 | 24 | 349.99 | 349.99 | 754.04 | 754.04 |

The status endpoint returned `completed=false` during the test, but it does not expose historical publication timestamps or historical versions. It is useful for gating live snapshot capture, not for reconstructing past point-in-time values.

## Endpoint Semantics

The endpoint name and documentation semantics are “interim MCP / K.PTF”. However, observed historical behavior is consistent with a retrospective canonical value endpoint rather than a point-in-time archive.

The API response contains delivery `date`, `hour`, and `marketTradePrice`, but no `publicationTimestamp`, no `version`, no `lastUpdatedAt`, no `mcpState`, and no revision history. Without those fields, a historical request made today cannot prove what was visible at the original publication time.

Legacy references mention endpoints such as `day-ahead-interim-mcp`, `day-ahead-mcp`, and `mcp-smp` with `mcpState`, but the current validated dataset still lacks a versioned historical snapshot. Legacy behavior needs a separate focused probe before it can be used as research-grade point-in-time data.

## Why This Is Leakage

Using this historical `marketTradePrice` as `interim_baseline_h = interim_mcp(t+h)` would inject the target into the feature set. Since `interim_mcp(t+h) == finalized_mcp(t+h)` for every aligned row, the residual target `finalized_mcp(t+h) - interim_mcp(t+h)` becomes exactly zero.

That creates:

- Retrospective leakage: the API appears to expose today’s canonical value for past delivery hours.
- Oracle baseline: the baseline is already the finalized target.
- Fake correction learning: the model learns a zero residual that cannot be trusted out of sample.
- Invalid backtest: any performance estimate would measure data leakage, not forecast skill.
- Future information contamination: target-hour final information enters the anchor-time feature matrix.

## Snapshot Strategy

The safe way to build K.PTF history is append-only point-in-time logging.

Recommended architecture:

- Run a snapshot job around the DAM publication window and query `interim-mcp-published-status`.
- When `completed=true`, fetch `interim-mcp` for the relevant delivery day and store the full response with `snapshot_ts`.
- Continue hourly snapshots until finalized MCP is available, so late changes can be detected.
- Store response hashes to detect silent overwrites.
- Use an append-only table keyed by `snapshot_ts`, `delivery_date`, and `delivery_hour`.
- Keep finalized MCP in a separate table and join only for target construction after the fact.

Minimum columns:

| Column | Purpose |
|---|---|
| `snapshot_ts` | When our system observed the value |
| `delivery_date` | Market delivery date |
| `delivery_hour` | Market delivery hour |
| `marketTradePrice` | Observed K.PTF value |
| `published_status_completed` | Status endpoint gate |
| `source_endpoint` | Reproducibility |
| `response_hash` | Overwrite/revision detection |
| `fetch_run_id` | Operational traceability |

Storage recommendation: partitioned parquet by `snapshot_date` and `delivery_date` for the archive. CSV can remain a lightweight export, but it should not be the canonical point-in-time store.

## Recommended Architecture

1. `fetch_interim_snapshot.py` captures append-only K.PTF snapshots.
2. `audit_interim_snapshots.py` verifies publication coverage, duplicate snapshots, delivery-day completeness, and revisions.
3. `build_interim_point_in_time_dataset.py` selects the latest snapshot available before the chosen anchor time.
4. `build_interim_residual_dataset.py` constructs `finalized_mcp - observed_interim_mcp` only from point-in-time safe snapshots.

## Final Answer

Real point-in-time K.PTF data is required before correction forecasting can be backtested reliably.

With the current historical `interim-mcp` CSV, the answer is no: correction forecasting cannot be trusted because the historical interim series is exactly equal to finalized PTF across the tested coverage. The next safe research step is to start point-in-time snapshot logging and exclude the current historical interim CSV from model training until endpoint semantics are proven with versioned or archived observations.

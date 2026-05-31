# K.PTF Snapshot Market Regime Detector Design

## Scope

This is an analysis/design note only. No model is trained, no feature engineering pipeline is changed, and no finalized MCP join is used.

Input considered:

- `data/snapshots/interim_mcp_snapshots.csv`
- Current remote snapshot coverage: 24 hourly rows for `2026-05-31`
- `published_status_completed=False`

Observed prices in the available snapshot:

- min: `0.00`
- median: `130.01`
- mean: `754.04`
- p75: `333.50`
- p90: `3565.00`
- p95: `4300.00`
- max: `4300.01`
- zero/negative pressure hours, `price <= 0`: `10`
- near-zero pressure hours, `price <= 50`: `12`
- cap/spike hours, `price >= 4300`: `3`

Because the current archive contains only one delivery day, the rules below should be treated as an initial operational heuristic, not a statistically calibrated detector. Calibration should wait until several weeks of point-in-time snapshots accumulate.

## Regime Definitions

The detector should assign one regime per `snapshot_ts + delivery_hour`.

Recommended regimes:

1. `negative_zero_pressure`
2. `normal`
3. `tight`
4. `spike_cap`

The labels are intentionally market-structure oriented rather than model oriented. They describe supply-demand pressure visible in K.PTF at publication/snapshot time.

## Initial Threshold Proposal

Use absolute thresholds first. They are easier to audit and safer while the snapshot history is short.

| Regime | Initial Rule | Interpretation |
|---|---:|---|
| `negative_zero_pressure` | `marketTradePrice <= 50` | oversupply / renewable or must-run pressure; includes zero-price hours |
| `normal` | `50 < marketTradePrice < 1500` | ordinary price formation zone |
| `tight` | `1500 <= marketTradePrice < 4000` | scarcity pressure, but not at/near cap |
| `spike_cap` | `marketTradePrice >= 4000` | cap-proximal or scarcity spike behavior |

Why these thresholds:

- `0` and near-zero prices are economically distinct from simply low prices.
- `50 TL/MWh` is a conservative near-zero band that catches hours where the order book is effectively under severe downward pressure.
- `1500 TL/MWh` separates ordinary formation from materially tight conditions without overreacting to moderate evening peaks.
- `4000 TL/MWh` captures cap-proximal behavior. The current snapshot has `4300` and `4300.01` hours, which are clearly spike/cap.

For production use, keep threshold config versioned:

```text
REGIME_ZERO_MAX = 50
REGIME_TIGHT_MIN = 1500
REGIME_SPIKE_MIN = 4000
```

## Volatility Band Proposal

Regime detection should not only look at level. K.PTF shape matters: a day with both zero hours and cap hours is structurally unstable even if many hours look normal.

Compute volatility features within each `snapshot_ts` and delivery day:

- `daily_price_std`
- `daily_price_range = max - min`
- `hourly_abs_delta = abs(price_t - price_t-1)`
- `hourly_pctile_rank_within_snapshot`
- `rolling_3h_range`

Suggested volatility bands:

| Band | Rule | Meaning |
|---|---:|---|
| `calm` | `daily_price_range < 750` and `daily_price_std < 300` | mostly stable day |
| `volatile` | `750 <= daily_price_range < 2500` or `300 <= daily_price_std < 900` | meaningful within-day shape risk |
| `extreme` | `daily_price_range >= 2500` or `daily_price_std >= 900` | regime-switching day; likely zero-to-spike or scarcity event |

Current snapshot would be `extreme`:

- range is about `4300`
- standard deviation is about `1437`
- the same day contains both zero-price and cap-price hours

## Relative Threshold Overlay

Once enough snapshots accumulate, add relative thresholds on top of the absolute rules:

- `spike_cap` if price is above rolling `p95` of same hour-of-day and above `REGIME_TIGHT_MIN`
- `negative_zero_pressure` if price is below rolling `p10` of same hour-of-day and below `REGIME_ZERO_MAX`
- `tight` if price is above rolling `p80` or `p90` but below cap band

Recommended calibration windows:

- same hour-of-day, last 30 delivery days
- same weekday/weekend class, last 8 weeks
- separate holiday flag later, but not in this design step

Do not use finalized MCP for calibration if the detector is meant to run point-in-time. Use only accumulated K.PTF snapshots available at or before the current `snapshot_ts`.

## Transition Logic

A useful detector should identify not only hour labels but transitions:

```text
negative_zero_pressure -> normal
normal -> tight
tight -> spike_cap
spike_cap -> normal
spike_cap -> negative_zero_pressure
```

Recommended transition features for later analysis:

- `regime_prev_hour`
- `regime_next_hour` for descriptive audit only, not live signal
- `regime_change = regime != regime_prev_hour`
- `transition_type = regime_prev_hour + "_to_" + regime`
- `abs_delta_from_prev_hour`
- `hours_since_last_spike_cap`
- `hours_since_last_zero_pressure`

For real-time use, only backward-looking transitions are safe:

- previous hour regime within the same delivery day
- previous snapshot of the same delivery hour
- previous snapshot response hash change

Forward-looking `next_hour` fields are useful for report visualization, but they should not enter any prediction input unless the full delivery curve is known at the same `snapshot_ts`. In K.PTF snapshots, the full day curve is usually visible together, so full-curve descriptors may be valid for day-ahead simulation, but this must be documented explicitly.

## Snapshot Version Transition Logic

Because this archive is point-in-time, the detector should also compare versions of the same `delivery_hour` across snapshots:

- `price_changed_since_last_snapshot`
- `price_delta_since_last_snapshot`
- `regime_changed_since_last_snapshot`
- `first_seen_snapshot_ts`
- `last_seen_snapshot_ts`
- `snapshot_revision_count`

This is central for leak-free correction research. If a delivery hour moves from `normal` to `tight` across snapshots before finalization, that transition is real market information. If it only appears in retrospective historical endpoints, it is leakage.

## Proposed Rule Order

Use deterministic priority ordering:

```text
if marketTradePrice <= 50:
    regime = "negative_zero_pressure"
elif marketTradePrice >= 4000:
    regime = "spike_cap"
elif marketTradePrice >= 1500:
    regime = "tight"
else:
    regime = "normal"
```

Priority matters because cap/zero behavior is qualitatively different. Absolute cap and zero bands should override relative bands.

## Current Snapshot Regime Sketch

Using the initial thresholds on the current 24-hour snapshot:

- `negative_zero_pressure`: 12 hours
- `normal`: 8 hours
- `tight`: 1 hour
- `spike_cap`: 3 hours

This day is a useful example of a mixed regime day:

- morning/midday zero pressure
- evening scarcity/cap pressure
- extreme within-day volatility

That kind of shape is exactly where persistence-style baselines can be misleading hour by hour, even if they remain strong on average.

## Implementation Notes For Later

When this becomes code, keep it separate from model training:

- `build_market_regime_labels.py` or `features/regime.py`
- pure deterministic labeling
- config-driven thresholds
- no finalized MCP dependency
- no future snapshots for live labels
- report threshold version in output metadata

Suggested output columns:

- `snapshot_ts`
- `delivery_hour`
- `marketTradePrice`
- `market_regime`
- `volatility_band`
- `regime_threshold_version`
- `price_delta_prev_hour`
- `regime_transition_prev_hour`
- `snapshot_revision_count`

## Recommendation

Start with absolute threshold rules now and mark them as `v0`. Do not overfit thresholds to the single current snapshot. After at least 30-60 delivery days of snapshots, recalibrate with hour-of-day rolling quantiles and test regime stability across snapshot versions.

The detector should be treated as market-state annotation first, not a predictive model. Its immediate value is to make K.PTF snapshots auditable: which hours are oversupply, ordinary, tight, or cap-like at the exact point in time the market saw them.

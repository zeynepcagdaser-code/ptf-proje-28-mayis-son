
## Executive Answer

The system that can most realistically beat persistence is not a single stronger price regressor. It is a **regime-aware forecasting system** that first asks:

```text
Is yesterday's market-clearing regime still valid for this delivery hour?
```

Persistence is strong because it carries yesterday's full market-clearing state: hourly shape, fuel-stack regime, price band, weekday pattern, and volatility state. The real alpha is concentrated where that inherited state breaks:

- `normal -> spike_cap`
- `tight -> spike_cap`
- `normal/tight -> negative_zero_pressure`
- weekend/weekday structural shifts
- solar cliff plus evening ramp
- high residual load plus outage/maintenance stress

The final architecture should therefore be a **mixture of regime experts** with persistence as the default anchor, not a global model that tries to learn all price states with one loss function.

No production training, hyperparameter tuning, or new ensemble is started in this step.

## A. Final Regime-Aware Forecasting Architecture

### 1. Point-In-Time Data Layer

Task:

- Build a leak-free feature store.
- Enforce anchor-time availability.
- Keep snapshot versions rather than overwriting market state.

Target:

- Data validity and reproducibility, not prediction.

Feature availability:

- safe: lagged PTF, calendar, public KGÜP, load forecast, wind forecast, outage publications
- safe with timing audit: KGÜP first/latest versions, forecast revisions, outage revisions
- unsafe as ex-ante target-hour features: same-hour finalized PTF, same-hour finalized SMF, same-hour realized YAL/YAT, historical `interim-mcp` that equals finalized PTF

Leakage risk:

- very high if historical K.PTF is treated as point-in-time
- high if forecast files contain retroactively revised values

Best horizons:

- all horizons; this is the foundation.

Status:

- mandatory. The append-only interim MCP snapshot pipeline is the right direction.

### 2. Regime Classifier

Task:

- Predict the probability of each market regime:
  - `negative_zero_pressure`
  - `normal`
  - `tight`
  - `spike_cap`

Target:

- regime probability, not price level.

Feature families:

- lag24 price band and regime
- hour, weekday, holiday/weekend
- residual load forecast
- solar/wind pressure
- KGÜP thermal/gas/hydro/VRE shares
- outage stress
- previous volatility band
- K.PTF snapshot curve once enough point-in-time data exists

Leakage risk:

- medium. Regime labels can use finalized PTF for training labels, but all inputs must be anchor-time safe.

Horizon usefulness:

- strongest for h1-h24 delivery-curve classification
- crucial for h17-h23 spike risk and h8-h16 zero-pressure risk

Why this is first:

- The biggest persistence failures are regime transitions, not ordinary level errors.
- If the model gets the regime wrong, a level regressor has little chance.

### 3. Normal Regime Forecaster

Task:

- Forecast continuous price or residual vs persistence for stable non-tail hours.

Target:

- `PTF - lag24_PTF`, conditional on `normal` or stable `tight`.

Feature families:

- lag24 PTF
- same-hour rolling median
- weekday/hour
- `load_minus_kgup`
- KGÜP stack
- wind/load share
- lagged volatility

Leakage risk:

- low if only ex-ante public features are used.

Useful horizons:

- h1-h12 strongest
- full h24 possible but less reliable around regime transitions

Expected role:

- Incremental improvement over persistence in stable hours.
- It should not be responsible for cap or zero events.

### 4. Zero-Pressure Model

Task:

- Detect oversupply/near-zero price risk.

Target:

- `P(PTF <= 50)`
- conditional price within zero-pressure band

Feature families:

- wind forecast
- solar/KGÜP güneş
- VRE/load ratio
- load forecast
- hydro must-run proxies
- lag24 zero-pressure state
- midday hour flags

Leakage risk:

- low for public forecasts
- medium if weather or generation forecasts are retrospectively revised

Useful horizons:

- strongest during solar window, roughly h8-h16

Why separate:

- Zero prices are compressed. A continuous regressor can look good on MAE while missing the regime entirely.

### 5. Spike / Cap Risk Model

Task:

- Detect scarcity and cap-entry risk.

Target:

- `P(PTF >= 4000)`
- probability of `tight -> spike_cap`
- cap-hour count in delivery day

Feature families:

- residual load ramp
- solar cliff# Final Regime-Aware PTF Forecasting Architecture & Feasibility Synthesis

- evening ramp hour
- KGÜP gas/thermal share
- active outage/maintenance stress
- low wind/solar relief
- previous-day cap/tight state
- K.PTF curve shape once snapshots mature

Leakage risk:

- medium. Outage timing and revision timing must be audited.
- high if same-day realized balancing data is used as if known.

Useful horizons:

- strongest h17-h23

Why separate:

- Spike/cap is a classification-plus-saturation problem. Once price is at cap, level variation is not the main signal.

### 6. Correction Layer

Task:

- After real point-in-time K.PTF snapshots accumulate, model:

```text
finalized_PTF - observed_KPTF_snapshot
```

Target:

- correction residual from observed K.PTF to finalized PTF.

Feature families:

- snapshot timestamp
- published status
- snapshot revision count
- K.PTF curve shape
- stress detector outputs
- late outage/KGÜP changes

Leakage risk:

- very high until enough true point-in-time snapshots exist.
- historical `interim-mcp` must not be used because it matched finalized PTF exactly in the audit.

Useful horizons:

- post-publication correction, not pre-market day-ahead forecasting.

Recommendation:

- Defer training. Continue snapshot collection first.

### 7. Uncertainty Estimator

Task:

- Produce intervals and probabilities, not just point forecasts.

Target:

- quantiles by regime
- conformal intervals by regime
- probability of cap/zero/tight

Feature families:

- regime probabilities
- volatility band
- persistence failure signals
- transition indicators

Leakage risk:

- low if calibrated on validation only.

Why needed:

- Market simulation needs distributions. A single point forecast hides the most important PTF risk: tail-regime probability.

### 8. Market Stress Detector

Task:

- Produce interpretable stress flags.

Target:

- diagnostic stress state:
  - solar cliff
  - residual load spike
  - outage stack high
  - gas marginality likely
  - SMF/PTF stress historical state

Feature families:

- residual load
- KGÜP thermal/gas
- outages
- lagged SMF/YAL-YAT
- wind/solar forecast

Leakage risk:

- low if same-hour realized balancing is excluded.

Role:

- explain forecasts
- route hours to the right expert
- support market simulation scenarios

## B. Persistence Failure Heatmap

### By Regime

Persistence MAE by finalized regime:

| Regime | MAE |
|---|---:|
| `spike_cap` | `516.0` |
| `tight` | `360.4` |
| `negative_zero_pressure` | `267.4` |
| `normal` | `197.4` |

Conclusion:

- Persistence is very strong in normal hours.
- It weakens in tight/spike hours.
- The main problem is not price level in stable regimes; it is tail-regime identification.

### By Transition

Worst same-hour previous-day regime transitions:

| Transition | Count | MAE |
|---|---:|---:|
| `normal -> spike_cap` | `38` | `3612.4` |
| `spike_cap -> normal` | `24` | `3512.6` |
| `negative_zero_pressure -> tight` | `106` | `2126.9` |
| `tight -> negative_zero_pressure` | `37` | `1799.1` |
| `tight -> spike_cap` | `500` | `1128.5` |
| `normal -> tight` | `2486` | `1125.1` |
| `tight -> normal` | `2565` | `1017.2` |
| `spike_cap -> tight` | `511` | `1013.9` |

Stable regimes:

| Transition | MAE |
|---|---:|
| `normal -> normal` | `89.1` |
| `negative_zero_pressure -> negative_zero_pressure` | `7.2` |
| `spike_cap -> spike_cap` | `152.4` |
| `tight -> tight` | `269.9` |

This is the clearest alpha map:

```text
alpha = regime transition detection
not average price-level smoothing
```

### By Hour

Worst persistence hours by MAE:

| Hour | MAE |
|---:|---:|
| 11 | `424.8` |
| 10 | `421.7` |
| 9 | `409.5` |
| 15 | `393.5` |
| 14 | `391.2` |
| 13 | `389.5` |
| 8 | `387.2` |

Best persistence hours are evening cap-prone hours in average MAE terms:

- 19: `169.6`
- 20: `156.2`
- 21: `172.3`

This looks counterintuitive until we remember price-band clustering: evening hours are often persistently high. The big failures happen when they switch band.

### Hour x Regime Hotspots

High-error combinations:

- 11 `spike_cap`
- 15 `spike_cap`
- 14 `spike_cap`
- 8 `spike_cap`
- 10 `tight`
- 11 `tight`
- 13 `tight`

Interpretation:

- Midday and shoulder-hour spikes are much harder than normal evening tightness.
- Evening high prices are common enough that persistence often already expects them.

### Residual Load

Persistence error by residual-load quintile:

| Residual bin | MAE |
|---|---:|
| Q1 low | `435.3` |
| Q2 | `318.2` |
| Q3 | `265.5` |
| Q4 | `241.6` |
| Q5 high | `213.6` |

This does not mean high residual load is unimportant. It means high residual load is often persistent and already priced by lag24. The real signal is **change in residual load vs yesterday**, especially solar cliff/ramp deviations.

### Load-KGÜP Gap

Persistence error by `load_forecast - KGÜP toplam` quintile:

| Gap bin | MAE |
|---|---:|
| Q5 high | `389.0` |
| Q1 low | `338.6` |
| Q4 | `277.9` |
| Q2 | `240.6` |
| Q3 | `227.8` |

Both extremes are hard:

- high positive gap: demand/supply tightness
- very low/negative gap: oversupply or renewable pressure

### Maintenance / Outage Stress

For 2026, active outage/maintenance operator-power proxy shows highest persistence failure when high residual load combines with high maintenance:

| Condition | MAE |
|---|---:|
| Q3 outage + Q4 residual load | `1013.3` |
| Q4 outage + Q4 residual load | `965.9` |

Outages alone are not enough. They become useful when residual load is high and the stack is already tight.

## C. Data Gap Analysis

### 1. Order Book / Bid Curve

Why critical:

- PTF is the intersection of supply/demand bids.
- The last few MW can move price by hundreds or thousands of TL.
- Spike/cap and zero-pressure are order-book shape events.

Public access:

- Not available at full participant bid curve granularity.

Proxy:

- K.PTF curve snapshots
- KGÜP stack
- residual load
- price band history
- volatility bands

Alpha potential:

- Very high. This is the single most important missing data source for 1-2 TL accuracy.

### 2. Participant Strategy

Why critical:

- Bids reflect risk appetite, fuel constraints, opportunity cost, hydro strategy, and expected balancing value.

Public access:

- No.

Proxy:

- historical regime behavior by hour/participant class is not directly observable
- aggregate market state can only approximate it

Alpha potential:

- Very high.

### 3. Intraday Revisions / True Point-In-Time K.PTF

Why critical:

- Historical `interim-mcp` behaved like an oracle in the audit.
- Real K.PTF snapshots can reveal publication-time market belief and revision dynamics.

Public access:

- Current state is available, but historical point-in-time archive must be built by us.

Proxy:

- append-only snapshot pipeline, already started.

Alpha potential:

- High for correction forecasting and market-state monitoring.

### 4. Balancing Expectation

Why critical:

- Day-ahead bids can include expectation of balancing costs or scarcity.
- SMF/YAL-YAT explain stress after the fact.

Public access:

- SMF/YAL-YAT are public, but same-hour values are not ex-ante safe.

Proxy:

- lagged SMF/PTF spread
- lagged YAL/YAT stress
- reserve/ancillary service history
- regime-specific balancing stress indicators

Alpha potential:

- Medium-high.

### 5. Reserve Scarcity / Ancillary Markets

Why critical:

- Scarcity of flexibility affects cap risk and gas marginality.

Public access:

- Partial, depending on PFK/SFK and ancillary service endpoints.

Proxy:

- lagged PFK/SFK
- lagged SMF
- YAL/YAT
- residual load ramp

Alpha potential:

- Medium-high, strongest in tight/spike regimes.

### 6. Congestion / Transmission Constraints

Why critical:

- Local constraints can alter dispatch and marginality.
- National aggregate features can miss constraint-driven price stress.

Public access:

- Limited.

Proxy:

- regional outage/plant maintenance
- import/export schedules
- congestion-like historical stress residuals

Alpha potential:

- Medium, but important for unexplained spikes.

### 7. Unit Commitment Visibility

Why critical:

- Whether a gas/coal/hydro unit is actually available and committed determines marginal stack.

Public access:

- Partial through KGÜP and outages.

Proxy:

- KGÜP fuel stack
- active maintenance
- lagged realized generation
- unit-level outage text classification

Alpha potential:

- High.

### 8. Fuel Forward Prices / Gas Marginality

Why critical:

- Gas marginality shifts the price level of tight regimes.
- Fuel opportunity cost changes slowly but materially.

Public access:

- Requires external market data.

Proxy:

- FX
- imported coal/gas price indices
- KGÜP gas share
- lagged high-tight/spike regimes

Alpha potential:

- High for medium-term level shifts, less for exact hourly spikes.

### 9. Hydro Strategy

Why critical:

- Hydro opportunity cost and reservoir strategy affect evening prices and scarcity relief.

Public access:

- Partial/indirect.

Proxy:

- KGÜP hydro
- dammed hydro generation
- seasonality
- water year/month
- lagged hydro dispatch pattern

Alpha potential:

- Medium-high.

### 10. Renewable Forecast Error

Why critical:

- Wind/solar forecast errors create zero-pressure or scarcity surprises.

Public access:

- Forecasts are public; revision history may require snapshots.

Proxy:

- forecast vs realized error after the fact
- point-in-time forecast snapshots if available

Alpha potential:

- High around zero-pressure and spike relief transitions.

### 11. Cross-Border Flow

Why critical:

- Import/export constraints affect residual balance and scarcity.

Public access:

- Partial.

Proxy:

- `importExport` in realized generation
- flow schedules if available

Alpha potential:

- Medium.

### 12. Gas Marginality State

Why critical:

- Determines whether the market is in cheap hydro/coal, gas, or scarcity stack.

Public access:

- Indirect.

Proxy:

- KGÜP gas share
- lagged natural gas realized generation
- SMF/PTF spread
- high-tight band persistence

Alpha potential:

- High in tight regimes.

## D. Information-Theoretic Limit

### Can public EPİAŞ + weather + outage + KGÜP + K.PTF snapshots reach 1-2 TL?

No, not for pre-market finalized PTF forecasting.

1-2 TL requires information close to the actual clearing order book:

- participant bid curves
- hidden strategic bidding
- exact unit commitment constraints
- opportunity costs
- reserve scarcity
- congestion
- hydro strategy
- balancing expectations
- last-minute forecast revisions

These are not fully public.

### Realistic Error Ranges

These are research priors, not trained results:

| Task | Realistic MAE range |
|---|---:|
| stable normal h1-h4 | `150-300 TL/MWh` |
| all-regime h1-h4 public-data system | `350-450 TL/MWh` |
| full h24 day-ahead | `450-700 TL/MWh` |
| zero/spike transition hours | often `700-1500+ TL/MWh` unless classified correctly |
| post-publication K.PTF correction | unknown until real snapshots accumulate |

Existing checkpoint:

- persistence h1-h4: `544.14`
- best validation-weighted ensemble h1-h4: `443.87`

So a regime-aware system has room to improve, but the improvement is bounded by unobserved microstructure.

### Persistence Ceiling

Persistence is already a compressed representation of yesterday's market clearing. It is strongest when:

- same price band persists
- fuel regime persists
- hour shape persists
- volatility regime persists

It fails when:

- price band changes
- residual load changes relative to yesterday
- outage/fuel stack changes
- zero/cap event appears or disappears

The ceiling for pure persistence has already been reached; the next gain must come from transition detection.

### The 1-2 TL Band

1-2 TL is not a realistic target for day-ahead finalized PTF with public aggregate data.

It may become plausible only in a narrow post-publication correction setting if:

- K.PTF snapshot is truly point-in-time
- objection/finalization corrections are usually tiny
- snapshot is taken after publication completion

But this is a different problem: K.PTF-to-final correction, not pre-clearing PTF forecasting.

## E. Final Recommendation

### 1. Most Promising Modeling Direction

Build a regime-aware mixture:

```text
regime classifier
    -> zero-pressure expert
    -> normal/tight residual forecaster
    -> spike/cap risk expert
    -> uncertainty layer
    -> optional K.PTF correction layer later
```

Persistence should remain the default anchor. Models should correct it only when evidence says the regime has shifted.

### 2. Modeling Direction To Leave Behind

Stop investing in:

- single global direct LSTM for price level
- training on historical `interim-mcp` as point-in-time
- global feature importance reports
- one-size-fits-all regression loss across zero, normal, tight, and cap hours

### 3. Most Critical Missing Data

Order book / bid curve is the most critical missing data.

Second tier:

- true point-in-time K.PTF revisions
- participant strategy proxies
- unit commitment visibility
- reserve/ancillary scarcity
- fuel forward/gas marginality data

### 4. Strongest Feature Family

The strongest production-ready family is:

```text
KGÜP stack + load forecast + wind/solar pressure + residual load
```

Outages become powerful only as interactions:

```text
outage stress * high residual load * evening ramp
```

SMF is excellent as an explanatory diagnostic, but same-hour SMF is not an ex-ante PTF predictor.

### 5. Realistic Horizon

Most realistic:

- h1-h4: strongest
- h1-h12: useful with regime-aware design
- full h24: should be probabilistic and regime-based, not only point forecast

### 6. Highest Alpha

Highest alpha is in transitions:

- `normal -> spike_cap`
- `tight -> spike_cap`
- `tight -> negative_zero_pressure`
- `negative_zero_pressure -> tight`
- midday/shoulder-hour spikes
- high residual load plus high maintenance stack
- solar cliff plus evening ramp

### 7. Next Pipeline To Build

Recommended next non-training pipeline:

1. `build_regime_labels.py`
2. `audit_persistence_failures.py`
3. `build_point_in_time_feature_store.py`
4. `audit_feature_availability.py`
5. `evaluate_by_regime.py`

Only after these exist should training restart.

### 8. Data To Snapshot Regularly

Must snapshot:

- interim MCP / K.PTF
- KGÜP first version and latest version
- load forecast if revisions exist
- wind/solar forecasts if revisions exist
- outage publications
- published status endpoints
- possibly PFK/SFK and other ancillary service states

The key is version history, not just latest CSV.

### 9. Production-Ready Feature Families

Production-ready or near-ready:

- calendar/hour/weekday
- lagged PTF
- KGÜP stack
- load forecast
- wind forecast
- outage data after publication timing audit
- lagged SMF/YAL-YAT summaries

Not yet production-ready:

- historical K.PTF as point-in-time
- same-hour finalized balancing variables
- outage MW loss fields without robust interpretation
- global microstructure aggregates without timing proof

### 10. Is Regime-Aware Architecture Necessary?

Yes.

A single model is being asked to learn four different markets:

- oversupply/zero-price market
- normal clearing market
- tight marginal-stack market
- cap/scarcity market

Those regimes have different feature usefulness, different error distributions, and different information limits.

The final system should be market-aware first and model-aware second. The architecture should encode the market structure instead of asking one learner to rediscover it from noisy aggregate data.

## Final Synthesis

The best achievable PTF system with public data is not an oracle. It will not produce 1-2 TL day-ahead accuracy across all hours.

But it can materially beat persistence where it matters most:

- by detecting when yesterday's regime is invalid
- by assigning cap/zero probabilities
- by routing hours to specialized regime experts
- by producing uncertainty intervals for market simulation
- by using K.PTF snapshots only when they are truly point-in-time

The next research step is not bigger training. It is building the leakage-safe regime feature store and evaluation harness that can prove improvements by regime transition.

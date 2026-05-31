# Persistence Decomposition Report

## Scope

This report explains why the persistence baseline is unusually strong in the PTF forecasting project.

No model is trained here. No feature engineering pipeline is changed. The analysis uses existing datasets only:

- `data/ptf_dataset.csv`
- `data/kgup_combined.csv` as a fuel mix / scheduled generation proxy
- Existing checkpoint: `reports/final_h1h4_summary.md`

## Baseline Context

The canonical persistence rule in this repo is:

```text
prediction(t+h) = PTF(t+h-24)
```

For the h1-h4 checkpoint:

- persistence mean MAE: `544.14 TL/MWh`
- best validation-weighted ensemble mean MAE: `443.87 TL/MWh`
- improvement over persistence: about `18.4%`

So persistence is beatable, but not by much. The reason is not that the market is trivial. It is that a large part of PTF is structured by slowly moving regimes and repeated intraday market mechanics.

## 1. Hourly Shape Persistence

PTF has a persistent daily shape:

- low overnight / midday pressure
- morning and evening demand ramps
- evening scarcity premium
- solar-driven midday trough in recent years

Measured on `2020-01-01` to `2026-05-31`:

| Lag rule | MAE | RMSE | Correlation |
|---|---:|---:|---:|
| previous hour, `lag1` | `191.03` | `357.12` | `0.9558` |
| same hour previous day, `lag24` | `294.87` | `549.86` | `0.8951` |
| same hour previous week, `lag168` | `312.45` | `562.12` | `0.8902` |

Daily shape correlation with the previous day:

- mean: `0.636`
- median: `0.712`
- p25: `0.495`
- p75: `0.837`

This is the core reason persistence works: yesterday's curve often carries the right clock-hour structure even when the level is imperfect.

Recent 365-day hourly medians show a strong recurring curve:

| Hour block | Typical behavior |
|---|---|
| 00-06 | still high but fading overnight prices |
| 10-15 | midday weakness, solar pressure |
| 17-21 | evening premium / tightness |
| 19-21 | strongest median price zone |

In the recent year, median prices around 19:00-21:00 are near `3399 TL/MWh`, while 12:00 median is about `1250 TL/MWh`. A same-hour previous-day rule captures much of this shape automatically.

## 2. Weekday Effect

The weekly calendar matters, but it is not enough to beat the daily shape on its own.

Measured MAE:

- `lag24`: `294.87`
- `lag168`: `312.45`
- `lag168 / lag24` error ratio: `1.057`

So previous week same hour is informative, but previous day same hour is slightly stronger overall.

Lag24 MAE by day of week:

| Day | MAE |
|---|---:|
| Monday | `520.0` |
| Tuesday | `251.1` |
| Wednesday | `216.0` |
| Thursday | `213.4` |
| Friday | `207.6` |
| Saturday | `272.6` |
| Sunday | `388.7` |

Interpretation:

- Tuesday-Friday are highly persistent because business-day structure repeats.
- Monday is hard because Sunday-to-Monday is a structural transition.
- Sunday is hard because Saturday-to-Sunday demand and solar/renewable effects differ.
- A pure daily persistence baseline is strongest in stable weekday blocks and weakest across weekend/weekday boundaries.

This explains why models can improve at short horizons by learning calendar transitions, but the baseline remains hard to crush.

## 3. Fuel Regime Persistence

PTF is anchored by fuel-stack regimes that do not reset every hour.

Using KGÜP as a scheduled fuel mix proxy:

| Variable | lag1 corr | lag24 corr | lag168 corr |
|---|---:|---:|---:|
| total KGÜP | `0.974` | `0.878` | `0.895` |
| natural gas KGÜP | `0.989` | `0.857` | `0.805` |
| thermal KGÜP | `0.991` | `0.876` | `0.828` |
| coal KGÜP | `0.992` | `0.913` | `0.839` |
| hydro KGÜP | `0.978` | `0.960` | `0.937` |
| variable renewable KGÜP | `0.980` | `0.855` | `0.702` |

Fuel-share persistence is also high:

| Share | lag24 corr |
|---|---:|
| gas share | `0.867` |
| coal share | `0.914` |
| hydro share | `0.970` |
| VRE share | `0.784` |

Gas regime buckets:

```text
gas_off:  gas_share <= 5%
gas_low:  5-15%
gas_mid:  15-30%
gas_high: >30%
```

Persistence of gas regime:

- same gas regime as previous day: `69.5%`
- same gas regime as previous week: `60.2%`

This is important: if the marginal stack is hydro/coal/gas dominated today, it is often similar tomorrow at the same hour. Persistence inherits this information without explicitly modeling fuel costs.

## 4. Price Band Clustering

PTF does not behave as a smooth Gaussian target. It clusters into bands.

Band definition used here:

| Band | Price range |
|---|---:|
| `zero_pressure` | `<= 50` |
| `low_normal` | `50-500` |
| `normal_mid` | `500-1500` |
| `tight` | `1500-3000` |
| `high_tight` | `3000-4000` |
| `spike_cap` | `>= 4000` |

Band share over the full PTF history:

| Band | Share |
|---|---:|
| `zero_pressure` | `1.7%` |
| `low_normal` | `27.2%` |
| `normal_mid` | `15.2%` |
| `tight` | `42.5%` |
| `high_tight` | `10.7%` |
| `spike_cap` | `2.7%` |

Same-hour previous-day band persistence:

- `79.4%`

Selected transition probabilities from yesterday's same-hour band:

| Previous day band | Same band today |
|---|---:|
| `low_normal` | `90%` |
| `normal_mid` | `63%` |
| `tight` | `83%` |
| `high_tight` | `68%` |
| `spike_cap` | `70%` |
| `zero_pressure` | `46%` |

This is a major reason persistence is strong. It often gets the correct price *region* even when it misses the exact price.

The hardest band is `zero_pressure`: it is less persistent and more event-driven. The most useful model improvements should come from detecting zero-pressure and spike transition days, not from trying to marginally improve all normal hours.

## 5. Volatility Clustering

Volatility itself clusters.

Absolute one-hour price change distribution:

| Quantile | `abs(price_t - price_t-1)` |
|---|---:|
| p50 | `60.00` |
| p75 | `252.17` |
| p90 | `550.01` |
| p95 | `799.99` |

Define high-volatility hours as top quartile of one-hour absolute changes.

Persistence of high-volatility state:

- base high-vol probability: `25.0%`
- `P(highvol_t | highvol_t-1)`: `47.4%`
- `P(highvol_t | highvol_t-24)`: `52.3%`

So volatile hours cluster both locally and by same-hour daily structure. This makes persistence surprisingly robust even in unstable regimes: yesterday's same hour carries information not only about level, but also about whether that hour tends to be a volatile transition point.

## 6. Why Direct Models Struggle Against It

Persistence is strong because it implicitly encodes multiple latent variables:

- clock-hour demand shape
- previous day's order-book / fuel-stack regime
- weekday/weekend market rhythm
- recent inflationary price level
- cap/zero band clustering
- volatility state
- hydro/thermal dispatch regime

A direct LSTM or generic model must relearn all of these from noisy inputs. If the model sees imperfect or mistimed features, the persistence baseline already has cleaner information: the realized price from the same market hour yesterday.

The target is also discontinuous. The important errors are not smooth:

- zero-price hours
- cap-price hours
- weekend-to-weekday transitions
- renewable ramps
- outage/fuel scarcity days

Average-loss training tends to smooth these events, while persistence preserves banded market behavior.

## 7. What Persistence Does Not Know

Persistence fails when the market state changes between yesterday and today:

- Sunday to Monday demand transition
- sudden solar/wind ramp change
- large outage or maintenance stack change
- gas marginality shift
- price cap event appearing or disappearing
- public information revision after yesterday's price
- high-renewable zero-pressure event

This is where advanced tree and microstructure models can add value. The h1-h4 ensemble improvement of about `18.4%` over persistence is meaningful because it is concentrated in these transition cases.

## 8. Practical Decomposition

Persistence strength can be decomposed as:

```text
persistence_power =
    hourly_shape_persistence
  + price_band_persistence
  + fuel_regime_persistence
  + weekday_structure
  + volatility_clustering
  - regime_transition_error
```

Where:

- hourly shape explains why `lag24` works across most ordinary days
- price band persistence explains why errors stay bounded inside the same market zone
- fuel regime persistence explains why marginal price regimes carry over
- weekday structure explains stable Tuesday-Friday performance
- volatility clustering explains why unstable hours often recur by clock hour
- regime transition error explains the remaining opportunity for models

## 9. Recommendation

Do not treat persistence as a weak baseline. In this market it is a compact proxy for yesterday's fully-cleared order-book state.

Future evaluation should report model gains by decomposition bucket:

- weekday vs weekend transition
- same-band vs band-transition hours
- low/normal/tight/spike regimes
- low-vol vs high-vol hours
- fuel regime stable vs fuel regime changed
- residual load ramp stable vs shifted

The key research question is not simply “can the model beat persistence?” but:

```text
Can the model identify the hours where yesterday's market state is no longer valid?
```

That framing is more aligned with the actual market mechanics and with the observed strength of the persistence baseline.

# Feature Usefulness By Market Regime

## Scope

This is a regime-conditional diagnostic report. It is not a global feature-importance exercise.

No model is trained. No feature engineering pipeline is changed.

Regimes are assigned from finalized PTF with the current deterministic `v0` bands:

| Regime | Rule |
|---|---:|
| `negative_zero_pressure` | `PTF <= 50` |
| `normal` | `50 < PTF < 1500` |
| `tight` | `1500 <= PTF < 4000` |
| `spike_cap` | `PTF >= 4000` |

Rows by regime:

| Regime | Rows |
|---|---:|
| `tight` | `29,878` |
| `normal` | `23,699` |
| `spike_cap` | `1,704` |
| `negative_zero_pressure` | `951` |

The goal is to answer: **which feature families become useful in which market states?**

## Executive Summary

Feature usefulness is strongly regime-dependent:

| Feature family | Most useful regime | Main role |
|---|---|---|
| wind forecast | `negative_zero_pressure`, `tight/spike_cap` boundary | explains oversupply pressure and scarcity relief |
| outages | `spike_cap` as contextual scarcity amplifier, not standalone linear signal | matters when residual load is already high |
| SMF | `normal`, `tight`, `spike_cap` ex-post diagnostic | strong contemporaneous balancing/settlement proxy, not clean ex-ante predictor |
| KGÜP | `tight`, midday `normal`, transition hours | scheduled stack and residual-load pressure |

The important point: a feature can be weak globally and still be useful in one regime. Conversely, a feature can look strong globally because it is proxying the regime label rather than explaining within-regime variation.

## Wind Forecast

Wind forecast is not uniformly useful. It acts differently by price state.

### Negative / Zero Pressure

Within `negative_zero_pressure`:

- `wind_forecast` vs price correlation: `-0.125`
- `wind_forecast_share` vs price correlation: `-0.062`
- `wind_forecast_error_abs` vs price correlation: `-0.263`
- high minus low wind forecast price difference: about `-5.5 TL/MWh`

Interpretation:

- Once the market is already near zero, price has little room to fall.
- Wind is useful less as a continuous price predictor and more as a **regime trigger**: high wind/renewable pressure helps identify entry into zero-pressure hours.
- Inside the zero regime, marginal variation in price is compressed, so correlations look small.

Best use:

- binary/threshold signal for zero-pressure risk
- interaction with solar/load forecast
- interaction with same-hour previous-day low-price state

Avoid:

- treating wind forecast as a globally linear price driver.

### Normal Regime

Within `normal`:

- `wind_forecast` vs price correlation: `0.271`
- `wind_forecast_share` vs price correlation: `0.232`
- high minus low wind forecast price difference: about `+301 TL/MWh`

This positive sign is not physically “wind raises price.” It is likely confounding:

- higher wind can occur in seasons/hours with higher load or different price level
- normal regime excludes the zero-pressure and spike tails
- absolute wind level may proxy weather regime, not just supply relief

Best use:

- wind should be normalized against load: `wind_forecast_share`
- use residual-load features rather than raw wind alone

### Tight And Spike/Cap

Within `tight`:

- `wind_forecast_share` vs price correlation: `-0.141`
- high minus low wind-share price difference: about `-227 TL/MWh`

Within `spike_cap`:

- `wind_forecast_share` vs price correlation: `-0.058`
- high minus low wind-share price difference: about `-54 TL/MWh`

Interpretation:

- In tight hours, wind becomes a scarcity relief variable.
- In spike/cap hours, prices are saturated; wind can help explain whether the hour escapes cap or stays at cap, but intra-cap price variation is limited.

Best use:

- tight-regime relief signal
- cap-entry / cap-exit transition signal
- residual load: `load_forecast - wind_forecast - solar/KGÜP güneş`

## Outages

Outages are not a clean global linear feature. They are a scarcity-context feature.

The full-history outage proxy needs careful handling because outage rows include long planned maintenance intervals and many fields are not true realized MW loss. For the 2026 window, using active `operatorPower` as a maintenance/outage capacity proxy gives:

| Regime | Mean active outage operator power |
|---|---:|
| `negative_zero_pressure` | `16,199 MW` |
| `normal` | `13,921 MW` |
| `spike_cap` | `15,436 MW` |
| `tight` | `9,445 MW` |

This confirms a key caution: **outages alone do not imply high price.** Planned maintenance is seasonal and can coexist with low-load or high-renewable periods.

Within-regime correlations in 2026 are not strongly positive:

- `normal`: outage proxy vs price correlation around `-0.338`
- `tight`: near `0`
- `spike_cap`: negative inside the saturated cap band

This does not mean outages are useless. It means outages are conditional:

```text
outage usefulness = high when residual load is high and flexible capacity is scarce
outage usefulness = weak or misleading when load is low or renewables are high
```

Best use:

- interaction with residual load
- interaction with evening ramp
- fuel-specific outages: gas/coal/hydro
- cap-risk classifier, not level regression

Most useful regimes:

- `spike_cap`: explains why the stack is vulnerable
- `tight`: helps distinguish ordinary high prices from scarcity risk

Least useful as standalone:

- `negative_zero_pressure`, because oversupply can dominate even with high maintenance
- broad `normal`, because planned maintenance seasonality confounds the sign

Recommended derived diagnostics later:

- `active_gas_maintenance_mw`
- `active_coal_maintenance_mw`
- `active_hydro_maintenance_mw`
- `outage_mw / load_forecast`
- `outage_mw * residual_load`
- `outage_mw * evening_ramp_flag`

## SMF

SMF is highly informative inside most regimes, but it is mostly an ex-post / contemporaneous balancing signal.

Within-regime correlations:

| Regime | `systemMarginalPrice` vs PTF | Notes |
|---|---:|---|
| `negative_zero_pressure` | `0.007` | weak; price compressed near zero |
| `normal` | `0.654` | strong |
| `tight` | `0.649` | strong |
| `spike_cap` | `0.479` | still meaningful despite cap saturation |

High vs low SMF price difference:

| Regime | High SMF minus low SMF PTF |
|---|---:|
| `normal` | `+742 TL/MWh` |
| `tight` | `+1116 TL/MWh` |
| `spike_cap` | `+561 TL/MWh` |

Interpretation:

- SMF is very useful for explaining balancing-market stress.
- In `normal` and `tight`, SMF carries information about marginal system conditions.
- In `spike_cap`, PTF is partly saturated, so SMF still moves but cannot fully explain cap-bound price variation.
- In `negative_zero_pressure`, SMF is not a strong price-level signal because PTF is compressed.

Important caveat:

- Same-hour SMF is not safely available for day-ahead PTF prediction.
- Treat it as explanatory/diagnostic or for post-market simulation.
- For predictive use, use lagged SMF, SMF volatility, previous-day SMF/PTF spread, or regime history only.

Best use:

- `normal`: balancing stress overlay
- `tight`: marginal scarcity stress
- `spike_cap`: cap confirmation and SMF/PTF decoupling analysis

## KGÜP

KGÜP is the most regime-conditional core feature family because it approximates the scheduled supply stack.

### Tight Regime

Within `tight`:

| Feature | Corr with PTF | High-low price difference |
|---|---:|---:|
| `kgup_thermal` | `0.476` | `+785 TL/MWh` |
| `dogalgaz` | `0.465` | `+749 TL/MWh` |
| `kgup_gas_share` | `0.403` | `+636 TL/MWh` |
| `toplam` | `0.353` | `+540 TL/MWh` |
| `forecast_residual_load` | `0.278` | `+441 TL/MWh` |

Interpretation:

- In tight hours, gas/thermal scheduling becomes a strong marginality signal.
- High gas share means the market is closer to expensive flexible generation.
- Residual load is useful, but stack composition is even more useful inside tight hours.

Best use:

- gas share
- thermal stack
- residual load
- evening ramp interaction

### Normal Regime

Within `normal`:

| Feature | Corr with PTF | High-low price difference |
|---|---:|---:|
| `kgup_vre` | `0.252` | `+374 TL/MWh` |
| `load_minus_kgup` | `0.244` | `+228 TL/MWh` |
| `kgup_gas_share` | `-0.157` | `-63 TL/MWh` |

The positive `kgup_vre` sign in normal hours is probably confounded by season/hour effects. The more robust normal-regime signal is:

- `load_minus_kgup`
- residual gap
- hour-specific KGÜP mismatch

### Negative / Zero Pressure

Within `negative_zero_pressure`:

- `toplam` vs price correlation: `-0.396`
- `kgup_vre` vs price correlation: `-0.381`
- `kgup_gas_share` vs price correlation: `0.305`

Interpretation:

- High scheduled supply and high VRE push toward zero.
- Gas share rising inside this regime often means prices are less deeply zero-pressure.

Best use:

- zero-pressure classifier
- VRE/load ratio
- oversupply flags

### Spike/Cap

Within `spike_cap`, many KGÜP level correlations weaken:

- `toplam` vs price correlation: `-0.164`
- `forecast_residual_load` vs price correlation: `-0.090`
- `kgup_thermal` vs price correlation: `0.031`

This is expected. Once the market is at cap, the target is saturated. KGÜP is more useful for predicting **entry into cap** than for explaining small variation inside cap.

Best use:

- cap-entry classifier
- transition from `tight` to `spike_cap`
- interaction with outages and low VRE

## KGÜP Hour-Of-Day Usefulness

`load_minus_kgup` is most useful in daytime/midday hours:

| Hour | Corr(`load_minus_kgup`, PTF) |
|---:|---:|
| 08 | `0.261` |
| 09 | `0.304` |
| 10 | `0.311` |
| 11 | `0.305` |
| 12 | `0.318` |
| 13 | `0.288` |
| 14 | `0.280` |
| 15 | `0.258` |

It is weaker overnight and late evening:

- 00-06 mostly below `0.10`
- 21-23 fades toward `0.09`, `0.04`, `0.02`

Interpretation:

- Midday KGÜP/load mismatch captures renewable and demand-shape pressure.
- Evening spike hours need more than `load_minus_kgup`: residual ramp, solar cliff, active outage stack, and gas/thermal marginality.

## Regime-Conditional Usefulness Matrix

| Feature family | Negative/zero pressure | Normal | Tight | Spike/cap |
|---|---|---|---|---|
| Wind forecast | strong as zero-risk trigger | confounded; normalize by load | useful as scarcity relief | useful for cap-exit / relief, not level |
| Outages | weak standalone | confounded by maintenance season | useful with residual load | strong context for cap vulnerability |
| SMF | weak level signal | strong diagnostic | strong diagnostic | cap/SMF decoupling diagnostic |
| KGÜP | strong oversupply signal | useful via load gap | strongest via gas/thermal stack | useful for cap-entry, weak inside cap |

## Recommended Next Diagnostics

Do not build a global feature-importance report. Instead produce these slices:

1. `same_band` vs `band_transition`
2. `tight -> spike_cap` transitions
3. `normal -> negative_zero_pressure` transitions
4. hour blocks: night, morning ramp, solar window, evening ramp
5. fuel regime stable vs fuel regime changed
6. high residual load plus high outage vs high residual load without outage

The most promising predictive framing is:

```text
feature usefulness = ability to detect regime transition,
not ability to explain average PTF across all regimes
```

This is why KGÜP and wind can appear weak globally but become critical around zero-pressure and tight/spike transitions.

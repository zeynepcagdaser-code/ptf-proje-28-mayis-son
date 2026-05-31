# Regime-Aware PTF Prototype Plan

## Scope

This step implements only:

1. `build_regime_labels.py`
2. `audit_persistence_failures.py`
3. `reports/regime_model_plan.md`
4. `reports/regime_model_plan.json`

This step explicitly does **not** implement:

- training
- feature store
- regime classifier
- price experts
- new ensemble
- hyperparameter tuning

## Target Policy

Finalized PTF is allowed only for:

- target regime labels
- transition labels
- persistence-error evaluation

It is not allowed as a downstream feature.

Regime labels:

| Regime | Rule |
|---|---:|
| `negative_zero_pressure` | `price <= 50` |
| `normal` | `50 < price < 1500` |
| `tight` | `1500 <= price < 4000` |
| `spike_cap` | `price >= 4000` |

Required derived labels:

```text
transition_label = lag24_regime -> target_regime
persistence_error = abs(price - price_lag_24)
```

## Script 1: `build_regime_labels.py`

Purpose:

- Read `data/ptf_dataset.csv`.
- Create finalized-PTF regime labels.
- Create lag24 persistence evaluation fields.
- Save labels for later leakage-safe feature-store joining.

Outputs:

- `data/regime_labels.csv`
- `reports/regime_label_summary.md`
- `reports/regime_label_summary.json`

Columns:

- `ts_hour`
- `price`
- `target_regime`
- `price_lag_24`
- `lag24_regime`
- `transition_label`
- `persistence_error`
- `hour`
- `weekday`
- `is_weekend`

Leakage note:

- `price`, `target_regime`, `transition_label`, and `persistence_error` are target/evaluation columns.
- They must not enter the future model feature matrix.

## Script 2: `audit_persistence_failures.py`

Purpose:

- Use `data/regime_labels.csv`.
- Attach diagnostic public context from load forecast and KGÜP.
- Compute where lag24 persistence fails most.
- Produce alpha map before feature-store/model work begins.

Outputs:

- `reports/persistence_failure_alpha_map.md`
- `reports/persistence_failure_alpha_map.json`

Required audit slices:

- worst hours by persistence MAE
- regime-wise MAE
- transition-wise MAE
- `hour x regime` MAE
- residual-load-bin MAE
- `load_forecast - KGÜP` gap-bin MAE
- 2026 active maintenance/outage proxy x residual-load MAE

Interpretation target:

```text
Where is yesterday's same-hour market regime no longer valid?
```

## H1-H4 vs Full H24 Policy

The project has two different meanings that must stay separate:

1. **H1-H4 model horizon**
   - Forecast target hours 1-4 ahead of an anchor timestamp.
   - This matters for future model evaluation.
   - Requires explicit anchor-time feature construction.

2. **Full H24 delivery curve**
   - All delivery hours 0-23 of a market day.
   - This matters for day-ahead market simulation and regime coverage.

In this current step:

- `build_regime_labels.py` labels all hourly finalized PTF rows.
- `audit_persistence_failures.py` reports full H24.
- Any H1-H4 number is diagnostic only unless built from explicit forecast anchors.

Future training must not confuse delivery hour with forecast horizon.

## Leakage Guards

Forbidden as features:

- same-hour finalized PTF
- same-hour realized SMF/YAL/YAT
- historical `interim-mcp` oracle data
- target columns from `data/regime_labels.csv`
- finalized target leakage of any form

Allowed later, with timing audit:

- lagged PTF
- calendar/hour/weekend
- KGÜP stack
- load forecast
- wind/solar forecast or KGÜP renewables
- residual load forecast
- active maintenance proxy
- lagged SMF/YAL/YAT
- true point-in-time K.PTF snapshots

## Current Step Completion Criteria

This step is complete when:

- `build_regime_labels.py` compiles and runs.
- `audit_persistence_failures.py` compiles and runs.
- `data/regime_labels.csv` exists.
- `reports/persistence_failure_alpha_map.md/json` exists.
- `reports/regime_model_plan.md/json` exists.
- No training artifacts are created.
- No feature store is created.

## Next Step

After this step, move to:

```text
build_regime_feature_store.py
```

Only after feature availability and leakage checks pass should the first model be trained:

```text
train_regime_classifier.py
```

# Persistence Failure Alpha Map

Generated: `2026-05-31T12:47:51.413416+00:00`

Evaluation-only report. No model is trained here.

Finalized PTF is used only for regime labels and persistence-error evaluation.

## H1-H4 vs Full H24

| Slice | Rows | Persistence MAE | Median error |
|---|---:|---:|---:|
| H1-H4 diagnostic slice | 9368 | 262.50 | 109.61 |
| Full H24 | 56208 | 294.87 | 100.00 |

H1-H4 here is a delivery-hour diagnostic slice. Later model evaluation must define horizon relative to anchor time separately.

## Worst Transitions

| Transition | Rows | MAE | Median |
|---|---:|---:|---:|
| `normal -> spike_cap` | 38 | 3612.35 | 3724.55 |
| `spike_cap -> normal` | 24 | 3512.63 | 3397.68 |
| `negative_zero_pressure -> tight` | 106 | 2126.93 | 2076.15 |
| `tight -> negative_zero_pressure` | 37 | 1799.12 | 1639.99 |
| `tight -> spike_cap` | 500 | 1128.49 | 1033.99 |
| `normal -> tight` | 2486 | 1125.14 | 1000.27 |
| `tight -> normal` | 2565 | 1017.25 | 927.01 |
| `spike_cap -> tight` | 511 | 1013.94 | 894.49 |
| `normal -> negative_zero_pressure` | 482 | 383.01 | 196.99 |
| `negative_zero_pressure -> normal` | 401 | 342.48 | 200.00 |
| `tight -> tight` | 26775 | 269.88 | 156.05 |
| `spike_cap -> spike_cap` | 1166 | 152.37 | 48.62 |

## Regime MAE

| Regime | Rows | MAE | Median |
|---|---:|---:|---:|
| `spike_cap` | 1704 | 515.95 | 204.97 |
| `tight` | 29878 | 360.36 | 198.79 |
| `negative_zero_pressure` | 951 | 267.41 | 90.00 |
| `normal` | 23675 | 197.41 | 30.40 |

## Hour x Regime Hotspots

| Hour | Regime | Rows | MAE | Median |
|---:|---|---:|---:|---:|
| 11 | `spike_cap` | 91 | 699.82 | 383.68 |
| 15 | `spike_cap` | 96 | 625.94 | 214.34 |
| 14 | `spike_cap` | 82 | 610.23 | 357.00 |
| 8 | `spike_cap` | 116 | 605.78 | 99.99 |
| 21 | `spike_cap` | 127 | 570.14 | 306.51 |
| 10 | `tight` | 1118 | 533.27 | 303.16 |
| 10 | `spike_cap` | 105 | 531.31 | 127.81 |
| 11 | `tight` | 1132 | 529.78 | 300.00 |
| 9 | `spike_cap` | 127 | 525.84 | 55.57 |
| 13 | `tight` | 1062 | 519.09 | 346.02 |
| 15 | `tight` | 1128 | 501.01 | 307.00 |
| 14 | `tight` | 1146 | 500.12 | 324.01 |
| 12 | `tight` | 906 | 491.11 | 320.00 |
| 9 | `tight` | 1188 | 490.57 | 210.99 |
| 8 | `tight` | 1222 | 455.44 | 169.81 |
| 16 | `spike_cap` | 119 | 444.30 | 49.98 |
| 19 | `spike_cap` | 155 | 443.62 | 129.89 |
| 20 | `spike_cap` | 159 | 422.84 | 149.99 |
| 0 | `spike_cap` | 59 | 419.94 | 230.00 |
| 7 | `tight` | 1154 | 405.14 | 279.69 |

## Residual Load x Outage Stress

The outage proxy is limited to 2026 active operator-power maintenance/outage windows.

| Outage stress | Residual load | Rows | MAE | Median |
|---|---|---:|---:|---:|
| `Q3` | `Q5_high` | 75 | 1058.04 | 790.01 |
| `Q4_high` | `Q5_high` | 74 | 953.59 | 520.99 |
| `Q3` | `Q4` | 145 | 921.43 | 717.00 |
| `Q4_high` | `Q4` | 116 | 856.36 | 500.01 |
| `Q1_low` | `Q1_low` | 219 | 836.19 | 684.00 |
| `Q1_low` | `Q2` | 170 | 695.58 | 500.01 |
| `Q1_low` | `Q3` | 177 | 692.89 | 545.48 |
| `Q1_low` | `Q4` | 165 | 634.78 | 393.98 |
| `Q2` | `Q1_low` | 164 | 626.26 | 460.00 |
| `Q4_high` | `Q3` | 186 | 593.82 | 204.50 |
| `Q2` | `Q2` | 181 | 591.37 | 458.47 |
| `Q2` | `Q4` | 173 | 513.71 | 349.00 |

## Alpha Map

The highest-alpha slices are not average hours. They are regime transitions where lag24 carries the wrong market state:

- `normal -> spike_cap`
- `spike_cap -> normal`
- `negative_zero_pressure -> tight`
- `tight -> negative_zero_pressure`
- `tight -> spike_cap`
- `normal -> tight`

## Leakage Policy

- Finalized PTF is allowed here only for labels and evaluation.
- `transition_label` is exactly `lag24_regime -> target_regime`.
- `persistence_error = abs(price - price_lag_24)`.
- Do not feed target/evaluation columns into the future feature store.

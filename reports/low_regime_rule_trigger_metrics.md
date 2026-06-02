# Low Regime Rule Trigger Metrics (Test)

- **Generated (UTC):** 2026-06-01T20:45:11.867150+00:00
- **Test anchors:** 3387
- **Signals CSV:** `data/predictions/low_regime_rule_signals.csv`

## Hybrid recommendation

**Use `balanced_rule`** for hybrid gating.

Recommended for hybrid: balanced_rule — strong low/zero recall from PTF regime history, materially lower false alarm rate than aggressive_rule, and more stable precision than renewable_rule alone.

## Overall metrics (stacked horizons)

| Trigger | low recall | zero recall | precision | F1 | FPR | alarm rate |
|---------|----------:|------------:|----------:|---:|----:|-----------:|
| balanced_rule | 0.8838 | 0.8820 | 0.2011 | 0.3277 | 0.4193 | 0.4689 |
| aggressive_rule | 0.9726 | 0.9714 | 0.1567 | 0.2699 | 0.6252 | 0.6622 |
| renewable_rule | 0.9435 | 0.9436 | 0.1639 | 0.2793 | 0.5748 | 0.6141 |

## Per-horizon low recall

| Trigger | h1 | h6 | h12 | h18 | h24 |
|---------|---:|---:|----:|----:|----:|
| balanced_rule | 0.961 | 0.893 | 0.876 | 0.861 | 0.866 |
| aggressive_rule | 1.000 | 1.000 | 0.967 | 0.934 | 0.978 |
| renewable_rule | 1.000 | 0.994 | 0.931 | 0.866 | 0.967 |

## Ranked triggers

- **aggressive_rule** score=0.6492 (low_rec=0.973, zero_rec=0.971, prec=0.157, alarm=0.662)
- **renewable_rule** score=0.6358 (low_rec=0.944, zero_rec=0.944, prec=0.164, alarm=0.614)
- **balanced_rule** score=0.6163 (low_rec=0.884, zero_rec=0.882, prec=0.201, alarm=0.469)

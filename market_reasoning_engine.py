#!/usr/bin/env python3
"""
Rule-based analyst reasoning features for regime-aware PTF research.

No model is trained. The engine converts leakage-safe feature-store columns into
interpretable regime/stress scores.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_STORE_PATH = PROJECT_ROOT / "data" / "features" / "regime_feature_store.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "market_reasoning_features.parquet"
DESIGN_JSON = PROJECT_ROOT / "reports" / "market_reasoning_engine_design.json"
DESIGN_MD = PROJECT_ROOT / "reports" / "market_reasoning_engine_design.md"


def robust_score(series: pd.Series, high_is_risky: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    q10 = numeric.quantile(0.10)
    q90 = numeric.quantile(0.90)
    if pd.isna(q10) or pd.isna(q90) or q90 == q10:
        return pd.Series(0.0, index=series.index)
    score = (numeric - q10) / (q90 - q10)
    if not high_is_risky:
        score = 1 - score
    return (score.clip(0, 1) * 100).fillna(0)


def weighted_sum(parts: list[tuple[pd.Series, float]]) -> pd.Series:
    total_weight = sum(weight for _, weight in parts)
    if total_weight <= 0:
        raise ValueError("Total weight must be positive.")
    out = sum(series.fillna(0) * weight for series, weight in parts) / total_weight
    return out.clip(0, 100)


def expected_regime(row: pd.Series) -> str:
    scores = {
        "negative_zero_pressure": row["analyst_zero_score"],
        "spike_cap": row["analyst_spike_score"],
        "tight": row["analyst_tight_score"],
    }
    best = max(scores, key=scores.get)
    if scores[best] < 45:
        return "normal"
    return best


def confidence_score(row: pd.Series) -> float:
    scores = sorted(
        [
            row["analyst_zero_score"],
            row["analyst_spike_score"],
            row["analyst_tight_score"],
        ],
        reverse=True,
    )
    separation = scores[0] - scores[1]
    confidence = 35 + 0.45 * scores[0] + 0.20 * separation
    if row["analyst_persistence_break_score"] > 70:
        confidence -= 8
    return float(np.clip(confidence, 0, 100))


def top_reasons(row: pd.Series) -> str:
    reasons: list[str] = []
    if row.get("zero_price_pressure_score", 0) >= 60:
        reasons.append("zero pressure: low demand, gas-off, and renewable oversupply")
    if row["analyst_zero_score"] >= 60:
        reasons.append("zero pressure: renewable oversupply / weak residual load")
    if row["analyst_spike_score"] >= 60:
        reasons.append("spike risk: high residual load, ramp, or maintenance stress")
    if row["analyst_tight_score"] >= 60:
        reasons.append("tight market: thermal/gas stack and residual load elevated")
    if row.get("gas_marginality_proxy", 0) >= 60:
        reasons.append("gas marginality elevated")
    if row.get("hydro_displacement_score", 0) >= 60:
        reasons.append("hydro displacement visible")
    if row.get("demand_weakness_score", 0) >= 60:
        reasons.append("demand weakness visible")
    if row["solar_cliff_score"] >= 1:
        reasons.append("solar cliff visible")
    if row["wind_relief_score"] >= 0.12:
        reasons.append("wind relief available")
    if row["outage_stress_index"] >= 0.35:
        reasons.append("maintenance stack elevated")
    if row["price_band_persistence"] == 0:
        reasons.append("lag24 and lag168 price bands disagree")
    if row["volatility_cluster_score"] >= 1:
        reasons.append("volatility cluster active")
    if not reasons:
        reasons.append("normal: no dominant stress signal")
    return "; ".join(reasons[:4])


def build_reasoning_features(features: pd.DataFrame) -> pd.DataFrame:
    f = features.copy()

    def col(name: str, default: float = 0.0) -> pd.Series:
        if name in f.columns:
            return f[name]
        return pd.Series(default, index=f.index)

    residual_high = robust_score(col("residual_load_forecast"))
    residual_low = robust_score(col("residual_load_forecast"), high_is_risky=False)
    residual_ramp = robust_score(col("residual_load_ramp").abs())
    load_gap_high = robust_score(col("load_minus_kgup"))
    load_gap_low = robust_score(col("load_minus_kgup"), high_is_risky=False)
    renewable_pressure = robust_score(col("renewable_oversupply_score"))
    solar_cliff = robust_score(col("solar_cliff_score"))
    wind_relief = robust_score(col("wind_relief_score"))
    wind_absent = robust_score(col("wind_relief_score"), high_is_risky=False)
    maintenance = robust_score(col("outage_stress_index"))
    gas_dependency = robust_score(col("gas_share"))
    hydro_dependency = robust_score(col("hydro_share"))
    thermal_dependency = robust_score(col("thermal_share"))
    gas_marginality = robust_score(col("gas_marginality_proxy"))
    hydro_displacement = robust_score(col("hydro_displacement_score"))
    demand_weakness = robust_score(col("demand_weakness_score"))
    zero_pressure = robust_score(col("zero_price_pressure_score"))
    low_demand_flag = col("low_demand_flag").fillna(0) * 100
    gas_off_flag = col("gas_off_flag").fillna(0) * 100
    renewable_share_high_flag = col("renewable_share_high_flag").fillna(0) * 100
    hydro_high_flag = col("hydro_high_flag").fillna(0) * 100
    volatility = robust_score(col("volatility_cluster_score"))
    persistence_unreliable = (1 - col("price_band_persistence").fillna(1)).clip(0, 1) * 100
    evening = col("evening_ramp_flag").fillna(0) * 100
    sunset = col("sunset_window_flag").fillna(0) * 100

    out = pd.DataFrame({"ts_hour": f["ts_hour"]})
    out["analyst_zero_score"] = weighted_sum(
        [
            (zero_pressure, 0.25),
            (renewable_pressure, 0.18),
            (residual_low, 0.14),
            (load_gap_low, 0.12),
            (wind_relief, 0.10),
            (low_demand_flag, 0.08),
            (gas_off_flag, 0.06),
            (renewable_share_high_flag, 0.04),
            (hydro_high_flag, 0.04),
            (persistence_unreliable, 0.04),
            (volatility, 0.05),
        ]
    )
    out["analyst_spike_score"] = weighted_sum(
        [
            (residual_high, 0.25),
            (residual_ramp, 0.15),
            (solar_cliff, 0.15),
            (maintenance, 0.15),
            (gas_dependency, 0.10),
            (gas_marginality, 0.10),
            (wind_absent, 0.08),
            (evening, 0.07),
            (persistence_unreliable, 0.05),
        ]
    )
    out["analyst_tight_score"] = weighted_sum(
        [
            (residual_high, 0.25),
            (load_gap_high, 0.15),
            (thermal_dependency, 0.15),
            (gas_dependency, 0.15),
            (gas_marginality, 0.10),
            (hydro_dependency, 0.10),
            (maintenance, 0.10),
            (volatility, 0.10),
        ]
    )
    out["analyst_persistence_break_score"] = weighted_sum(
        [
            (persistence_unreliable, 0.25),
            (residual_ramp, 0.20),
            (solar_cliff, 0.15),
            (volatility, 0.15),
            (load_gap_high.combine(load_gap_low, max), 0.15),
            (demand_weakness, 0.05),
            (gas_marginality, 0.05),
            (maintenance, 0.10),
        ]
    )
    out["analyst_expected_regime"] = out.apply(expected_regime, axis=1)
    out["analyst_confidence_score"] = out.apply(confidence_score, axis=1)

    reason_input = pd.concat(
        [
            pd.DataFrame(
                {
                    "solar_cliff_score": col("solar_cliff_score"),
                    "wind_relief_score": col("wind_relief_score"),
                    "outage_stress_index": col("outage_stress_index"),
                    "price_band_persistence": col("price_band_persistence"),
                    "volatility_cluster_score": col("volatility_cluster_score"),
                    "gas_marginality_proxy": col("gas_marginality_proxy"),
                    "hydro_displacement_score": col("hydro_displacement_score"),
                    "zero_price_pressure_score": col("zero_price_pressure_score"),
                    "demand_weakness_score": col("demand_weakness_score"),
                }
            ).reset_index(drop=True),
            out.drop(columns=["ts_hour"]).reset_index(drop=True),
        ],
        axis=1,
    )
    out["analyst_reason_text"] = reason_input.apply(top_reasons, axis=1)
    return out


def build_design_report(reasoning: pd.DataFrame) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(FEATURE_STORE_PATH.relative_to(PROJECT_ROOT)),
        "output": str(OUTPUT_PATH.relative_to(PROJECT_ROOT)),
        "rows": int(len(reasoning)),
        "columns": list(reasoning.columns),
        "score_ranges": {
            column: {
                "min": float(reasoning[column].min()),
                "mean": float(reasoning[column].mean()),
                "max": float(reasoning[column].max()),
            }
            for column in [
                "analyst_zero_score",
                "analyst_spike_score",
                "analyst_tight_score",
                "analyst_persistence_break_score",
                "analyst_confidence_score",
            ]
        },
        "expected_regime_counts": reasoning["analyst_expected_regime"].value_counts().to_dict(),
        "logic": {
            "persistence_reliability": [
                "price_band_persistence",
                "volatility_cluster_score",
                "residual_load_ramp",
                "solar_cliff_score",
                "zero_price_pressure_score",
            ],
            "regime_shift_risk": [
                "normal/tight -> spike through residual load, solar cliff, maintenance, gas dependency",
                "normal/tight -> zero through renewable oversupply and low residual load",
                "gas marginality and hydro displacement shape marginal fuel",
            ],
            "physical_market_context": [
                "residual load level and ramp",
                "solar cliff",
                "wind relief",
                "maintenance stack",
                "gas/hydro/thermal dependency",
                "gas marginality proxy",
                "hydro displacement score",
                "demand weakness score",
                "evening ramp and sunset window",
            ],
        },
        "leakage_policy": {
            "training": False,
            "uses_only_feature_store": True,
            "same_hour_finalized_ptf": False,
            "same_hour_smf_yal_yat": False,
            "historical_interim_oracle": False,
        },
    }


def write_outputs(reasoning: pd.DataFrame, report: dict[str, Any]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DESIGN_JSON.parent.mkdir(parents=True, exist_ok=True)
    from src.utils.safe_io import atomic_parquet_write
    atomic_parquet_write(reasoning, str(OUTPUT_PATH), index=False)
    DESIGN_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "# Market Reasoning Engine Design",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "No training is performed. This is a deterministic analyst-reasoning layer built on the leakage-safe feature store.",
        "",
        f"- Input: `{report['input']}`",
        f"- Output: `{report['output']}`",
        f"- Rows: `{report['rows']}`",
        "",
        "## Produced Columns",
        "",
    ]
    for column in report["columns"]:
        lines.append(f"- `{column}`")
    lines.extend(["", "## Score Ranges", "", "| Score | Min | Mean | Max |", "|---|---:|---:|---:|"])
    for column, stats in report["score_ranges"].items():
        lines.append(f"| `{column}` | {stats['min']:.2f} | {stats['mean']:.2f} | {stats['max']:.2f} |")
    lines.extend(["", "## Expected Regime Counts", "", "| Regime | Rows |", "|---|---:|"])
    for regime, count in report["expected_regime_counts"].items():
        lines.append(f"| `{regime}` | {count} |")
    lines.extend(
        [
            "",
            "## Reasoning Logic",
            "",
            "- `analyst_zero_score`: renewable oversupply, low residual load, low load-KGÜP gap, wind relief.",
            "- `analyst_spike_score`: high residual load, residual ramp, solar cliff, maintenance stress, gas dependency, low wind relief, evening ramp.",
            "- `analyst_tight_score`: residual load, load-KGÜP gap, thermal/gas/hydro dependency, maintenance and volatility.",
            "- `analyst_persistence_break_score`: lagged band disagreement, residual ramp, solar cliff, volatility and maintenance.",
            "",
            "## Leakage Policy",
            "",
            "- Uses only `regime_feature_store.parquet`.",
            "- Does not use finalized target PTF.",
            "- Does not use same-hour realized SMF/YAL/YAT.",
            "- Does not use historical oracle `interim-mcp`.",
        ]
    )
    DESIGN_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    if not FEATURE_STORE_PATH.exists():
        raise FileNotFoundError(
            f"Missing feature store: {FEATURE_STORE_PATH}. Run build_regime_feature_store.py first."
        )
    from src.utils.io_utils import read_parquet_with_normalized_ts
    features = read_parquet_with_normalized_ts(FEATURE_STORE_PATH)
    reasoning = build_reasoning_features(features)
    report = build_design_report(reasoning)
    write_outputs(reasoning, report)
    print(f"Wrote {OUTPUT_PATH} rows={len(reasoning)} columns={len(reasoning.columns)}")
    print(f"Wrote {DESIGN_JSON}")
    print(f"Wrote {DESIGN_MD}")


if __name__ == "__main__":
    main()

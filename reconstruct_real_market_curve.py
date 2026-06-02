#!/usr/bin/env python3
"""
Reconstruct a real DAM supply-demand curve from the smoke raw response.

Uses only the raw EPİAŞ smoke files saved under data/raw/dam_supply_demand_curve_smoke.
No proxy fallback, no training.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "dam_supply_demand_curve_smoke"
SDE_BODY = RAW_DIR / "supply_demand_attempt_01.body.txt"
PTF_BODY = RAW_DIR / "ptf_attempt_01.body.txt"

OUT_PATH = PROJECT_ROOT / "data" / "features" / "reconstructed_market_curve_features.parquet"
REPORT_MD = PROJECT_ROOT / "reports" / "reconstructed_market_curve_analysis.md"
REPORT_JSON = PROJECT_ROOT / "reports" / "reconstructed_market_curve_analysis.json"
DEBUG_PLOT = PROJECT_ROOT / "reports" / "curve_debug_examples" / "reconstructed_curve.png"

SMOKE_TS = "2026-06-01T00:00:00+03:00"


def load_raw_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_raw_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    supply_obj = load_raw_json(SDE_BODY)
    ptf_obj = load_raw_json(PTF_BODY)

    supply_items = supply_obj.get("items", [])
    ptf_items = ptf_obj.get("items", [])
    if not ptf_items:
        nested = ptf_obj.get("body", {})
        if isinstance(nested, dict):
            content = nested.get("content", {})
            if isinstance(content, dict) and isinstance(content.get("response"), dict):
                ptf_items = [content["response"]]

    s_rows = pd.DataFrame(supply_items)
    p_rows = pd.DataFrame(ptf_items)
    return s_rows, p_rows


def build_axis_tables(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    supply = raw[raw.columns.intersection(["amount", "supplyPrice"])].dropna(subset=["amount", "supplyPrice"]).copy()
    demand = raw[raw.columns.intersection(["amount", "demandPrice"])].dropna(subset=["amount", "demandPrice"]).copy()

    supply = supply.rename(columns={"amount": "supply_mwh", "supplyPrice": "price"}).sort_values("price").reset_index(drop=True)
    demand = demand.rename(columns={"amount": "demand_mwh", "demandPrice": "price"}).sort_values("price").reset_index(drop=True)
    return supply, demand


def piecewise_step(x: np.ndarray, xp: np.ndarray, yp: np.ndarray, *, side: str = "left") -> np.ndarray:
    if len(xp) == 0:
        return np.full_like(x, np.nan, dtype=float)
    if side == "left":
        idx = np.searchsorted(xp, x, side="right") - 1
        idx = np.clip(idx, 0, len(yp) - 1)
    else:
        idx = np.searchsorted(xp, x, side="left")
        idx = np.clip(idx, 0, len(yp) - 1)
    return yp[idx]


def find_clearing_price(supply: pd.DataFrame, demand: pd.DataFrame) -> tuple[float, float, dict[str, Any]]:
    if supply.empty or demand.empty:
        raise ValueError("Supply or demand curve is empty.")

    price_axis = np.unique(np.concatenate([supply["price"].to_numpy(float), demand["price"].to_numpy(float)]))
    price_axis.sort()

    supply_price = supply["price"].to_numpy(float)
    supply_qty = supply["supply_mwh"].to_numpy(float)
    demand_price = demand["price"].to_numpy(float)
    demand_qty = demand["demand_mwh"].to_numpy(float)

    s_on_axis = piecewise_step(price_axis, supply_price, supply_qty, side="left")
    d_on_axis = piecewise_step(price_axis, demand_price, demand_qty, side="right")

    diff = s_on_axis - d_on_axis
    valid = np.isfinite(diff)
    if not valid.any():
        raise ValueError("No finite intersection candidates.")

    sign = np.sign(diff)
    crossings = np.where(np.diff(sign) != 0)[0]
    if len(crossings) > 0:
        i = crossings[0]
        x0, x1 = price_axis[i], price_axis[i + 1]
        y0, y1 = diff[i], diff[i + 1]
        if y1 == y0:
            clearing_price = float(x0)
        else:
            clearing_price = float(x0 - y0 * (x1 - x0) / (y1 - y0))
    else:
        # choose nearest point if no explicit crossing
        i = int(np.nanargmin(np.abs(diff)))
        clearing_price = float(price_axis[i])

    clearing_supply = float(np.interp(clearing_price, supply_price, supply_qty, left=supply_qty[0], right=supply_qty[-1]))
    clearing_demand = float(np.interp(clearing_price, demand_price, demand_qty, left=demand_qty[0], right=demand_qty[-1]))
    clearing_volume = float((clearing_supply + clearing_demand) / 2.0)
    meta = {
        "price_axis_points": int(len(price_axis)),
        "crossings_found": int(len(crossings)),
        "supply_points": int(len(supply)),
        "demand_points": int(len(demand)),
        "s_on_axis": s_on_axis,
        "d_on_axis": d_on_axis,
        "price_axis": price_axis,
    }
    return clearing_price, clearing_volume, meta


def compute_features(supply: pd.DataFrame, demand: pd.DataFrame, clearing_price: float, clearing_volume: float, meta: dict[str, Any], mcp_price: float, mcp_volume: float) -> pd.DataFrame:
    price_axis = meta["price_axis"]
    s_on_axis = meta["s_on_axis"]
    d_on_axis = meta["d_on_axis"]

    idx = int(np.nanargmin(np.abs(price_axis - clearing_price)))
    idx_prev = max(idx - 1, 0)
    idx_next = min(idx + 1, len(price_axis) - 1)
    slope_near = float((s_on_axis[idx_next] - s_on_axis[idx_prev]) / max(price_axis[idx_next] - price_axis[idx_prev], 1e-9))
    elasticity_near = float(abs((d_on_axis[idx_next] - d_on_axis[idx_prev]) / max(price_axis[idx_next] - price_axis[idx_prev], 1e-9)))
    fragility = float(np.clip(abs(slope_near) + abs(elasticity_near), 0, 1e6))

    volume_needed_100 = float(abs(np.interp(clearing_price + 100, price_axis, s_on_axis) - np.interp(clearing_price, price_axis, s_on_axis)))
    volume_needed_500 = float(abs(np.interp(clearing_price + 500, price_axis, s_on_axis) - np.interp(clearing_price, price_axis, s_on_axis)))
    oversupply_pressure = float(np.clip((np.interp(clearing_price, price_axis, s_on_axis) - np.interp(clearing_price, price_axis, d_on_axis)) / max(clearing_volume, 1.0), 0, 1))
    cap_risk_score = float(np.clip((clearing_price - 3500) / 800.0, 0, 1))

    out = pd.DataFrame(
        [
            {
                "delivery_hour": pd.to_datetime(SMOKE_TS),
                "clearing_price": clearing_price,
                "clearing_volume": clearing_volume,
                "slope_near_clearing": slope_near,
                "elasticity_near_clearing": elasticity_near,
                "curve_fragility_score": fragility,
                "volume_needed_for_100TL_move": volume_needed_100,
                "volume_needed_for_500TL_move": volume_needed_500,
                "oversupply_pressure": oversupply_pressure,
                "cap_risk_score": cap_risk_score,
                "mcp_price": mcp_price,
                "mcp_volume": mcp_volume,
                "mcp_price_delta": clearing_price - mcp_price,
                "mcp_volume_delta": clearing_volume - mcp_volume,
                "supply_points": int(len(supply)),
                "demand_points": int(len(demand)),
                "price_axis_points": int(meta["price_axis_points"]),
            }
        ]
    )
    return out


def plot_curve(supply: pd.DataFrame, demand: pd.DataFrame, clearing_price: float, clearing_volume: float) -> None:
    DEBUG_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(supply["price"], supply["supply_mwh"], where="post", label="Supply curve", color="#4cc9f0")
    ax.step(demand["price"], demand["demand_mwh"], where="post", label="Demand curve", color="#f94144")
    ax.scatter([clearing_price], [clearing_volume], color="#f9c74f", s=70, zorder=5, label="Clearing intersection")
    ax.set_xlabel("Price")
    ax.set_ylabel("MWh")
    ax.set_title("Reconstructed DAM Supply-Demand Curve Smoke")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(DEBUG_PLOT, dpi=180)
    plt.close(fig)


def main() -> None:
    if not SDE_BODY.exists() or not PTF_BODY.exists():
        raise FileNotFoundError("Smoke raw files not found. Run fetch_dam_supply_demand_curve.py first.")

    raw_supply, raw_ptf = extract_raw_tables()
    supply, demand = build_axis_tables(raw_supply)

    if supply.empty or demand.empty:
        raise RuntimeError("Could not build both supply and demand curves from the raw response.")

    clearing_price, clearing_volume, meta = find_clearing_price(supply, demand)
    ptf_row = raw_ptf.iloc[0] if not raw_ptf.empty else {}
    mcp_price = float(ptf_row.get("mcpPrice", np.nan))
    mcp_volume = float(ptf_row.get("matchingQuantity", np.nan))

    features = compute_features(supply, demand, clearing_price, clearing_volume, meta, mcp_price, mcp_volume)
    from src.utils.safe_io import atomic_parquet_write
    atomic_parquet_write(features, str(OUT_PATH), index=False)
    plot_curve(supply, demand, clearing_price, clearing_volume)

    analysis = {
        "available": True,
        "mode": "raw_smoke",
        "delivery_hour": SMOKE_TS,
        "supply_points": int(len(supply)),
        "demand_points": int(len(demand)),
        "price_axis_points": int(meta["price_axis_points"]),
        "crossings_found": int(meta["crossings_found"]),
        "clearing_price": clearing_price,
        "clearing_volume": clearing_volume,
        "mcp_price": mcp_price,
        "mcp_volume": mcp_volume,
        "mcp_price_delta": float(clearing_price - mcp_price),
        "mcp_volume_delta": float(clearing_volume - mcp_volume),
        "feature_means": features.iloc[0].to_dict(),
        "notes": [
            "Real smoke raw curve body used directly.",
            "No proxy fallback.",
            "Clearing intersection is derived from the reconstructed supply and demand step curves.",
        ],
    }

    REPORT_JSON.write_text(json.dumps(analysis, ensure_ascii=False, indent=2, default=str) + "\n")
    REPORT_MD.write_text(
        "\n".join(
            [
                "# Reconstructed Market Curve Analysis",
                "",
                f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
                "",
                "## Smoke Summary",
                "",
                f"- Supply points: `{analysis['supply_points']}`",
                f"- Demand points: `{analysis['demand_points']}`",
                f"- Price axis points: `{analysis['price_axis_points']}`",
                f"- Crossings found: `{analysis['crossings_found']}`",
                "",
                "## Clearing Validation",
                "",
                f"- Clearing price: `{analysis['clearing_price']}`",
                f"- Clearing volume: `{analysis['clearing_volume']}`",
                f"- PTF mcpPrice: `{analysis['mcp_price']}`",
                f"- PTF matchingQuantity: `{analysis['mcp_volume']}`",
                f"- Price delta vs PTF: `{analysis['mcp_price_delta']}`",
                f"- Volume delta vs PTF: `{analysis['mcp_volume_delta']}`",
                "",
                "## Feature Extracts",
                "",
            ]
            + [f"- `{k}`: `{v}`" for k, v in features.iloc[0].to_dict().items()]
            + [
                "",
                "## Notes",
                "",
                "This is a direct smoke reconstruction from the raw EPİAŞ GÖP curve response. No proxy fallback was used.",
                f"Debug plot: `{DEBUG_PLOT.relative_to(PROJECT_ROOT)}`",
            ]
        )
        + "\n"
    )

    print(f"Wrote {OUT_PATH}")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {DEBUG_PLOT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Final prediction combiner (backtest-style, no training).

User requirements (June 2026):
  1) Ensure spike classifier exists (models/spike_classifier.pkl) (produced by train_spike_classifier.py).
  2) Rewrite prediction logic:
     - h1-h4: ONLY microstructure_h1h4_predictions.csv
     - h5-h12: ONLY microstructure_h5h12_predictions.csv OR tree_test_predictions.csv
     - Report MAE separately:
         * h1-h4 MAE
         * h5-h12 MAE
         * overall h1-h12 MAE
  3) Baseline comparisons for h1-h4:
     - microstructure alone MAE
     - advanced tree alone MAE
     - ensemble MAE (optimized weights)
  4) reports/final_metrics.json must include horizon-group MAEs.

Outputs:
  - data/predictions/final_predictions.csv
  - reports/final_metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

# Inputs
ADV_TREE_PREDS = PROJECT_ROOT / "data" / "predictions" / "tree_test_predictions.csv"
MICRO_H1H4_PREDS = PROJECT_ROOT / "data" / "predictions" / "microstructure_h1h4_predictions.csv"
MICRO_H5H12_PREDS = PROJECT_ROOT / "data" / "predictions" / "microstructure_h5h12_predictions.csv"
ANCHOR_TEST_PATH = PROJECT_ROOT / "data" / "model" / "anchor_test.csv"

SPIKE_CLASSIFIER_PKL = PROJECT_ROOT / "models" / "spike_classifier.pkl"
SPIKE_PROBABILITY_CSV = PROJECT_ROOT / "data" / "predictions" / "spike_probability.csv"

H1H4_OPT_WEIGHTS_JSON = PROJECT_ROOT / "reports" / "h1h4_optimized_weights.json"
H5H12_VALID_WEIGHT_METRICS_JSON = PROJECT_ROOT / "reports" / "h5h12_validation_weighted_ensemble_metrics.json"

# Outputs
OUT_CSV = PROJECT_ROOT / "data" / "predictions" / "final_predictions.csv"
OUT_METRICS = PROJECT_ROOT / "reports" / "final_metrics.json"

H1H4 = [1, 2, 3, 4]
H5H12 = list(range(5, 13))
H_ALL = H1H4 + H5H12


def mae(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    return float(np.mean(np.abs(a - b)))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def load_anchor_filter() -> pd.DataFrame | None:
    if not ANCHOR_TEST_PATH.exists():
        return None
    anchors = pd.read_csv(ANCHOR_TEST_PATH)
    anchors["anchor_ts_hour"] = pd.to_datetime(anchors["anchor_ts_hour"], utc=True).dt.tz_convert(
        "Europe/Istanbul"
    )
    return anchors[["anchor_ts_hour"]].drop_duplicates()


def load_weights_h1h4() -> tuple[float, float, dict[str, Any]]:
    """
    Returns (w_adv, w_micro, meta). Falls back to (0.7, 0.3).
    """
    meta: dict[str, Any] = {"available": False, "source": "fallback_default"}
    if H1H4_OPT_WEIGHTS_JSON.exists():
        try:
            payload = json.loads(H1H4_OPT_WEIGHTS_JSON.read_text(encoding="utf-8"))
            meta = payload
            if payload.get("available"):
                w = payload.get("optimized_weights") or {}
                w_adv = float(w.get("advanced_tree"))
                w_micro = float(w.get("microstructure"))
                if 0.0 <= w_adv <= 1.0 and 0.0 <= w_micro <= 1.0 and abs((w_adv + w_micro) - 1.0) <= 1e-6:
                    return w_adv, w_micro, meta
        except Exception as e:
            meta = {"available": False, "source": "weights_load_failed", "error": f"{type(e).__name__}: {e}"}
    return 0.7, 0.3, meta


def load_weights_h5h12() -> dict[int, float]:
    """
    Returns per-horizon advanced-tree weight w_adv for h=5..12.
    Default: 1.0 (advanced-only).
    """
    out = {h: 1.0 for h in H5H12}
    if not H5H12_VALID_WEIGHT_METRICS_JSON.exists():
        return out
    try:
        payload = json.loads(H5H12_VALID_WEIGHT_METRICS_JSON.read_text(encoding="utf-8"))
        sel = payload.get("selected_weights_validation") or {}
        for h in H5H12:
            if str(h) in sel:
                w = float(sel[str(h)])
                if 0.0 <= w <= 1.0:
                    out[h] = w
    except Exception:
        return out
    return out


def load_predictions() -> pd.DataFrame:
    if not ADV_TREE_PREDS.exists():
        raise FileNotFoundError(f"Missing {ADV_TREE_PREDS}")
    if not MICRO_H1H4_PREDS.exists():
        raise FileNotFoundError(f"Missing {MICRO_H1H4_PREDS}")
    if not MICRO_H5H12_PREDS.exists():
        raise FileNotFoundError(f"Missing {MICRO_H5H12_PREDS}")

    anchors = load_anchor_filter()

    adv = pd.read_csv(ADV_TREE_PREDS)
    adv = adv[adv["target_hour"].isin(H_ALL)].copy()
    adv["anchor_ts_hour"] = pd.to_datetime(adv["anchor_ts_hour"], utc=True).dt.tz_convert("Europe/Istanbul")
    adv = adv.rename(
        columns={
            "predicted_price": "advanced_tree_pred",
            "persistence_price": "persistence_pred",
            "prob_spike": "advanced_tree_prob_spike",
        }
    )
    adv = adv[
        [
            "anchor_ts_hour",
            "target_hour",
            "actual_price",
            "persistence_pred",
            "advanced_tree_pred",
            "advanced_tree_prob_spike",
        ]
    ]
    if anchors is not None:
        adv = adv.merge(anchors, on="anchor_ts_hour", how="inner")

    micro1 = pd.read_csv(MICRO_H1H4_PREDS)
    micro1 = micro1[micro1["target_hour"].isin(H1H4)].copy()
    micro1["anchor_ts_hour"] = pd.to_datetime(micro1["anchor_ts_hour"], utc=True).dt.tz_convert(
        "Europe/Istanbul"
    )
    micro1 = micro1.rename(columns={"predicted_price": "microstructure_pred"})[
        ["anchor_ts_hour", "target_hour", "microstructure_pred"]
    ]
    if anchors is not None:
        micro1 = micro1.merge(anchors, on="anchor_ts_hour", how="inner")

    micro2 = pd.read_csv(MICRO_H5H12_PREDS)
    micro2 = micro2[micro2["target_hour"].isin(H5H12)].copy()
    micro2["anchor_ts_hour"] = pd.to_datetime(micro2["anchor_ts_hour"], utc=True).dt.tz_convert(
        "Europe/Istanbul"
    )
    micro2 = micro2.rename(columns={"predicted_price": "microstructure_pred"})[
        ["anchor_ts_hour", "target_hour", "microstructure_pred"]
    ]
    if anchors is not None:
        micro2 = micro2.merge(anchors, on="anchor_ts_hour", how="inner")

    micro = pd.concat([micro1, micro2], ignore_index=True)

    merged = adv.merge(micro, on=["anchor_ts_hour", "target_hour"], how="inner")
    merged = merged.sort_values(["anchor_ts_hour", "target_hour"]).reset_index(drop=True)
    return merged


def load_spike_prob(df: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    """
    Use spike_probability.csv if present (generated by train_spike_classifier.py).
    Fallback: advanced_tree_prob_spike from advanced tree predictions.
    """
    if SPIKE_PROBABILITY_CSV.exists():
        sp = pd.read_csv(SPIKE_PROBABILITY_CSV)
        sp["anchor_ts_hour"] = pd.to_datetime(sp["anchor_ts_hour"], utc=True).dt.tz_convert("Europe/Istanbul")
        sp = sp[sp["target_hour"].isin(H_ALL)][["anchor_ts_hour", "target_hour", "spike_prob"]]
        merged = df[["anchor_ts_hour", "target_hour"]].merge(sp, on=["anchor_ts_hour", "target_hour"], how="left")
        return merged["spike_prob"].fillna(0.0).astype(float).clip(0.0, 1.0), {
            "source": "data/predictions/spike_probability.csv",
            "path": str(SPIKE_PROBABILITY_CSV),
        }

    return df["advanced_tree_prob_spike"].astype(float).clip(0.0, 1.0), {
        "source": "advanced_tree_prob_spike_fallback",
        "path": str(ADV_TREE_PREDS),
    }


def compute_metrics(frame: pd.DataFrame, *, pred_col: str) -> dict[str, float]:
    a = frame["actual_price"].to_numpy(dtype=np.float64)
    p = frame[pred_col].to_numpy(dtype=np.float64)
    return {"mae": mae(a, p), "rmse": rmse(a, p), "rows": float(len(frame))}


def main() -> None:
    df = load_predictions()
    spike_prob, spike_meta = load_spike_prob(df)
    df["spike_prob"] = spike_prob

    # Baselines (always computable)
    df["micro_only"] = df["microstructure_pred"]
    df["adv_only"] = df["advanced_tree_pred"]

    # h1-h4 optimized ensemble (baseline comparison only; NOT used for final h1-h4 per requirement)
    w_adv, w_micro, meta_w = load_weights_h1h4()
    df["ensemble_h1h4"] = np.where(
        df["target_hour"].isin(H1H4),
        w_adv * df["advanced_tree_pred"] + w_micro * df["microstructure_pred"],
        np.nan,
    )

    # Final policy (as requested):
    # - h1-h4: micro only
    # - h5-h12: choose micro OR advanced based on validation-per-horizon weights (currently often advanced-only)
    w_adv_h5h12 = load_weights_h5h12()
    df["final_pred"] = df["microstructure_pred"]  # default micro
    df["final_source"] = "microstructure"
    mask_h5h12 = df["target_hour"].isin(H5H12)
    if mask_h5h12.any():
        # If validation says w_adv==1.0, use advanced tree. If w_adv==0.0, micro. Else weight.
        def final_for_row(h: int, adv: float, micro: float) -> tuple[float, str]:
            w = float(w_adv_h5h12.get(int(h), 1.0))
            if w >= 0.999:
                return adv, "advanced_tree"
            if w <= 0.001:
                return micro, "microstructure"
            return w * adv + (1.0 - w) * micro, f"weighted_{w:.2f}_{1.0-w:.2f}"

        sub = df.loc[mask_h5h12, ["target_hour", "advanced_tree_pred", "microstructure_pred"]]
        vals = [final_for_row(int(r.target_hour), float(r.advanced_tree_pred), float(r.microstructure_pred)) for r in sub.itertuples(index=False)]
        df.loc[mask_h5h12, "final_pred"] = [v for v, _ in vals]
        df.loc[mask_h5h12, "final_source"] = [s for _, s in vals]

    # Group metrics
    g_h1h4 = df[df["target_hour"].isin(H1H4)].copy()
    g_h5h12 = df[df["target_hour"].isin(H5H12)].copy()

    metrics: dict[str, Any] = {
        "rows": int(len(df)),
        "spike_prob_source": spike_meta,
        "h1h4_optimized_weights": {"w_adv": w_adv, "w_micro": w_micro, "meta": meta_w, "path": str(H1H4_OPT_WEIGHTS_JSON)},
        "h5h12_per_horizon_w_adv": {"weights": w_adv_h5h12, "path": str(H5H12_VALID_WEIGHT_METRICS_JSON)},
        "mae_groups": {
            "h1_h4": compute_metrics(g_h1h4, pred_col="final_pred"),
            "h5_h12": compute_metrics(g_h5h12, pred_col="final_pred"),
            "h1_h12_overall": compute_metrics(df, pred_col="final_pred"),
        },
        "h1h4_baselines": {
            "micro_only": compute_metrics(g_h1h4, pred_col="micro_only"),
            "advanced_tree_only": compute_metrics(g_h1h4, pred_col="adv_only"),
            "optimized_ensemble": compute_metrics(g_h1h4.dropna(subset=["ensemble_h1h4"]), pred_col="ensemble_h1h4"),
        },
    }

    # Save outputs
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_cols = [
        "anchor_ts_hour",
        "target_hour",
        "actual_price",
        "persistence_pred",
        "microstructure_pred",
        "advanced_tree_pred",
        "spike_prob",
        "final_pred",
        "final_source",
    ]
    df[out_cols].to_csv(OUT_CSV, index=False)

    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    OUT_METRICS.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    # Print concise summary to terminal
    print(f"Wrote: {OUT_CSV} ({len(df)} rows)")
    print(f"Wrote: {OUT_METRICS}")
    print(f"h1-h4 MAE: {metrics['mae_groups']['h1_h4']['mae']:.2f}")
    print(f"h5-h12 MAE: {metrics['mae_groups']['h5_h12']['mae']:.2f}")
    print(f"overall MAE: {metrics['mae_groups']['h1_h12_overall']['mae']:.2f}")
    print("h1-h4 baselines:")
    b = metrics["h1h4_baselines"]
    print(f"  micro_only MAE: {b['micro_only']['mae']:.2f}")
    print(f"  adv_only   MAE: {b['advanced_tree_only']['mae']:.2f}")
    print(f"  ensemble   MAE: {b['optimized_ensemble']['mae']:.2f}")


if __name__ == "__main__":
    main()

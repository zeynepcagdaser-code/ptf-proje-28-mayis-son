#!/usr/bin/env python3
"""
Main regression tabular baseline (40-feature sequences) + balanced-rule hybrid.

Uses data/model main_regression sequences. Does NOT train LSTM.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parent
SEQUENCE_DIR = PROJECT_ROOT / "data" / "model"
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "lstm_next24_v1.parquet"
MASTER_PATH = PROJECT_ROOT / "data" / "master" / "master_hourly_v1.parquet"
REPORTS_DIR = PROJECT_ROOT / "reports"
PRED_DIR = PROJECT_ROOT / "data" / "predictions"

MAIN_REG_JSON = REPORTS_DIR / "main_tabular_regression_metrics.json"
MAIN_REG_MD = REPORTS_DIR / "main_tabular_regression_metrics.md"
HYBRID_JSON = REPORTS_DIR / "hybrid_balanced_rule_metrics.json"
HYBRID_MD = REPORTS_DIR / "hybrid_balanced_rule_metrics.md"
MAIN_PRED_CSV = PRED_DIR / "main_tabular_predictions.csv"
HYBRID_PRED_CSV = PRED_DIR / "hybrid_balanced_rule_predictions.csv"

HORIZONS = list(range(1, 25))
WINDOW_24 = 24
WINDOW_168 = 168
LOW_THRESHOLD = 50.0
NORMAL_THRESHOLD = 100.0

ANCHOR_FILES = {
    "train": "anchor_train.csv",
    "validation": "anchor_val.csv",
    "test": "anchor_test.csv",
}
SPLIT_KEYS = {
    "train": "train",
    "validation": "val",
    "test": "test",
}


def _sequences_to_tabular(X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    n, window_len, n_feat = X.shape
    if n_feat != len(feature_names):
        raise ValueError(f"Feature dim {n_feat} != names {len(feature_names)}")
    if window_len < WINDOW_168:
        raise ValueError(f"Window {window_len} < {WINDOW_168}")

    last = X[:, -1, :]
    win_24 = X[:, -WINDOW_24:, :]
    win_168 = X[:, -WINDOW_168:, :]
    mean_24 = np.mean(win_24, axis=1)

    cols: dict[str, np.ndarray] = {}
    reducers = (
        ("mean", np.mean),
        ("std", np.std),
        ("min", np.min),
        ("max", np.max),
    )
    for i, name in enumerate(feature_names):
        cols[f"{name}_last"] = last[:, i]
        cols[f"{name}_trend_last_minus_mean_24h"] = last[:, i] - mean_24[:, i]
        for stat_name, reducer in reducers:
            cols[f"{name}_{stat_name}_24h"] = reducer(win_24, axis=1)[:, i]
            cols[f"{name}_{stat_name}_168h"] = reducer(win_168, axis=1)[:, i]

    return pd.DataFrame(cols)


def _load_split_arrays(split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    key = SPLIT_KEYS[split]
    X = np.load(SEQUENCE_DIR / f"X_{key}.npy")
    y = np.load(SEQUENCE_DIR / f"y_{key}.npy")
    anchor = pd.read_csv(SEQUENCE_DIR / ANCHOR_FILES[split])
    anchor["anchor_ts_hour"] = pd.to_datetime(anchor["anchor_ts_hour"], utc=True)
    if len(anchor) != len(X):
        raise ValueError(f"{split}: anchor {len(anchor)} != X {len(X)}")
    return X, y, anchor


def _load_feature_names() -> list[str]:
    meta = json.loads((SEQUENCE_DIR / "sequence_metadata.json").read_text(encoding="utf-8"))
    path = SEQUENCE_DIR / "feature_columns.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return list(meta["feature_columns"])


def _inverse_targets(y_scaled: np.ndarray) -> np.ndarray:
    scaler = joblib.load(SEQUENCE_DIR / "target_scaler.pkl")
    return scaler.inverse_transform(y_scaled).astype(np.float64)


def _load_ptf_lookup() -> pd.DataFrame:
    feat_cols = [
        "ts_hour",
        "ptf_lag_24",
        "ptf_low_ratio_24",
        "ptf_zero_ratio_24",
        "ptf_zero_ratio_168",
    ]
    from src.utils.io_utils import read_parquet_with_normalized_ts
    feat = read_parquet_with_normalized_ts(FEATURES_PATH, columns=feat_cols)
    feat["ts_hour"] = pd.to_datetime(feat["ts_hour"], utc=True)
    from src.utils.io_utils import read_parquet_with_normalized_ts
    master = read_parquet_with_normalized_ts(MASTER_PATH, columns=["ts_hour", "ptf_price"])
    master["ts_hour"] = pd.to_datetime(master["ts_hour"], utc=True)
    df = feat.merge(master, on="ts_hour", how="left")
    return df.set_index("ts_hour")


def _balanced_rule_signal(ptf_lookup: pd.DataFrame, anchor_ts: pd.Series) -> np.ndarray:
    row = ptf_lookup.reindex(anchor_ts)
    sig = (
        (row["ptf_low_ratio_24"].fillna(0) > 0)
        | (row["ptf_zero_ratio_24"].fillna(0) > 0)
        | (row["ptf_zero_ratio_168"].fillna(0) > 0.05)
    )
    return sig.fillna(False).astype(int).to_numpy()


def _persistence_prices(
    ptf_lookup: pd.DataFrame,
    anchor_ts: pd.Series,
    horizon: int,
) -> np.ndarray:
    """PTF(t+h-24) at each anchor; fallback to ptf_lag_24-based same-hour proxy."""
    persist_ts = anchor_ts + pd.to_timedelta(horizon - 24, unit="h")
    prices = ptf_lookup["ptf_price"].reindex(persist_ts).to_numpy(dtype=float).copy()
    fallback = ptf_lookup["ptf_lag_24"].reindex(anchor_ts).to_numpy(dtype=float)
    mask = np.isnan(prices)
    prices[mask] = fallback[mask]
    return prices


def _make_regressor(backend: str) -> Any:
    if backend == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=42,
        )
    if backend == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("lightgbm not installed") from exc
        return LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=40,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        min_samples_leaf=40,
        random_state=42,
    )


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float("nan")
    return float(mean_absolute_error(y_true[mask], y_pred[mask]))


def _slice_metrics(
    actual: np.ndarray,
    preds: dict[str, np.ndarray],
) -> dict[str, Any]:
    zero = actual == 0
    low = actual <= LOW_THRESHOLD
    normal = actual > NORMAL_THRESHOLD

    out: dict[str, Any] = {"rows": int(len(actual))}
    for name, pred in preds.items():
        out[name] = {
            "overall_mae": _mae(actual, pred),
            "zero_only_mae": _mae(actual[zero], pred[zero]) if zero.any() else None,
            "low_le_50_mae": _mae(actual[low], pred[low]) if low.any() else None,
            "normal_price_mae": _mae(actual[normal], pred[normal]) if normal.any() else None,
        }
    return out


def run_pipeline(*, backend: str = "hist_gradient_boosting") -> dict[str, Any]:
    feature_names = _load_feature_names()
    ptf_lookup = _load_ptf_lookup()

    splits_data: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        X, y_scaled, anchor = _load_split_arrays(split)
        X_tab = _sequences_to_tabular(X, feature_names)
        y_tl = _inverse_targets(y_scaled)
        splits_data[split] = {
            "X_tab": X_tab,
            "y_tl": y_tl,
            "anchor": anchor,
            "balanced": _balanced_rule_signal(ptf_lookup, anchor["anchor_ts_hour"]),
        }

    models: dict[int, Any] = {}
    pred_frames: list[pd.DataFrame] = []

    for h in HORIZONS:
        hi = h - 1
        X_tr = splits_data["train"]["X_tab"]
        y_tr = splits_data["train"]["y_tl"][:, hi]
        model = _make_regressor(backend)
        model.fit(X_tr, y_tr)
        models[h] = model

        for split in ("train", "validation", "test"):
            blob = splits_data[split]
            anchor = blob["anchor"]
            actual = blob["y_tl"][:, hi]
            main_pred = model.predict(blob["X_tab"]).astype(float)
            persist_pred = _persistence_prices(
                ptf_lookup, anchor["anchor_ts_hour"], h
            )
            balanced = blob["balanced"]
            hybrid_pred = np.where(
                balanced == 1,
                np.minimum(main_pred, persist_pred),
                main_pred,
            )

            chunk = pd.DataFrame(
                {
                    "sample_index": anchor["sample_index"].astype(int),
                    "anchor_ts_hour": anchor["anchor_ts_hour"],
                    "split": split,
                    "horizon": h,
                    "actual_ptf": actual,
                    "main_pred": main_pred,
                    "persistence_pred": persist_pred,
                    "hybrid_pred": hybrid_pred,
                    "balanced_rule_signal": balanced,
                }
            )
            pred_frames.append(chunk)

    long_df = pd.concat(pred_frames, ignore_index=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(MAIN_PRED_CSV, index=False)
    long_df.to_csv(HYBRID_PRED_CSV, index=False)

    main_report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "base_features": len(feature_names),
        "tabular_features": int(splits_data["train"]["X_tab"].shape[1]),
        "splits": {},
        "per_horizon": [],
    }
    hybrid_report: dict[str, Any] = {
        "generated_at_utc": main_report["generated_at_utc"],
        "hybrid_policy": (
            "if balanced_rule_signal==1: hybrid=min(main_pred,persistence_pred) "
            "else hybrid=main_pred"
        ),
        "persistence_definition": "PTF at anchor+(h-24)h from features parquet; fallback ptf_lag_24",
        "balanced_rule": (
            "ptf_low_ratio_24>0 OR ptf_zero_ratio_24>0 OR ptf_zero_ratio_168>0.05"
        ),
        "splits": {},
        "per_horizon": [],
    }

    for split in ("train", "validation", "test"):
        sub = long_df[long_df["split"] == split]
        actual = sub["actual_ptf"].to_numpy(dtype=float)
        preds = {
            "main": sub["main_pred"].to_numpy(dtype=float),
            "persistence": sub["persistence_pred"].to_numpy(dtype=float),
            "hybrid": sub["hybrid_pred"].to_numpy(dtype=float),
        }
        main_report["splits"][split] = _slice_metrics(actual, preds)
        hybrid_report["splits"][split] = _slice_metrics(actual, preds)

    for h in HORIZONS:
        sub = long_df[long_df["horizon"] == h]
        actual = sub["actual_ptf"].to_numpy(dtype=float)
        preds = {
            "main": sub["main_pred"].to_numpy(dtype=float),
            "persistence": sub["persistence_pred"].to_numpy(dtype=float),
            "hybrid": sub["hybrid_pred"].to_numpy(dtype=float),
        }
        row = _slice_metrics(actual, preds)
        row["horizon"] = h
        main_report["per_horizon"].append(row)
        hybrid_report["per_horizon"].append(row)

    main_report["outputs"] = {"predictions_csv": str(MAIN_PRED_CSV.relative_to(PROJECT_ROOT))}
    hybrid_report["outputs"] = {"predictions_csv": str(HYBRID_PRED_CSV.relative_to(PROJECT_ROOT))}

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MAIN_REG_JSON.write_text(json.dumps(main_report, indent=2, default=str), encoding="utf-8")
    MAIN_REG_MD.write_text(_main_md(main_report), encoding="utf-8")
    HYBRID_JSON.write_text(json.dumps(hybrid_report, indent=2, default=str), encoding="utf-8")
    HYBRID_MD.write_text(_hybrid_md(hybrid_report, main_report), encoding="utf-8")

    return {
        "main_report": main_report,
        "hybrid_report": hybrid_report,
        "main_report_md": str(MAIN_REG_MD),
        "hybrid_report_md": str(HYBRID_MD),
    }


def _main_md(report: dict[str, Any]) -> str:
    lines = [
        "# Main Tabular Regression Metrics",
        "",
        f"- **Backend:** `{report['backend']}`",
        f"- **Tabular features:** {report['tabular_features']} (from {report['base_features']} sequence features)",
        "",
        "## MAE by split",
        "",
        "| Split | main | persistence | hybrid |",
        "|-------|-----:|------------:|-------:|",
    ]
    for split, blob in report["splits"].items():
        lines.append(
            f"| {split} | {blob['main']['overall_mae']:.2f} | "
            f"{blob['persistence']['overall_mae']:.2f} | "
            f"{blob['hybrid']['overall_mae']:.2f} |"
        )
    lines += [
        "",
        "## Test slices (main model)",
        "",
    ]
    test = report["splits"]["test"]["main"]
    lines.append(f"- Overall MAE: **{test['overall_mae']:.2f}**")
    lines.append(f"- Zero-only MAE: **{test['zero_only_mae']:.2f}**")
    lines.append(f"- Low<=50 MAE: **{test['low_le_50_mae']:.2f}**")
    lines.append(f"- Normal-price MAE: **{test['normal_price_mae']:.2f}**")
    lines += ["", "## Per-horizon MAE (test, main)", "", "| h | MAE |", "|--:|----:|"]
    for row in report["per_horizon"]:
        sub = row["main"]["overall_mae"]
        if row["horizon"] <= 3 or row["horizon"] in (6, 12, 18, 24):
            lines.append(f"| {row['horizon']} | {sub:.2f} |")
    return "\n".join(lines) + "\n"


def _hybrid_md(hybrid: dict[str, Any], main: dict[str, Any]) -> str:
    lines = [
        "# Hybrid Balanced-Rule Metrics",
        "",
        hybrid["hybrid_policy"],
        "",
        f"**Balanced rule:** {hybrid['balanced_rule']}",
        "",
        "## MAE comparison (test)",
        "",
        "| Model | overall | zero-only | low<=50 | normal |",
        "|-------|--------:|----------:|--------:|-------:|",
    ]
    for name in ("main", "persistence", "hybrid"):
        t = hybrid["splits"]["test"][name]
        lines.append(
            f"| {name} | {t['overall_mae']:.2f} | "
            f"{'' if t['zero_only_mae'] is None else f'{t['zero_only_mae']:.2f}'} | "
            f"{'' if t['low_le_50_mae'] is None else f'{t['low_le_50_mae']:.2f}'} | "
            f"{'' if t['normal_price_mae'] is None else f'{t['normal_price_mae']:.2f}'} |"
        )
    test_main = hybrid["splits"]["test"]["main"]["overall_mae"]
    test_hyb = hybrid["splits"]["test"]["hybrid"]["overall_mae"]
    delta = test_main - test_hyb
    lines += [
        "",
        f"Hybrid vs main on test overall: **{delta:+.2f} TL** "
        f"({'better' if delta > 0 else 'worse'} when positive means hybrid lower MAE)",
        "",
        "## Per-horizon MAE (test)",
        "",
        "| h | main | persistence | hybrid |",
        "|--:|-----:|------------:|-------:|",
    ]
    for row in hybrid["per_horizon"]:
        lines.append(
            f"| {row['horizon']} | {row['main']['overall_mae']:.2f} | "
            f"{row['persistence']['overall_mae']:.2f} | "
            f"{row['hybrid']['overall_mae']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def _default_backend() -> str:
    try:
        import lightgbm  # noqa: F401

        return "lightgbm"
    except ImportError:
        return "hist_gradient_boosting"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["hist_gradient_boosting", "random_forest", "lightgbm"],
        default=None,
    )
    args = parser.parse_args()
    backend = args.backend or _default_backend()
    result = run_pipeline(backend=backend)
    mr = result["main_report"]
    hr = result["hybrid_report"]
    print("=== Main Tabular + Hybrid Baseline ===")
    print("Backend:", mr["backend"])
    print("Tabular features:", mr["tabular_features"])
    for split in ("train", "validation", "test"):
        m = mr["splits"][split]["main"]["overall_mae"]
        h = hr["splits"][split]["hybrid"]["overall_mae"]
        p = hr["splits"][split]["persistence"]["overall_mae"]
        print(f"  {split}: main MAE={m:.2f} persistence={p:.2f} hybrid={h:.2f}")
    t = hr["splits"]["test"]
    print("Test hybrid: zero MAE", t["hybrid"]["zero_only_mae"], "low MAE", t["hybrid"]["low_le_50_mae"])
    print("Wrote:", result["main_report_md"], result["hybrid_report_md"])


if __name__ == "__main__":
    main()

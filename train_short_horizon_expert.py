#!/usr/bin/env python3
"""
Short-horizon expert (h1–h4 only): persistence + residual LightGBM per horizon.

Focused feature set; no h5–h24 training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "lstm_next24_v1.parquet"
MODEL_DIR = PROJECT_ROOT / "models" / "short_horizon_expert"
ANCHOR_TEST_PATH = PROJECT_ROOT / "data" / "model" / "anchor_test.csv"
PREDICTIONS_CSV = PROJECT_ROOT / "data" / "predictions" / "short_horizon_expert_predictions.csv"
METRICS_JSON = PROJECT_ROOT / "reports" / "short_horizon_expert_metrics.json"
METRICS_MD = PROJECT_ROOT / "reports" / "short_horizon_expert_metrics.md"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

ADVANCED_TREE_PRED = PROJECT_ROOT / "data" / "predictions" / "tree_advanced_test_predictions.csv"
RESIDUAL_LSTM_PRED = PROJECT_ROOT / "data" / "predictions" / "lstm_residual_test_predictions.csv"
ADVANCED_TREE_METRICS = PROJECT_ROOT / "reports" / "tree_advanced_metrics.json"
RESIDUAL_LSTM_METRICS = PROJECT_ROOT / "reports" / "lstm_residual_metrics.json"
PERSISTENCE_METRICS = PROJECT_ROOT / "reports" / "persistence_metrics.json"

SHORT_HORIZONS = [1, 2, 3, 4]
EARLY_STOPPING_ROUNDS = 50
MAX_BOOST_ROUNDS = 2000

BASE_FEATURE_COLS = [
    "ptf_lag_1",
    "ptf_lag_24",
    "ptf_lag_48",
    "ptf_lag_168",
    "ptf_roll_mean_24",
    "ptf_roll_std_24",
    "kgup_total_minus_load",
    "kgup_renewable_share",
    "wind_forecast_share",
    "smf_ptf_spread_lag_24",
    "smf_ptf_spread_lag_168",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "is_holiday_tr",
]


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def add_persistence(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("ts_hour").reset_index(drop=True)
    for h in SHORT_HORIZONS:
        out[f"persistence_{h}h"] = out[f"target_{h}h"].shift(24)
    return out


def horizon_features(h: int) -> list[str]:
    return BASE_FEATURE_COLS + [f"persistence_{h}h"]


def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
) -> tuple[Any, dict[str, float]]:
    import lightgbm as lgb

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    params = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": 42,
    }
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=MAX_BOOST_ROUNDS,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(0)],
    )
    importance = dict(zip(feature_names, booster.feature_importance(importance_type="gain")))
    return booster, importance


def load_splits(
    df: pd.DataFrame, h: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    tcol = f"target_{h}h"
    pcol = f"persistence_{h}h"
    fcols = horizon_features(h)
    cols = fcols + [tcol, pcol, "ts_hour", "split"]

    train = df[df["split"] == "train"].dropna(subset=cols)
    val = df[df["split"] == "validation"].dropna(subset=cols)
    test = df[df["split"] == "test"].dropna(subset=cols)
    return train, val, test, fcols


def filter_lstm_anchors(pred_df: pd.DataFrame) -> pd.DataFrame:
    if not ANCHOR_TEST_PATH.exists():
        return pred_df
    anchors = pd.read_csv(ANCHOR_TEST_PATH)
    anchors["anchor_ts_hour"] = pd.to_datetime(
        anchors["anchor_ts_hour"], utc=True
    ).dt.tz_convert("Europe/Istanbul")
    out = pred_df.copy()
    out["anchor_ts_hour"] = pd.to_datetime(out["anchor_ts_hour"], utc=True).dt.tz_convert(
        "Europe/Istanbul"
    )
    return out.merge(anchors[["anchor_ts_hour"]], on="anchor_ts_hour", how="inner")


def horizon_mae_from_df(df: pd.DataFrame, pred_col: str = "predicted_price") -> dict[int, float]:
    out = {}
    for h in SHORT_HORIZONS:
        sub = df[df["target_hour"] == h]
        if len(sub):
            out[h] = mae(sub["actual_price"].to_numpy(), sub[pred_col].to_numpy())
    return out


def compare_baseline_mae(
    aligned: pd.DataFrame,
    *,
    pred_path: Path,
    label: str,
) -> dict[str, float] | None:
    if not pred_path.exists():
        return None
    other = pd.read_csv(pred_path)
    other = other[other["target_hour"].isin(SHORT_HORIZONS)]
    other["anchor_ts_hour"] = pd.to_datetime(other["anchor_ts_hour"], utc=True).dt.tz_convert(
        "Europe/Istanbul"
    )
    merged = aligned.merge(
        other[["anchor_ts_hour", "target_hour", "predicted_price"]].rename(
            columns={"predicted_price": "baseline_pred"}
        ),
        on=["anchor_ts_hour", "target_hour"],
        how="inner",
    )
    if merged.empty:
        return None
    by_h = horizon_mae_from_df(merged, pred_col="baseline_pred")
    by_h["mean_h1_h4"] = float(np.mean(list(by_h.values())))
    by_h["label"] = label
    return by_h


def plot_horizon_mae(expert: dict[int, float], persistence: dict[int, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hours = SHORT_HORIZONS
    x = np.arange(len(hours))
    w = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - w / 2, [expert[h] for h in hours], width=w, label="Short expert", color="seagreen")
    plt.bar(x + w / 2, [persistence[h] for h in hours], width=w, label="Persistence", color="#6baed6")
    plt.xticks(x, [f"h{h}" for h in hours])
    plt.ylabel("MAE (TL/MWh)")
    plt.title("Short horizon expert vs persistence (aligned test)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_feature_importance(all_importance: dict[int, dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    for idx, h in enumerate(SHORT_HORIZONS):
        imp = all_importance[h]
        sorted_items = sorted(imp.items(), key=lambda x: -x[1])[:12]
        names = [k for k, _ in sorted_items]
        vals = [v for _, v in sorted_items]
        axes[idx].barh(names[::-1], vals[::-1], color="steelblue")
        axes[idx].set_title(f"h{h} — top features (gain)")
        axes[idx].set_xlabel("importance")
    plt.suptitle("Short horizon expert — feature importance")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_md(report: dict) -> str:
    cmp_ = report["comparisons"]
    expert = report["expert_aligned"]
    lines = [
        "# Short Horizon Expert (h1–h4)",
        "",
        "Final prediction: `persistence_h + predicted_residual`",
        "",
        f"- Features: {len(BASE_FEATURE_COLS)} base + `persistence_h` per horizon",
        f"- Test anchors (aligned): {expert['test_anchors']}",
        "",
        "## Expert MAE (aligned)",
        "",
        f"| Horizon | MAE |",
        f"|--------|-----:|",
    ]
    for h in SHORT_HORIZONS:
        lines.append(f"| h{h} | {expert['horizon_mae'][str(h)]:.2f} |")
    lines.append(f"| **Mean h1–h4** | **{expert['mean_mae_h1_h4']:.2f}** |")
    lines.append("")

    def _cmp_block(name: str, data: dict | None) -> list[str]:
        if not data:
            return [f"### {name}", "", "_Not available._", ""]
        rows = [f"### {name}", "", "| Horizon | MAE |", "|--------|-----:|"]
        for h in SHORT_HORIZONS:
            key = str(h) if str(h) in data else h
            if key in data:
                rows.append(f"| h{h} | {data[key]:.2f} |")
        if "mean_h1_h4" in data:
            rows.append(f"| Mean h1–h4 | {data['mean_h1_h4']:.2f} |")
        rows.append("")
        return rows

    lines.extend(_cmp_block("Persistence", cmp_.get("persistence")))
    lines.extend(_cmp_block("Advanced tree", cmp_.get("advanced_tree")))
    lines.extend(_cmp_block("Residual LSTM", cmp_.get("residual_lstm")))

    lines.extend(
        [
            "## vs persistence (expert − persistence MAE)",
            "",
        ]
    )
    for h in SHORT_HORIZONS:
        delta = report["delta_vs_persistence"][str(h)]
        lines.append(f"- h{h}: {delta:+.2f} TL/MWh")
    lines.append(f"- Mean: {report['delta_vs_persistence']['mean_h1_h4']:+.2f} TL/MWh")
    lines.append("")
    lines.extend(
        [
            "## Summary",
            "",
            f"- Beats persistence (mean): **{report.get('beats_persistence_mean_h1_h4')}**",
            f"- Beats advanced tree (mean): **{report.get('beats_advanced_tree_mean_h1_h4')}**",
            f"- Beats residual LSTM (mean): **{report.get('beats_residual_lstm_mean_h1_h4')}**",
            "",
            f"**{report['verdict']}**",
        ]
    )
    return "\n".join(lines)


def run(*, smoke: bool = False) -> dict:
    try:
        import lightgbm  # noqa: F401
    except ImportError as e:
        raise ImportError("LightGBM required. pip install lightgbm") from e

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing {FEATURES_PATH}")

    from src.utils.io_utils import read_parquet_with_normalized_ts
    df = add_persistence(read_parquet_with_normalized_ts(FEATURES_PATH))
    missing = [c for c in BASE_FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    horizons = SHORT_HORIZONS[:2] if smoke else SHORT_HORIZONS
    all_importance: dict[int, dict[str, float]] = {}
    rows: list[dict] = []

    for h in horizons:
        train, val, test, fcols = load_splits(df, h)
        tcol = f"target_{h}h"
        pcol = f"persistence_{h}h"

        X_train = train[fcols].to_numpy(dtype=np.float64)
        y_train = (train[tcol] - train[pcol]).to_numpy(dtype=np.float64)
        X_val = val[fcols].to_numpy(dtype=np.float64)
        y_val = (val[tcol] - val[pcol]).to_numpy(dtype=np.float64)

        booster, imp = train_lgbm(X_train, y_train, X_val, y_val, fcols)
        all_importance[h] = imp
        model_path = MODEL_DIR / f"horizon_{h:02d}.txt"
        booster.save_model(str(model_path))
        (MODEL_DIR / f"horizon_{h:02d}_importance.json").write_text(
            json.dumps(imp, indent=2), encoding="utf-8"
        )

        X_test = test[fcols].to_numpy(dtype=np.float64)
        residual_pred = booster.predict(X_test)

        for i, idx in enumerate(test.index):
            row = test.loc[idx]
            actual = float(row[tcol])
            persistence = float(row[pcol])
            pred = persistence + float(residual_pred[i])
            rows.append(
                {
                    "anchor_ts_hour": str(row["ts_hour"]),
                    "target_hour": h,
                    "actual_price": actual,
                    "persistence_price": persistence,
                    "predicted_residual": float(residual_pred[i]),
                    "predicted_price": pred,
                    "absolute_error": abs(actual - pred),
                    "persistence_error": abs(actual - persistence),
                }
            )

        val_mae = mae(y_val, booster.predict(X_val))
        print(f"h{h} val residual MAE={val_mae:.2f} best_iter={booster.best_iteration}")

    pred_df = pd.DataFrame(rows)
    PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(PREDICTIONS_CSV, index=False)

    aligned = filter_lstm_anchors(pred_df)
    expert_h = horizon_mae_from_df(aligned)
    persist_h = horizon_mae_from_df(aligned, pred_col="persistence_price")
    expert_mean = float(np.mean(list(expert_h.values())))
    persist_mean = float(np.mean(list(persist_h.values())))

    cmp_persist = {str(k): v for k, v in persist_h.items()}
    cmp_persist["mean_h1_h4"] = persist_mean

    cmp_adv = compare_baseline_mae(aligned, pred_path=ADVANCED_TREE_PRED, label="advanced_tree")
    cmp_res = compare_baseline_mae(aligned, pred_path=RESIDUAL_LSTM_PRED, label="residual_lstm")

    delta = {str(h): expert_h[h] - persist_h[h] for h in expert_h}
    delta["mean_h1_h4"] = expert_mean - persist_mean

    beats_persistence = expert_mean < persist_mean
    adv_mean = cmp_adv["mean_h1_h4"] if cmp_adv else None
    res_mean = cmp_res["mean_h1_h4"] if cmp_res else None
    beats_advanced = adv_mean is not None and expert_mean < adv_mean
    beats_residual = res_mean is not None and expert_mean < res_mean

    if not beats_persistence:
        verdict = (
            "Short expert is WORSE than persistence on mean h1–h4 MAE — do not deploy as-is."
        )
    elif adv_mean is not None and expert_mean >= adv_mean:
        verdict = (
            f"Short expert beats persistence ({expert_mean:.1f} vs {persist_mean:.1f}) "
            f"but is worse than advanced tree h1–h4 ({adv_mean:.1f}). "
            "Consider using advanced tree for short horizons or tuning this expert."
        )
    elif res_mean is not None and expert_mean >= res_mean:
        verdict = (
            f"Short expert beats persistence but does not beat residual LSTM on h1–h4 "
            f"({expert_mean:.1f} vs {res_mean:.1f})."
        )
    else:
        verdict = "Short expert beats persistence on mean h1–h4 MAE."

    report["beats_advanced_tree_mean_h1_h4"] = beats_advanced
    report["beats_residual_lstm_mean_h1_h4"] = beats_residual

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_horizon_mae(expert_h, persist_h, FIGURES_DIR / "short_horizon_mae.png")
    if not smoke:
        plot_feature_importance(all_importance, FIGURES_DIR / "short_horizon_feature_importance.png")

    report = {
        "scope": "target_1h..target_4h only",
        "method": "persistence_plus_residual_lgbm",
        "feature_columns": {f"h{h}": horizon_features(h) for h in horizons},
        "model_dir": str(MODEL_DIR),
        "predictions_path": str(PREDICTIONS_CSV),
        "expert_aligned": {
            "test_anchors": int(aligned["anchor_ts_hour"].nunique()),
            "prediction_rows": int(len(aligned)),
            "horizon_mae": {str(k): v for k, v in expert_h.items()},
            "mean_mae_h1_h4": expert_mean,
        },
        "comparisons": {
            "persistence": cmp_persist,
            "advanced_tree": cmp_adv,
            "residual_lstm": cmp_res,
        },
        "reference_reports": {
            "persistence_metrics_json": str(PERSISTENCE_METRICS)
            if PERSISTENCE_METRICS.exists()
            else None,
            "advanced_tree_h1_h4_from_report": _json_horizons(ADVANCED_TREE_METRICS, SHORT_HORIZONS)
            if ADVANCED_TREE_METRICS.exists()
            else None,
            "residual_lstm_test_mae": _json_get(RESIDUAL_LSTM_METRICS, "test_mae"),
        },
        "delta_vs_persistence": delta,
        "beats_persistence_mean_h1_h4": beats_persistence,
        "verdict": verdict,
        "feature_importance": {str(h): all_importance[h] for h in horizons},
        "smoke_test": smoke,
    }

    METRICS_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    METRICS_MD.write_text(write_md(report), encoding="utf-8")
    return report


def _json_horizons(path: Path, horizons: list[int]) -> dict | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    hm = data.get("lstm_anchor_aligned", {}).get("horizon_mae", {})
    if not hm:
        return None
    out = {str(h): hm.get(str(h)) for h in horizons}
    out["mean_h1_h4"] = float(np.mean([out[str(h)] for h in horizons if out[str(h)] is not None]))
    return out


def _json_get(path: Path, key: str) -> float | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get(key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train short-horizon h1–h4 expert")
    parser.add_argument("--smoke-test", action="store_true", help="Train h1–h2 only")
    args = parser.parse_args()

    report = run(smoke=args.smoke_test)
    e = report["expert_aligned"]
    print("\n=== Short horizon expert ===")
    for h in SHORT_HORIZONS:
        if str(h) in e["horizon_mae"]:
            print(f"  h{h} MAE: {e['horizon_mae'][str(h)]:.2f}")
    print(f"  Mean h1–h4: {e['mean_mae_h1_h4']:.2f}")
    print(report["verdict"])
    print(f"Metrics: {METRICS_JSON}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""h1–h4 LightGBM on microstructure features (persistence + residual)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "lstm_microstructure_next24_v1.parquet"
MODEL_DIR = PROJECT_ROOT / "models" / "microstructure_h1h4"
ANCHOR_TEST_PATH = PROJECT_ROOT / "data" / "model" / "anchor_test.csv"
PREDICTIONS_CSV = PROJECT_ROOT / "data" / "predictions" / "microstructure_h1h4_predictions.csv"
METRICS_JSON = PROJECT_ROOT / "reports" / "microstructure_h1h4_metrics.json"
METRICS_MD = PROJECT_ROOT / "reports" / "microstructure_h1h4_metrics.md"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "microstructure_h1h4_mae.png"
OPTUNA_BEST_PARAMS_JSON = PROJECT_ROOT / "reports" / "optuna_best_params.json"

BASELINE_PREDS = {
    "advanced_tree": PROJECT_ROOT / "data" / "predictions" / "tree_advanced_test_predictions.csv",
    "short_expert": PROJECT_ROOT / "data" / "predictions" / "short_horizon_expert_predictions.csv",
}

HORIZONS = [1, 2, 3, 4]
EARLY_STOPPING_ROUNDS = 50
MAX_BOOST_ROUNDS = 2000


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def add_persistence(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("ts_hour").reset_index(drop=True)
    for h in HORIZONS:
        out[f"persistence_{h}h"] = out[f"target_{h}h"].shift(24)
    return out


def resolve_base_features(df: pd.DataFrame) -> list[str]:
    target_cols = [c for c in df.columns if c.startswith("target_")]
    exclude = {"ts_hour", "split", *target_cols}
    exclude.update(c for c in df.columns if c.startswith("persistence_"))
    return [c for c in df.columns if c not in exclude]


def horizon_features(base: list[str], h: int) -> list[str]:
    return base + [f"persistence_{h}h"]


def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    params_override: dict[str, Any] | None = None,
) -> Any:
    import lightgbm as lgb

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    base_params: dict[str, Any] = {
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
    if params_override:
        base_params.update(params_override)

    booster = lgb.train(
        base_params,
        train_set,
        num_boost_round=int(base_params.get("num_boost_round", MAX_BOOST_ROUNDS)),
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(0)],
    )
    return booster


def load_splits(df: pd.DataFrame, h: int, base: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    tcol = f"target_{h}h"
    pcol = f"persistence_{h}h"
    fcols = horizon_features(base, h)
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


def horizon_mae_df(df: pd.DataFrame, pred_col: str) -> dict[int, float]:
    out = {}
    for h in HORIZONS:
        sub = df[df["target_hour"] == h]
        if len(sub):
            out[h] = mae(sub["actual_price"].to_numpy(), sub[pred_col].to_numpy())
    return out


def baseline_mae(aligned: pd.DataFrame, path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    other = pd.read_csv(path)
    other = other[other["target_hour"].isin(HORIZONS)]
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
    by_h = horizon_mae_df(merged, "baseline_pred")
    out = {str(k): v for k, v in by_h.items()}
    out["mean_h1_h4"] = float(np.mean(list(by_h.values())))
    return out


def plot_mae_comparison(
    model: dict[int, float],
    baselines: dict[str, dict[int, float] | None],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(HORIZONS))
    width = 0.15
    n = 1 + sum(1 for v in baselines.values() if v)
    offset = np.linspace(-(n - 1) / 2 * width, (n - 1) / 2 * width, n)

    plt.figure(figsize=(10, 5))
    plt.bar(x + offset[0], [model[h] for h in HORIZONS], width=width, label="Microstructure", color="seagreen")
    idx = 1
    colors = {"persistence": "#6baed6", "advanced_tree": "#e6550d", "short_expert": "#756bb1"}
    for name, data in baselines.items():
        if not data:
            continue
        plt.bar(
            x + offset[idx],
            [data.get(h, data.get(str(h))) for h in HORIZONS],
            width=width,
            label=name.replace("_", " "),
            color=colors.get(name, "gray"),
        )
        idx += 1
    plt.xticks(x, [f"h{h}" for h in HORIZONS])
    plt.ylabel("MAE (TL/MWh)")
    plt.title("h1–h4 MAE — microstructure vs baselines (aligned test)")
    plt.legend(fontsize=8)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_md(report: dict) -> str:
    m = report["model_aligned"]
    cmp_ = report["comparisons"]
    lines = [
        "# Microstructure h1–h4 LightGBM",
        "",
        f"- **Source:** `{report['features_path']}`",
        f"- **Features per horizon:** {report['feature_count_per_horizon']} (all non-target + `persistence_h`)",
        f"- **Method:** persistence + residual",
        "",
        "## Model MAE (aligned test)",
        "",
        "| Horizon | MAE |",
        "|--------|-----:|",
    ]
    for h in HORIZONS:
        lines.append(f"| h{h} | {m['horizon_mae'][str(h)]:.2f} |")
    lines.append(f"| **Mean h1–h4** | **{m['mean_mae_h1_h4']:.2f}** |")
    lines.append("")

    for name in ["persistence", "advanced_tree", "short_expert"]:
        data = cmp_.get(name)
        lines.append(f"### {name.replace('_', ' ').title()}")
        if not data:
            lines.append("_Not available._\n")
            continue
        lines.append("| Horizon | MAE |")
        lines.append("|--------|-----:|")
        for h in HORIZONS:
            lines.append(f"| h{h} | {data[str(h)]:.2f} |")
        lines.append(f"| Mean | {data['mean_h1_h4']:.2f} |")
        lines.append("")

    lines.extend(
        [
            "## Delta vs baselines (model − baseline, TL/MWh)",
            "",
        ]
    )
    for name, delta in report["delta_vs_baselines"].items():
        lines.append(f"- **{name}** mean: {delta['mean_h1_h4']:+.2f}")
    lines.append("")
    lines.append(f"**{report['verdict']}**")
    return "\n".join(lines)


def optuna_search_params(
    *,
    df: pd.DataFrame,
    base_features: list[str],
    n_trials: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Optimize hyperparameters on VALIDATION ONLY.

    Objective: mean validation MAE (price) over h1–h4.
    This does NOT touch the test split.
    """
    import optuna

    # Materialize splits and arrays once (speed + determinism).
    split_cache: dict[int, dict[str, Any]] = {}
    for h in HORIZONS:
        train, val, _test, fcols = load_splits(df, h, base_features)
        tcol, pcol = f"target_{h}h", f"persistence_{h}h"
        split_cache[h] = {
            "fcols": fcols,
            "X_train": train[fcols].to_numpy(dtype=np.float64),
            "y_train": (train[tcol] - train[pcol]).to_numpy(dtype=np.float64),
            "X_val": val[fcols].to_numpy(dtype=np.float64),
            "y_val_res": (val[tcol] - val[pcol]).to_numpy(dtype=np.float64),
            "y_val_price": val[tcol].to_numpy(dtype=np.float64),
            "val_persistence": val[pcol].to_numpy(dtype=np.float64),
        }

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "seed": seed,
            # Search space (requested)
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "min_data_in_leaf": trial.suggest_int("min_child_samples", 5, 100),
            "feature_fraction": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("subsample", 0.5, 1.0),
            "bagging_freq": 1,
            "num_boost_round": trial.suggest_int("n_estimators", 200, 2000),
        }

        # Safety: LGBM requires num_leaves <= 2^max_depth (roughly); constrain softly.
        # If invalid, optuna can handle exceptions, but we reduce wasted trials.
        max_leaves = 2 ** params["max_depth"]
        if params["num_leaves"] > max_leaves:
            params["num_leaves"] = max_leaves

        maes: list[float] = []
        for h in HORIZONS:
            cached = split_cache[h]
            booster = train_lgbm(
                cast(np.ndarray, cached["X_train"]),
                cast(np.ndarray, cached["y_train"]),
                cast(np.ndarray, cached["X_val"]),
                cast(np.ndarray, cached["y_val_res"]),
                cast(list[str], cached["fcols"]),
                params_override=params,
            )
            pred_res = booster.predict(cast(np.ndarray, cached["X_val"]))
            pred_price = cast(np.ndarray, cached["val_persistence"]) + pred_res
            maes.append(mae(cast(np.ndarray, cached["y_val_price"]), pred_price))

        return float(np.mean(maes))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = dict(study.best_params)
    # Normalize param names back to LGBM conventions used in this script.
    best_params: dict[str, Any] = {
        "learning_rate": float(best["learning_rate"]),
        "max_depth": int(best["max_depth"]),
        "num_leaves": int(best["num_leaves"]),
        "min_data_in_leaf": int(best["min_child_samples"]),
        "feature_fraction": float(best["colsample_bytree"]),
        "bagging_fraction": float(best["subsample"]),
        "bagging_freq": 1,
        "num_boost_round": int(best["n_estimators"]),
        "seed": seed,
    }
    max_leaves = 2 ** best_params["max_depth"]
    if best_params["num_leaves"] > max_leaves:
        best_params["num_leaves"] = max_leaves

    OPTUNA_BEST_PARAMS_JSON.parent.mkdir(parents=True, exist_ok=True)
    OPTUNA_BEST_PARAMS_JSON.write_text(
        json.dumps(
            {
                "study_name": study.study_name,
                "n_trials": n_trials,
                "best_value_validation_mean_mae_h1_h4": float(study.best_value),
                "best_params": best_params,
                "search_space": {
                    "n_estimators": [200, 2000],
                    "max_depth": [3, 10],
                    "learning_rate": [0.005, 0.3],
                    "num_leaves": [20, 300],
                    "subsample": [0.5, 1.0],
                    "colsample_bytree": [0.5, 1.0],
                    "min_child_samples": [5, 100],
                },
                "note": "Optimized on VALIDATION ONLY; test split not used in objective.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return best_params


def run(*, smoke: bool = False, optuna_trials: int = 0) -> dict:
    import lightgbm  # noqa: F401

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing {FEATURES_PATH}. Run build_microstructure_features.py")

    df = add_persistence(pd.read_parquet(FEATURES_PATH))
    base_features = resolve_base_features(df)
    horizons = HORIZONS[:2] if smoke else HORIZONS

    params_override: dict[str, Any] | None = None
    if optuna_trials and not smoke:
        params_override = optuna_search_params(df=df, base_features=base_features, n_trials=optuna_trials)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for h in horizons:
        train, val, test, fcols = load_splits(df, h, base_features)
        tcol, pcol = f"target_{h}h", f"persistence_{h}h"

        X_train = train[fcols].to_numpy(dtype=np.float64)
        y_train = (train[tcol] - train[pcol]).to_numpy(dtype=np.float64)
        X_val = val[fcols].to_numpy(dtype=np.float64)
        y_val = (val[tcol] - val[pcol]).to_numpy(dtype=np.float64)

        booster = train_lgbm(X_train, y_train, X_val, y_val, fcols, params_override=params_override)
        booster.save_model(str(MODEL_DIR / f"horizon_{h:02d}.txt"))
        (MODEL_DIR / f"horizon_{h:02d}_features.json").write_text(
            json.dumps(fcols, indent=2), encoding="utf-8"
        )

        X_test = test[fcols].to_numpy(dtype=np.float64)
        res_pred = booster.predict(X_test)
        print(f"h{h} val_residual_mae={mae(y_val, booster.predict(X_val)):.2f} iter={booster.best_iteration}")

        for i, idx in enumerate(test.index):
            row = test.loc[idx]
            actual = float(row[tcol])
            persistence = float(row[pcol])
            pred = persistence + float(res_pred[i])
            rows.append(
                {
                    "anchor_ts_hour": str(row["ts_hour"]),
                    "target_hour": h,
                    "actual_price": actual,
                    "persistence_price": persistence,
                    "predicted_residual": float(res_pred[i]),
                    "predicted_price": pred,
                    "absolute_error": abs(actual - pred),
                }
            )

    pred_df = pd.DataFrame(rows)
    PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(PREDICTIONS_CSV, index=False)

    aligned = filter_lstm_anchors(pred_df)
    model_h = horizon_mae_df(aligned, "predicted_price")
    persist_h = horizon_mae_df(aligned, "persistence_price")
    model_mean = float(np.mean(list(model_h.values())))
    persist_mean = float(np.mean(list(persist_h.values())))

    comparisons = {
        "persistence": {**{str(k): v for k, v in persist_h.items()}, "mean_h1_h4": persist_mean},
        "advanced_tree": baseline_mae(aligned, BASELINE_PREDS["advanced_tree"]),
        "short_expert": baseline_mae(aligned, BASELINE_PREDS["short_expert"]),
    }

    delta_vs = {}
    for name, base in comparisons.items():
        if not base:
            continue
        delta_vs[name] = {
            str(h): model_h[h] - base.get(h, base.get(str(h))) for h in HORIZONS
        }
        delta_vs[name]["mean_h1_h4"] = model_mean - base["mean_h1_h4"]

    adv_mean = comparisons["advanced_tree"]["mean_h1_h4"] if comparisons["advanced_tree"] else None
    beats_adv = adv_mean is not None and model_mean < adv_mean
    beats_persist = model_mean < persist_mean
    beats_short = (
        comparisons["short_expert"]
        and model_mean < comparisons["short_expert"]["mean_h1_h4"]
    )

    if adv_mean is not None and not beats_adv:
        verdict = (
            f"Microstructure h1–h4 is WORSE than advanced tree on mean MAE "
            f"({model_mean:.1f} vs {adv_mean:.1f}). Advanced tree remains best for short horizons."
        )
    elif beats_adv:
        verdict = (
            f"Microstructure h1–h4 beats advanced tree ({model_mean:.1f} vs {adv_mean:.1f} mean MAE)."
        )
    elif beats_persist:
        verdict = f"Microstructure beats persistence ({model_mean:.1f} vs {persist_mean:.1f}) but advanced tree comparison unavailable."
    else:
        verdict = "Microstructure does not beat persistence on h1–h4."

    plot_mae_comparison(
        model_h,
        {"persistence": persist_h, **{k: v for k, v in comparisons.items() if k != "persistence"}},
        FIGURE_PATH,
    )

    report = {
        "features_path": str(FEATURES_PATH),
        "feature_count_base": len(base_features),
        "feature_count_per_horizon": len(base_features) + 1,
        "method": "persistence_plus_residual_lgbm",
        "model_dir": str(MODEL_DIR),
        "predictions_path": str(PREDICTIONS_CSV),
        "model_aligned": {
            "test_anchors": int(aligned["anchor_ts_hour"].nunique()),
            "prediction_rows": len(aligned),
            "horizon_mae": {str(k): v for k, v in model_h.items()},
            "mean_mae_h1_h4": model_mean,
        },
        "comparisons": comparisons,
        "delta_vs_baselines": delta_vs,
        "beats_persistence": beats_persist,
        "beats_advanced_tree": beats_adv,
        "beats_short_expert": beats_short,
        "verdict": verdict,
        "smoke_test": smoke,
    }

    METRICS_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    METRICS_MD.write_text(write_md(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train microstructure LGBM h1–h4")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--optuna-trials", type=int, default=0, help="If set (>0), run Optuna on validation then retrain with best params.")
    args = parser.parse_args()

    report = run(smoke=args.smoke_test, optuna_trials=args.optuna_trials)
    m = report["model_aligned"]
    print("\n=== Microstructure h1–h4 ===")
    for h in HORIZONS:
        if str(h) in m["horizon_mae"]:
            print(f"  h{h} MAE: {m['horizon_mae'][str(h)]:.2f}")
    print(f"  Mean: {m['mean_mae_h1_h4']:.2f}")
    print(report["verdict"])
    print(f"Metrics: {METRICS_JSON}")
    if args.optuna_trials and not args.smoke_test:
        print(f"Optuna best params: {OPTUNA_BEST_PARAMS_JSON}")


if __name__ == "__main__":
    main()

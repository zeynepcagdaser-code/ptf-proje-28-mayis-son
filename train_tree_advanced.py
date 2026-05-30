#!/usr/bin/env python3
"""
Advanced tree pipeline:
  - persistence + residual (per horizon)
  - hour-of-day specific models (24 × 24)
  - zero-price & spike classifiers
  - rolling online refit (train+val) with recency weights (30–90d)
  - microstructure features
"""

from __future__ import annotations

import argparse

from tree_advanced.config import DEFAULT_RECENCY_BOOST, DEFAULT_RECENCY_DAYS, FEATURES_PATH
from tree_advanced.pipeline import run_training


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train advanced hour×horizon tree models")
    p.add_argument("--smoke-test", action="store_true", help="2 horizons × 3 hours")
    p.add_argument("--skip-train", action="store_true", help="Predict only (models exist)")
    p.add_argument("--no-rolling", action="store_true", help="Disable online refit pass")
    p.add_argument(
        "--no-classifier-overrides",
        action="store_true",
        help="Disable zero/spike hard overrides at inference",
    )
    p.add_argument("--recency-days", type=int, default=DEFAULT_RECENCY_DAYS, help="Full boost window")
    p.add_argument("--recency-medium-days", type=int, default=90, help="Partial boost window")
    p.add_argument("--recency-boost", type=float, default=DEFAULT_RECENCY_BOOST)
    p.add_argument("--features", type=str, default=str(FEATURES_PATH))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_classifier_overrides:
        import tree_advanced.config as tc

        tc.APPLY_CLASSIFIER_OVERRIDES = False

    report = run_training(
        features_path=args.features,
        smoke=args.smoke_test,
        recency_days=args.recency_days,
        recency_medium_days=args.recency_medium_days,
        recency_boost=args.recency_boost,
        rolling_refit=not args.no_rolling,
        skip_train=args.skip_train,
    )
    a = report["lstm_anchor_aligned"]
    c = a["persistence_comparison"]
    print("\n=== Tree advanced complete ===")
    print(f"Backend: {report['backend']}")
    print(f"Aligned MAE: {a['mae']:.2f}")
    print(f"Persistence: {c['persistence_mae']:.2f} ({c['improvement_pct_vs_persistence']:.2f}%)")
    if "residual_lstm_mae" in a:
        print(f"Residual LSTM: {a['residual_lstm_mae']:.2f}")
    from tree_advanced import config as cfg

    print(f"Metrics: {cfg.METRICS_JSON}")


if __name__ == "__main__":
    main()

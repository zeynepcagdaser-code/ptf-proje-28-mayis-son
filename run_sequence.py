#!/usr/bin/env python3
"""Build LSTM-ready numpy sequences and leakage-safe scalers."""

from __future__ import annotations

import argparse

from sequence.config import DEFAULT_FEATURE_PROFILE, FEATURE_PROFILES
from sequence.pipeline import run_pipeline


def _print_report(report: dict) -> None:
    profile = report.get("feature_profile", DEFAULT_FEATURE_PROFILE)
    n_feat = report["feature_count"]
    print("Sequence dataset built.")
    print("Feature profile:", profile)
    print("Output dir:", report["output_dir"])
    if profile == "main_regression":
        print("Main regression sequence feature count:", n_feat)
    elif profile == "low_price_classifier":
        print("Low-price classifier sequence feature count:", n_feat)
    else:
        print("Sequence feature count:", n_feat)
    print(
        "Feature resolution:",
        f"{report.get('resolved_feature_count', n_feat)}/"
        f"{report.get('requested_feature_count', n_feat)} resolved,",
        f"{report.get('missing_feature_count', 0)} missing",
    )
    if report.get("missing_features"):
        print("Missing features:", ", ".join(report["missing_features"]))
    print("Targets:", report["target_count"])
    print("Window:", report["window_size"], "| Horizon:", report["horizon"])
    print("\nShapes:")
    for split, shape_info in report["shapes"].items():
        x_shape = tuple(shape_info["X"])
        print(f"  {split}: X={x_shape}, y={tuple(shape_info['y'])}")
        if split == "train":
            print(f"  (train feature dim check: window={x_shape[1]}, features={x_shape[2]})")
    print("\nDropped (NaN sequences):")
    for split, count in report["dropped_nan_sequences"].items():
        print(f"  {split}: {count}")
    print("\nScaler fit split:", report["scaler_fit_split"])
    if report.get("anchor_files"):
        print("\nAnchor files:")
        for split, path in report["anchor_files"].items():
            print(f"  {split}: {path}")
    print("Metadata:", report.get("metadata_file"))
    print("JSON report:", report["report_json"])
    print("Markdown report:", report["report_md"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build scaled sequence datasets from lstm_next24_v1.parquet."
    )
    parser.add_argument(
        "--feature-profile",
        choices=sorted(FEATURE_PROFILES.keys()),
        default=DEFAULT_FEATURE_PROFILE,
        help=(
            "Feature bucket from features.config "
            f"(default: {DEFAULT_FEATURE_PROFILE})"
        ),
    )
    args = parser.parse_args()

    report = run_pipeline(feature_profile=args.feature_profile)
    _print_report(report)


if __name__ == "__main__":
    main()

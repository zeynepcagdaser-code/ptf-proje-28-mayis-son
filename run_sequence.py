#!/usr/bin/env python3
"""Build LSTM-ready numpy sequences and leakage-safe scalers."""

from sequence.pipeline import run_pipeline


def main() -> None:
    report = run_pipeline()
    print("Sequence dataset built.")
    print("Output dir:", report["output_dir"])
    print("Features:", report["feature_count"], "| Targets:", report["target_count"])
    print("Window:", report["window_size"], "| Horizon:", report["horizon"])
    print("\nShapes:")
    for split, shape_info in report["shapes"].items():
        print(f"  {split}: X={tuple(shape_info['X'])}, y={tuple(shape_info['y'])}")
    print("\nDropped (NaN sequences):")
    for split, count in report["dropped_nan_sequences"].items():
        print(f"  {split}: {count}")
    print("\nScaler fit split:", report["scaler_fit_split"])
    if report.get("anchor_files"):
        print("\nAnchor files:")
        for split, path in report["anchor_files"].items():
            print(f"  {split}: {path}")
    print("JSON report:", report["report_json"])
    print("Markdown report:", report["report_md"])


if __name__ == "__main__":
    main()

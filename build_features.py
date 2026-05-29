#!/usr/bin/env python3
"""Build leakage-safe LSTM feature dataset (tabular)."""

from features.build_features import run_build


def main() -> None:
    report = run_build()
    print("Feature dataset built.")
    print("Output:", report["output_path"])
    print("Rows:", report["row_count"])
    print("Features:", report["feature_count"], "| Targets:", report["target_count"])
    print("Dropped (targets):", report["rows_dropped_missing_targets"])
    print("Dropped (history):", report["rows_dropped_insufficient_history"])
    print("Splits:", report["split_counts"])
    print("JSON report:", report["report_json"])
    print("Markdown report:", report["report_md"])


if __name__ == "__main__":
    main()

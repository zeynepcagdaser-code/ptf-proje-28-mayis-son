#!/usr/bin/env python3
"""Build residual-learning feature dataset (persistence + residual targets)."""

from features.build_residual_features import run_build


def main() -> None:
    report = run_build()
    print("Residual feature dataset built.")
    print("Output:", report["output_path"])
    print("Rows:", report["row_count"])
    print("Features:", report["feature_count"])
    print("Residual targets:", report["target_count"])
    print("Price targets kept:", len(report.get("price_target_columns", [])))
    print("Persistence columns:", len(report.get("persistence_columns", [])))
    print("Splits:", report["split_counts"])
    print("JSON report:", report["report_json"])
    print("Markdown report:", report["report_md"])


if __name__ == "__main__":
    main()

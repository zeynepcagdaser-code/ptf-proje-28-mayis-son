#!/usr/bin/env python3
"""Build tree + microstructure feature parquet."""

from features.build_tree_features import run_build


def main() -> None:
    report = run_build()
    print("Tree feature dataset built.")
    print("Output:", report["output_path"])
    print("Rows:", report["row_count"])
    print("Features:", report["feature_count"])
    print("Microstructure:", report.get("microstructure_feature_count"))


if __name__ == "__main__":
    main()

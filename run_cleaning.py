#!/usr/bin/env python3
"""Run the data cleaning pipeline (raw CSV → data/clean/*.parquet)."""

from cleaning.pipeline import run_pipeline


def main() -> None:
    report = run_pipeline()
    print("Cleaning complete.")
    print("JSON report:", report["report_json"])
    print("Markdown report:", report["report_md"])
    for name, info in report["datasets"].items():
        if info.get("error"):
            print(f"  {name}: ERROR — {info['error']}")
        else:
            print(f"  {name}: {info['rows_in']} → {info['rows_out']} rows → {info['output_path']}")


if __name__ == "__main__":
    main()

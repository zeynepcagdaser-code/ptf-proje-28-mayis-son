#!/usr/bin/env python3
"""Build master_hourly_v1 from data/clean/*.parquet."""

from master.build_master import run_build


def main() -> None:
    report = run_build()
    print("Master dataset built.")
    print("Output:", report["output_path"])
    print("Rows:", report["row_count"], "| Columns:", report["column_count"])
    print("ts_hour:", report["ts_hour_start"], "→", report["ts_hour_end"])
    print("Spine match:", report["row_count_matches_spine"])
    print("JSON report:", report["report_json"])
    print("Markdown report:", report["report_md"])
    print("\nColumns by dataset:")
    for name, count in report["column_count_by_dataset"].items():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()

"""Basit denetim: feature parquet'larında hedef kolonların (ör. `target_ptf`, `price`) varlığını ve doluluk oranını raporlar."""
from pathlib import Path
import sys
import argparse

import pandas as pd


DEFAULT_TARGETS = ["target_ptf", "price", "target_regime"]


def inspect_parquet(path: Path, targets=DEFAULT_TARGETS):
    try:
        from src.utils.io_utils import read_parquet_with_normalized_ts
        df = read_parquet_with_normalized_ts(path)
    except Exception as e:
        return {"error": str(e)}

    report = {}
    for t in targets:
        if t in df.columns:
            nonnull = df[t].notna().sum()
            total = len(df)
            report[t] = {"present": True, "nonnull": int(nonnull), "total": int(total)}
        else:
            report[t] = {"present": False}
    return report


def main(root: str = "data/features"):
    p = Path(root)
    if not p.exists():
        print(f"Path not found: {root}")
        return 2

    for f in sorted(p.rglob("*.parquet")):
        r = inspect_parquet(f)
        print(f"{f.relative_to(p)}: {r}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="data/features", help="Features folder to scan")
    args = parser.parse_args()
    sys.exit(main(args.root))

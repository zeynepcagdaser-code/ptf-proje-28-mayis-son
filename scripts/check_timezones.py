"""Basit bir araç: `data/` altındaki CSV/parquet dosyalarında örnek satırlar okuyup zaman damgası türlerini raporlar."""
import argparse
import sys
from pathlib import Path

import pandas as pd


def sample_file(path: Path, n: int = 5):
    try:
        if path.suffix.lower() in (".csv", ".txt"):
            df = pd.read_csv(path, parse_dates=True, nrows=n)
        elif path.suffix.lower() in (".parquet", ".pq"):
            from src.utils.io_utils import read_parquet_with_normalized_ts
            df = read_parquet_with_normalized_ts(path, columns=None)
            df = df.head(n)
        else:
            return None
    except Exception as e:
        return f"error reading: {e}"

    res = {}
    for c in df.columns:
        try:
            s = pd.to_datetime(df[c], errors="coerce")
            tz = s.dt.tz is not None
            res[c] = "tz-aware" if tz else "naive-or-non-datetime"
        except Exception:
            res[c] = "non-datetime"
    return res


def main(root: str):
    p = Path(root)
    if not p.exists():
        print(f"Path not found: {root}")
        return 2

    for f in sorted(p.rglob("*.csv"))[:50]:
        r = sample_file(f)
        print(f"{f.relative_to(p)}: {r}")

    for f in sorted(p.rglob("*.parquet"))[:50]:
        r = sample_file(f)
        print(f"{f.relative_to(p)}: {r}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="data", help="Root data folder to scan")
    args = parser.parse_args()
    sys.exit(main(args.root))

#!/usr/bin/env python3
"""Run hourly: update EPİAŞ PTF, rebuild features, retrain and forecast next delivery day.

Usage: ./run_hourly_pipeline.py  (runs forever) or run under systemd/cron.
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=PROJECT_ROOT, check=check)


def read_max_ptf_ts() -> datetime | None:
    import pandas as pd

    p = PROJECT_ROOT / "data" / "ptf_dataset.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        if "date" in df.columns:
            s = pd.to_datetime(df["date"], errors="coerce")
            if s.notna().any():
                return s.max().to_pydatetime()
    except Exception:
        return None
    return None


def next_top_of_hour() -> float:
    now = datetime.now()
    next = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
    return (next - now).total_seconds()


def main_loop() -> None:
    py = sys.executable
    print("Starting hourly pipeline loop. Press Ctrl-C to stop.")
    while True:
        start = datetime.now()
        try:
            # 1) update EPİAŞ dataset (fetch up to default forward look or explicit target)
            run([py, "update_dataset.py"])  # uses FORWARD_LOOK_DAYS by default

            # 2) determine next-day target: day after last published PTF hour
            max_ts = read_max_ptf_ts()
            if max_ts is None:
                target = (datetime.now().date() + timedelta(days=2)).isoformat()
            else:
                target = (max_ts.date() + timedelta(days=1)).isoformat()

            # 3) run end-to-end D+2 pipeline for that target
            run([py, "run_d2_ptf_pipeline.py", target])

        except subprocess.CalledProcessError as exc:
            print(f"Step failed: {exc}; will retry on next hour.")
        except Exception as exc:
            print(f"Unexpected error: {exc}; will retry on next hour.")

        # Sleep until next top of hour
        secs = next_top_of_hour()
        # If we've already spent >50 seconds into the hour, run after remaining
        if secs <= 0:
            secs = 3600
        print(f"Waiting {int(secs)}s until next run (next hour).\n")
        time.sleep(secs)


if __name__ == "__main__":
    main_loop()

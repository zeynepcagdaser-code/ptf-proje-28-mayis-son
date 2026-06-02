#!/usr/bin/env python3
"""End-to-end D+2 PTF pipeline: refresh inputs, build features, forecast."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    target = (datetime.now().date() + timedelta(days=2)).isoformat()
    if len(sys.argv) > 1:
        target = sys.argv[1]

    optional_fetches = [
        [sys.executable, "fetch_load_forecast.py"],
        [sys.executable, "fetch_kgup_combined.py"],
        [sys.executable, "fetch_wind_forecast.py"],
    ]
    for cmd in optional_fetches:
        try:
            run(cmd)
        except subprocess.CalledProcessError as exc:
            print(f"Warning: fetch step failed ({exc}); continuing with cached data.")

    run([sys.executable, "build_d2_ptf_features.py", "--target-date", target])
    run([sys.executable, "train_d2_ptf_forecaster.py"])
    run([sys.executable, "run_d2_ptf_forecast.py", "--target-date", target])
    print(f"\nDone. Forecast for {target} -> data/predictions/d2_ptf_forecast.csv")


if __name__ == "__main__":
    main()

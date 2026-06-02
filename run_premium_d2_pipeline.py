#!/usr/bin/env python3
"""Premium D+2 pipeline: proprietary models + plant KGUP + intraday curves."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run(cmd: list[str], optional: bool = False) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        if optional:
            print(f"Warning: optional step failed: {exc}")
        else:
            raise


def main() -> None:
    target = (datetime.now().date() + timedelta(days=2)).isoformat()
    if len(sys.argv) > 1:
        target = sys.argv[1]

    py = sys.executable
    for cmd in [
        [py, "fetch_load_forecast.py"],
        [py, "fetch_kgup_combined.py"],
        [py, "fetch_wind_forecast.py"],
    ]:
        run(cmd, optional=True)

    run([py, "build_plant_level_kgup_pipeline.py"], optional=True)
    run([py, "build_must_run_proxy_v2.py"], optional=True)
    run([py, "fetch_and_reconstruct_weekly_dam_curves.py"], optional=True)

    run([py, "build_premium_d2_features.py", "--target-date", target])
    run([py, "train_d2_ptf_forecaster.py"], optional=True)
    run([py, "train_premium_d2_blend.py"], optional=True)
    run([py, "run_premium_d2_forecast.py", "--target-date", target])
    print(f"\nPremium D+2 forecast ready for {target}")


if __name__ == "__main__":
    main()

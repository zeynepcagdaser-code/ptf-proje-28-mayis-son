#!/usr/bin/env python3
"""Run the production-style rolling next-24h PTF pipeline.

Default behavior:
1. Refresh EPİAŞ datasets that are safe/useful for live forecasting.
2. Train the rolling model if no selected model exists, or when --train is passed.
3. Write data/predictions/rolling_ptf_next24_forecast.csv.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run(cmd: list[str], *, optional: bool = False) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError:
        if optional:
            print(f"Warning: optional step failed: {' '.join(cmd)}", flush=True)
            return
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-fetch", action="store_true", help="Use existing local data only.")
    parser.add_argument("--train", action="store_true", help="Retrain the selected rolling model before forecasting.")
    parser.add_argument("--quick", action="store_true", help="Use quick LightGBM training settings.")
    parser.add_argument("--profile", default="full_market", help="Model profile to train/use.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    py = sys.executable

    if not args.no_fetch:
        # PTF refresh moves the anchor forward. Forecast/planned sources provide delivery-hour inputs.
        for script in [
            "update_dataset.py",
            "fetch_load_forecast.py",
            "fetch_kgup_combined.py",
            "fetch_wind_forecast.py",
            "fetch_yal_yat.py",
            "fetch_smf.py",
            "fetch_real_consumption.py",
            "fetch_realtime_generation.py",
            "fetch_tcmb_exchange_rates.py",
            "fetch_grf_daily_reference_price.py",
            "fetch_reservoir.py",
            "fetch_idm_prices.py",
            "fetch_international_prices.py",
        ]:
            run([py, script], optional=True)

    selected = PROJECT_ROOT / "models" / "rolling_ptf_next24" / "selected_profile.json"
    if args.train or not selected.exists():
        cmd = [py, "rolling_ptf_forecast_system.py", "train", "--profile", args.profile]
        if args.quick:
            cmd.append("--quick")
        run(cmd)

    run([py, "rolling_ptf_forecast_system.py", "forecast", "--profile", args.profile])


if __name__ == "__main__":
    main()

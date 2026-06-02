#!/usr/bin/env python3
"""
Fetch historical hourly temperature for Ankara (lat=39.9, lon=32.85) from Open-Meteo archive API.

Output:
  data/raw/temperature_hourly.csv

Notes:
  - This is a raw download step (no feature engineering here).
  - Timezone is Europe/Istanbul to match the rest of the repo.
"""

from __future__ import annotations

from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_PATH = PROJECT_ROOT / "data" / "raw" / "temperature_hourly.csv"


def main() -> None:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 39.9,
        "longitude": 32.85,
        "hourly": ["temperature_2m", "apparent_temperature"],
        "start_date": "2020-01-01",
        "end_date": "2025-12-31",
        "timezone": "Europe/Istanbul",
        "format": "csv",
    }
    r = requests.get(url, params=params, timeout=120)
    # Save body even on non-200 to help debug.
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(r.text, encoding="utf-8")
    if not r.ok:
        raise RuntimeError(f"Open-Meteo request failed: status={r.status_code}. Saved body to {OUT_PATH}")
    print("Indirildi:", len(r.text), "karakter")
    print("Yazildi:", OUT_PATH)


if __name__ == "__main__":
    main()


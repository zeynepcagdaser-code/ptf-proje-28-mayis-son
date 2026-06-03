#!/usr/bin/env python3
"""Fetch hourly weather data for 9 major Turkish cities from Open-Meteo.

Produces population-weighted national aggregates and HVAC-load features
for use in rolling_ptf_forecast_system.py.

Open-Meteo Archive API: historical data (start → today-8 days)
Open-Meteo Forecast API: recent gap + 48h ahead (past_days=10)

No authentication required.

Output: data/processed/temperature_hourly.parquet

Per-city columns (9 cities × 2 variables):
  temp_{city}, apparent_temp_{city}

National weighted aggregates:
  tr_temp_mean, tr_apparent_temp_mean, tr_humidity_mean,
  tr_cloud_cover_mean, tr_radiation_mean, tr_wind_speed_mean

Derived HVAC features:
  tr_cooling_degree  = max(0, tr_temp_mean - 22)
  tr_heating_degree  = max(0, 18 - tr_temp_mean)
  tr_heatwave_flag   = 1 if tr_temp_mean >= 35 else 0
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH   = PROCESSED_DIR / "temperature_hourly.parquet"
REPORT_JSON   = PROJECT_ROOT / "reports" / "temperature_report.json"
REPORT_MD     = PROJECT_ROOT / "reports" / "temperature_report.md"

ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "cloud_cover",
    "shortwave_radiation",
    "wind_speed_10m",
]

# Population from TÜİK 2023 provincial estimates
CITIES: list[dict[str, Any]] = [
    {"id": "istanbul",   "lat": 41.0082, "lon": 28.9784, "pop": 15_840_900},
    {"id": "ankara",     "lat": 39.9334, "lon": 32.8597, "pop":  5_782_300},
    {"id": "izmir",      "lat": 38.4189, "lon": 27.1287, "pop":  4_479_600},
    {"id": "bursa",      "lat": 40.1885, "lon": 29.0610, "pop":  3_194_700},
    {"id": "antalya",    "lat": 36.8969, "lon": 30.7133, "pop":  2_688_000},
    {"id": "adana",      "lat": 37.0000, "lon": 35.3213, "pop":  2_274_100},
    {"id": "konya",      "lat": 37.8715, "lon": 32.4846, "pop":  2_330_300},
    {"id": "diyarbakir", "lat": 37.9144, "lon": 40.2306, "pop":  1_791_300},
    {"id": "samsun",     "lat": 41.2867, "lon": 36.3300, "pop":  1_363_900},
]

_total_pop = sum(c["pop"] for c in CITIES)
for _c in CITIES:
    _c["weight"] = _c["pop"] / _total_pop

# Variable name → column prefix used in per-city and aggregate columns
_VAR_PREFIX: dict[str, str] = {
    "temperature_2m":       "temp",
    "apparent_temperature": "apparent_temp",
    "relative_humidity_2m": "humidity",
    "cloud_cover":          "cloud",
    "shortwave_radiation":  "radiation",
    "wind_speed_10m":       "wind",
}

# Per-city columns to keep in the final parquet (other intermediates are dropped)
_KEEP_PER_CITY_PREFIXES = {"temp", "apparent_temp"}

# Prefix → final weighted-aggregate column name
_AGG_COLS: dict[str, str] = {
    "temp":          "tr_temp_mean",
    "apparent_temp": "tr_apparent_temp_mean",
    "humidity":      "tr_humidity_mean",
    "cloud":         "tr_cloud_cover_mean",
    "radiation":     "tr_radiation_mean",
    "wind":          "tr_wind_speed_mean",
}

COOLING_THRESHOLD  = 22.0   # °C comfort baseline for Turkey
HEATING_THRESHOLD  = 18.0   # °C heating demand onset
HEATWAVE_THRESHOLD = 35.0   # °C national weighted mean → heatwave flag

ARCHIVE_LAG_DAYS    = 8    # archive API has ~5-7 day delay; be conservative
FORECAST_PAST_DAYS  = 10   # days of past data from forecast API (fills the gap)
FORECAST_FUTURE_DAYS = 2   # days ahead for live forecasting
ROLLING_REFRESH_HOURS = 72  # re-fetch last 72 h on incremental run
REQUEST_TIMEOUT     = (10, 90)
MAX_RETRIES         = 3
SLEEP_BETWEEN_CITIES = 0.5
HISTORY_START       = datetime(2020, 1, 1)


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _get_with_retries(url: str, params: dict) -> dict:
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429 and attempt < MAX_RETRIES:
                wait = min(120, 5 * 2 ** (attempt - 1))
                _log(f"  429 rate-limit → wait {wait}s (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
            return resp.json()
        except (requests.exceptions.RequestException, RuntimeError) as exc:
            last_err = exc
            wait = min(60, 3 * 2 ** (attempt - 1))
            _log(f"  retry {attempt}/{MAX_RETRIES} after {wait}s — {exc}")
            time.sleep(wait)
    raise last_err  # type: ignore[misc]


def _parse_city_response(data: dict, city_id: str) -> pd.DataFrame:
    """Convert Open-Meteo JSON response into a tidy per-city DataFrame."""
    hourly = data.get("hourly", {})
    times  = hourly.get("time", [])
    if not times:
        return pd.DataFrame()

    ts = pd.to_datetime(times, errors="coerce").floor("h")
    # Open-Meteo returns naive strings in the requested timezone (Istanbul local)
    if getattr(ts, "tz", None) is not None:
        ts = ts.tz_localize(None)

    df = pd.DataFrame({"ts_hour": ts})
    for var in HOURLY_VARS:
        if var not in hourly:
            continue
        prefix = _VAR_PREFIX[var]
        col    = f"{prefix}_{city_id}"
        df[col] = pd.to_numeric(hourly[var], errors="coerce")

    return (
        df.dropna(subset=["ts_hour"])
        .drop_duplicates("ts_hour")
        .sort_values("ts_hour")
        .reset_index(drop=True)
    )


def _fetch_archive_chunk(city: dict, start: datetime, end: datetime) -> pd.DataFrame:
    params = {
        "latitude":        city["lat"],
        "longitude":       city["lon"],
        "start_date":      start.strftime("%Y-%m-%d"),
        "end_date":        end.strftime("%Y-%m-%d"),
        "hourly":          ",".join(HOURLY_VARS),
        "timezone":        "Europe/Istanbul",
        "wind_speed_unit": "ms",
    }
    return _parse_city_response(_get_with_retries(ARCHIVE_URL, params), city["id"])


def _fetch_forecast_chunk(city: dict) -> pd.DataFrame:
    params = {
        "latitude":        city["lat"],
        "longitude":       city["lon"],
        "hourly":          ",".join(HOURLY_VARS),
        "timezone":        "Europe/Istanbul",
        "past_days":       FORECAST_PAST_DAYS,
        "forecast_days":   FORECAST_FUTURE_DAYS,
        "wind_speed_unit": "ms",
    }
    return _parse_city_response(_get_with_retries(FORECAST_URL, params), city["id"])


def fetch_city(city: dict, archive_start: datetime, archive_end: datetime) -> pd.DataFrame:
    """Fetch full time series for one city: archive (chunked yearly) + forecast."""
    frames: list[pd.DataFrame] = []

    # Archive: year-by-year chunks
    cur_year = archive_start.year
    while True:
        chunk_start = max(archive_start, datetime(cur_year, 1, 1))
        chunk_end   = min(archive_end,   datetime(cur_year, 12, 31))
        if chunk_start > archive_end:
            break
        _log(f"  archive {city['id']}: {chunk_start.date()} → {chunk_end.date()}")
        try:
            chunk = _fetch_archive_chunk(city, chunk_start, chunk_end)
            if not chunk.empty:
                frames.append(chunk)
        except Exception as exc:
            _log(f"  WARN: {city['id']} archive {cur_year} failed — {exc}")
        time.sleep(SLEEP_BETWEEN_CITIES)
        if cur_year >= archive_end.year:
            break
        cur_year += 1

    # Forecast: fills the archive lag gap + 48h ahead
    _log(f"  forecast {city['id']}: past={FORECAST_PAST_DAYS}d ahead={FORECAST_FUTURE_DAYS}d")
    try:
        fc = _fetch_forecast_chunk(city)
        if not fc.empty:
            frames.append(fc)
    except Exception as exc:
        _log(f"  WARN: {city['id']} forecast failed — {exc}")
    time.sleep(SLEEP_BETWEEN_CITIES)

    if not frames:
        _log(f"  ERROR: no data for {city['id']}")
        return pd.DataFrame()

    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("ts_hour", keep="last")
        .sort_values("ts_hour")
        .reset_index(drop=True)
    )


def _incremental_archive_start() -> datetime:
    if not OUTPUT_PATH.exists():
        return HISTORY_START
    try:
        df = pd.read_parquet(OUTPUT_PATH, columns=["ts_hour"])
        last = pd.to_datetime(df["ts_hour"]).max()
        if pd.isna(last):
            return HISTORY_START
        return (last - timedelta(hours=ROLLING_REFRESH_HOURS)).to_pydatetime()
    except Exception:
        return HISTORY_START


def build_wide_table(city_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Outer-join all per-city DataFrames on ts_hour."""
    dfs = [f for f in city_frames.values() if not f.empty]
    if not dfs:
        raise RuntimeError("All city fetches returned empty DataFrames — cannot continue")
    wide = dfs[0]
    for df in dfs[1:]:
        wide = wide.merge(df, on="ts_hour", how="outer")
    return wide.sort_values("ts_hour").reset_index(drop=True)


def compute_aggregates(wide: pd.DataFrame) -> pd.DataFrame:
    """Add population-weighted national aggregate columns."""
    out = wide.copy()
    for prefix, agg_col in _AGG_COLS.items():
        city_cols = [
            f"{prefix}_{c['id']}"
            for c in CITIES
            if f"{prefix}_{c['id']}" in out.columns
        ]
        if not city_cols:
            continue
        weights = np.array([
            c["weight"] for c in CITIES if f"{prefix}_{c['id']}" in out.columns
        ])
        data = out[city_cols].astype(float).to_numpy()
        valid = (~np.isnan(data)).astype(float)
        w_sum = valid @ weights            # effective weight sum per row
        w_val = np.where(np.isnan(data), 0.0, data) @ weights
        out[agg_col] = np.where(w_sum > 0, w_val / w_sum, np.nan)
    return out


def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "tr_temp_mean" not in out.columns:
        return out
    temp = pd.to_numeric(out["tr_temp_mean"], errors="coerce")
    out["tr_cooling_degree"] = (temp - COOLING_THRESHOLD).clip(lower=0)
    out["tr_heating_degree"] = (HEATING_THRESHOLD - temp).clip(lower=0)
    out["tr_heatwave_flag"]  = (temp >= HEATWAVE_THRESHOLD).astype("int8")
    return out


def drop_intermediate_city_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Remove per-city humidity/cloud/radiation/wind columns (only aggregates needed)."""
    transient_prefixes = tuple(
        f"{p}_" for p in _VAR_PREFIX.values() if p not in _KEEP_PER_CITY_PREFIXES
    )
    drop = [c for c in df.columns if c.startswith(transient_prefixes)]
    return df.drop(columns=drop)


def merge_with_existing(new_wide: pd.DataFrame) -> pd.DataFrame:
    if not OUTPUT_PATH.exists():
        return new_wide
    try:
        old = pd.read_parquet(OUTPUT_PATH)
        old["ts_hour"] = pd.to_datetime(old["ts_hour"]).dt.floor("h")
        # Align columns: add any new columns missing from old data
        for col in new_wide.columns:
            if col not in old.columns:
                old[col] = np.nan
        combined = (
            pd.concat([old, new_wide], ignore_index=True)
            .drop_duplicates("ts_hour", keep="last")
            .sort_values("ts_hour")
            .reset_index(drop=True)
        )
        return combined
    except Exception as exc:
        _log(f"WARN: could not merge with existing parquet ({exc}) — replacing fully")
        return new_wide


def write_report(df: pd.DataFrame) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    numeric_cols = [c for c in df.columns if c != "ts_hour" and pd.api.types.is_numeric_dtype(df[c])]
    report: dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "rows":     int(len(df)),
        "ts_start": str(df["ts_hour"].min()),
        "ts_end":   str(df["ts_hour"].max()),
        "cities":   [c["id"] for c in CITIES],
        "population_weights": {c["id"]: round(c["weight"], 4) for c in CITIES},
        "columns":  list(df.columns),
        "missing_pct": {
            col: float(round(df[col].isna().mean() * 100, 2))
            for col in numeric_cols
        },
        "tr_temp_mean_stats": {},
        "heatwave_hours": 0,
    }
    if "tr_temp_mean" in df.columns:
        t = df["tr_temp_mean"].dropna()
        report["tr_temp_mean_stats"] = {
            "mean": float(t.mean()),
            "min":  float(t.min()),
            "max":  float(t.max()),
            "p5":   float(t.quantile(0.05)),
            "p95":  float(t.quantile(0.95)),
        }
    if "tr_heatwave_flag" in df.columns:
        report["heatwave_hours"] = int(df["tr_heatwave_flag"].sum())

    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    weights_md = "\n".join(
        f"  {c['id']:12s} lat={c['lat']:7.4f}  lon={c['lon']:7.4f}  weight={c['weight']:.4f}"
        for c in CITIES
    )
    bad_missing = {k: v for k, v in report["missing_pct"].items() if v > 1.0}
    missing_md = (
        "\n".join(f"  {col}: {pct:.1f}%" for col, pct in bad_missing.items())
        or "  (all < 1%)"
    )
    ts = report["tr_temp_mean_stats"]
    temp_md = (
        f"  mean={ts['mean']:.1f}  min={ts['min']:.1f}  max={ts['max']:.1f}"
        f"  p5={ts['p5']:.1f}  p95={ts['p95']:.1f}"
        if ts else "  (no data)"
    )
    lines = [
        "# Temperature Data Report",
        "",
        f"Generated: `{report['generated_at']}`  "
        f"Rows: `{report['rows']}`  Range: `{report['ts_start']}` → `{report['ts_end']}`",
        "",
        "## Cities & Population Weights",
        "",
        f"```\n{weights_md}\n```",
        "",
        "## National Weighted Temperature",
        "",
        f"```\n{temp_md}\n```",
        "",
        f"Heatwave hours (tr_temp_mean ≥ {HEATWAVE_THRESHOLD}°C): "
        f"`{report['heatwave_hours']}`",
        "",
        "## Missing Data (> 1%)",
        "",
        f"```\n{missing_md}\n```",
        "",
        "## Output Columns",
        "",
        "| Column | Description |",
        "|--------|-------------|",
        "| `tr_temp_mean` | Population-weighted national temperature (°C) |",
        "| `tr_apparent_temp_mean` | Pop-weighted feels-like temperature (°C) |",
        "| `tr_humidity_mean` | Pop-weighted relative humidity (%) |",
        "| `tr_cloud_cover_mean` | Pop-weighted cloud cover (%) |",
        "| `tr_radiation_mean` | Pop-weighted shortwave radiation (W/m²) |",
        "| `tr_wind_speed_mean` | Pop-weighted wind speed (m/s) |",
        "| `tr_cooling_degree` | max(0, tr_temp_mean − 22) — AC load proxy |",
        "| `tr_heating_degree` | max(0, 18 − tr_temp_mean) — heating load proxy |",
        "| `tr_heatwave_flag` | 1 if tr_temp_mean ≥ 35 °C |",
        "| `temp_{city}` | Per-city temperature_2m (9 cities) |",
        "| `apparent_temp_{city}` | Per-city feels-like temperature (9 cities) |",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full", action="store_true",
                   help="Re-fetch everything from 2020-01-01, ignoring existing parquet.")
    p.add_argument("--start-date", default=None,
                   help="Override archive fetch start (YYYY-MM-DD).")
    p.add_argument("--end-date", default=None,
                   help="Override archive fetch end (YYYY-MM-DD). Default: today−8 days.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    archive_end = (
        pd.Timestamp(args.end_date).to_pydatetime()
        if args.end_date
        else datetime.now() - timedelta(days=ARCHIVE_LAG_DAYS)
    )

    if args.full or not OUTPUT_PATH.exists():
        archive_start = HISTORY_START
        _log("Full fetch mode: 2020-01-01 → …")
    elif args.start_date:
        archive_start = pd.Timestamp(args.start_date).to_pydatetime()
        _log(f"Custom range: {archive_start.date()} → {archive_end.date()}")
    else:
        archive_start = _incremental_archive_start()
        _log(f"Incremental: archive {archive_start.date()} → {archive_end.date()}")

    # ── fetch all cities ─────────────────────────────────────────────────
    city_frames: dict[str, pd.DataFrame] = {}
    for city in CITIES:
        _log(f"Fetching {city['id']} (weight={city['weight']:.3f})")
        df = fetch_city(city, archive_start, archive_end)
        city_frames[city["id"]] = df
        _log(f"  → {len(df)} rows")

    # ── build wide table ─────────────────────────────────────────────────
    _log("Building wide table…")
    wide = build_wide_table(city_frames)

    # ── weighted aggregates ──────────────────────────────────────────────
    _log("Computing weighted aggregates…")
    wide = compute_aggregates(wide)

    # ── derived HVAC features ────────────────────────────────────────────
    wide = compute_derived(wide)

    # ── drop transient per-city columns (humidity, cloud, radiation, wind)
    wide = drop_intermediate_city_cols(wide)

    # ── merge with existing parquet ──────────────────────────────────────
    if args.full:
        final = wide
    else:
        _log("Merging with existing parquet…")
        final = merge_with_existing(wide)

    # ── save ─────────────────────────────────────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUTPUT_PATH, index=False)
    _log(f"Saved: {OUTPUT_PATH}  rows={len(final)}  cols={len(final.columns)}")

    write_report(final)
    _log(f"Report: {REPORT_MD}")

    if "tr_temp_mean" in final.columns:
        t = final["tr_temp_mean"].dropna()
        _log(
            f"tr_temp_mean  mean={t.mean():.1f}°C  "
            f"min={t.min():.1f}°C  max={t.max():.1f}°C"
        )
    if "tr_heatwave_flag" in final.columns:
        _log(f"Heatwave hours (≥{HEATWAVE_THRESHOLD}°C): {int(final['tr_heatwave_flag'].sum())}")


if __name__ == "__main__":
    main()

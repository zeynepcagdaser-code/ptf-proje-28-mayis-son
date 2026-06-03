#!/usr/bin/env python3
"""Fetch international commodity prices and align to project hourly time axis.

Sources (all free, no auth required):
  - Brent Crude Oil:     Yahoo Finance  BZ=F
  - Henry Hub Nat. Gas:  Yahoo Finance  NG=F
  - TTF Nat. Gas (EU):   Yahoo Finance  TTF=F  (may be unavailable; falls back to NG=F proxy)
  - Coal API2 proxy:     Yahoo Finance  MTF=F  (ICE Rotterdam coal futures)

Why these matter for Turkish PTF:
  - Brent / NG=F: long-term gas import contract price anchors
  - TTF: European spot gas; Turkey imports at TTF-linked or take-or-pay prices
  - Coal: marginal cost of Turkish import-coal plants (Zonguldak / İskenderun)

All prices stored in USD; TRY equivalents computed via TCMB rates when available.

Outputs:
  - Raw CSV:  data/external/international/commodity_prices_raw.csv
  - Daily parquet: data/processed/international_commodity_prices_daily.parquet
  - Hourly parquet: data/processed/international_commodity_prices_hourly.parquet
  - Report:  reports/international_commodity_prices_report.{json,md}

Dependencies: yfinance (pip install yfinance)
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DIR = PROJECT_ROOT / "data" / "external" / "international"
RAW_CSV = RAW_DIR / "commodity_prices_raw.csv"
DAILY_PATH = PROJECT_ROOT / "data" / "processed" / "international_commodity_prices_daily.parquet"
HOURLY_PATH = PROJECT_ROOT / "data" / "processed" / "international_commodity_prices_hourly.parquet"
REPORT_JSON = PROJECT_ROOT / "reports" / "international_commodity_prices_report.json"
REPORT_MD = PROJECT_ROOT / "reports" / "international_commodity_prices_report.md"

TCMB_HOURLY = PROJECT_ROOT / "data" / "processed" / "tcmb_exchange_rates_hourly.parquet"

# Yahoo Finance tickers → column names
TICKERS: dict[str, str] = {
    "BZ=F": "brent_usd",       # Brent crude USD/barrel
    "NG=F": "henry_hub_usd",   # Henry Hub USD/MMBtu
    "TTF=F": "ttf_eur_mwh",    # TTF EUR/MWh (may not always be available)
    "MTF=F": "coal_api2_usd",  # Coal API2 USD/t (ICE Rotterdam, may need alt ticker)
}

# Minimum required tickers — script continues even if optional ones fail
REQUIRED_TICKERS = {"BZ=F", "NG=F"}


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _ptf_date_range(start_date: str) -> tuple[datetime, datetime]:
    start = pd.Timestamp(start_date).to_pydatetime()
    ptf_path = PROJECT_ROOT / "data" / "ptf_dataset.csv"
    if ptf_path.exists():
        ts = pd.to_datetime(pd.read_csv(ptf_path, usecols=["date"])["date"], errors="coerce")
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
        end = ts.max().to_pydatetime().replace(tzinfo=None)
    else:
        end = datetime.now()
    return start, end


def _import_yfinance() -> Any:
    try:
        import yfinance as yf
        return yf
    except ImportError:
        raise SystemExit(
            "yfinance paketi bulunamadı. Lütfen kurun:\n"
            "  pip install yfinance\n"
            "veya:\n"
            "  pip install -r requirements.txt"
        )


def fetch_ticker(yf: Any, ticker: str, start: datetime, end: datetime) -> pd.Series | None:
    """Download a single Yahoo Finance ticker; returns daily Close as pd.Series indexed by date."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
        if data.empty:
            _log(f"  {ticker}: boş yanıt")
            return None
        close = data["Close"]
        if hasattr(close, "squeeze"):
            close = close.squeeze()
        # Normalize index to naive UTC dates
        idx = pd.to_datetime(close.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        close.index = idx.floor("D")
        close = close.dropna()
        _log(f"  {ticker}: {len(close)} gün, {close.index.min().date()} → {close.index.max().date()}")
        return close
    except Exception as exc:
        _log(f"  {ticker}: hata — {exc}")
        return None


def build_daily(start: datetime, end: datetime) -> pd.DataFrame:
    yf = _import_yfinance()
    spine_dates = pd.date_range(pd.Timestamp(start).floor("D"), pd.Timestamp(end).floor("D"), freq="D")
    result = pd.DataFrame({"ts_day": spine_dates})

    missing_required = []
    for ticker, col_name in TICKERS.items():
        _log(f"İndiriliyor: {ticker} → {col_name}")
        series = fetch_ticker(yf, ticker, start, end)
        if series is None:
            if ticker in REQUIRED_TICKERS:
                missing_required.append(ticker)
                _log(f"  UYARI: zorunlu ticker {ticker} alınamadı")
            result[col_name] = float("nan")
            continue
        s_df = series.reset_index()
        s_df.columns = ["ts_day", col_name]
        s_df["ts_day"] = pd.to_datetime(s_df["ts_day"]).dt.floor("D")
        result = result.merge(s_df, on="ts_day", how="left")
        # Forward-fill weekends and market holidays
        result[col_name] = result[col_name].ffill()

    if missing_required:
        raise SystemExit(f"Zorunlu ticker(lar) alınamadı: {missing_required}")

    # Derived: weekly momentum and rolling mean
    for base_col in ["brent_usd", "henry_hub_usd", "ttf_eur_mwh", "coal_api2_usd"]:
        if base_col not in result.columns:
            continue
        s = result[base_col]
        result[f"{base_col}_lag_7d"] = s.shift(7)
        result[f"{base_col}_change_7d"] = s - s.shift(7)
        result[f"{base_col}_pct_change_7d"] = s.pct_change(7)
        result[f"{base_col}_roll_mean_30d"] = s.rolling(30, min_periods=7).mean()

    return result.sort_values("ts_day").reset_index(drop=True)


def add_try_equivalents(daily: pd.DataFrame) -> pd.DataFrame:
    """Multiply USD prices by USD/TRY if TCMB daily data is available."""
    tcmb_daily = PROJECT_ROOT / "data" / "processed" / "tcmb_exchange_rates_daily.parquet"
    if not tcmb_daily.exists():
        return daily
    try:
        fx = pd.read_parquet(tcmb_daily)[["ts_day", "usd_try_buy", "eur_try_buy"]].copy()
        fx["ts_day"] = pd.to_datetime(fx["ts_day"]).dt.floor("D")
        out = daily.merge(fx, on="ts_day", how="left")
        for usd_col in ["brent_usd", "henry_hub_usd", "coal_api2_usd"]:
            if usd_col in out.columns:
                try_col = usd_col.replace("_usd", "_try")
                out[try_col] = out[usd_col] * out["usd_try_buy"]
        if "ttf_eur_mwh" in out.columns:
            out["ttf_try_mwh"] = out["ttf_eur_mwh"] * out["eur_try_buy"]
        out = out.drop(columns=["usd_try_buy", "eur_try_buy"], errors="ignore")
        return out
    except Exception as exc:
        _log(f"TRY dönüşümü atlandı: {exc}")
        return daily


def align_to_hourly(daily: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    spine = pd.DataFrame({
        "ts_hour": pd.date_range(
            pd.Timestamp(start).floor("h"),
            pd.Timestamp(end).floor("h"),
            freq="h",
        )
    })
    d = daily.copy()
    d["ts_day"] = pd.to_datetime(d["ts_day"]).dt.floor("D")
    hourly = spine.copy()
    hourly["ts_day"] = hourly["ts_hour"].dt.floor("D")
    hourly = hourly.merge(d, on="ts_day", how="left").sort_values("ts_hour")
    value_cols = [c for c in d.columns if c != "ts_day"]
    hourly[value_cols] = hourly[value_cols].ffill()
    return hourly.drop(columns=["ts_day"]).reset_index(drop=True)


def merge_with_existing(new_daily: pd.DataFrame) -> pd.DataFrame:
    if not RAW_CSV.exists():
        return new_daily
    old = pd.read_csv(RAW_CSV)
    old["ts_day"] = pd.to_datetime(old["ts_day"]).dt.floor("D")
    combined = pd.concat([old, new_daily], ignore_index=True)
    combined = combined.drop_duplicates("ts_day", keep="last").sort_values("ts_day").reset_index(drop=True)
    # Re-apply forward-fill after merge
    value_cols = [c for c in combined.columns if c != "ts_day"]
    combined[value_cols] = combined[value_cols].ffill()
    return combined


def write_report(daily: pd.DataFrame, hourly: pd.DataFrame) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    price_cols = [c for c in TICKERS.values() if c in daily.columns]
    stats = {
        col: {
            "mean": float(daily[col].mean()),
            "min": float(daily[col].min()),
            "max": float(daily[col].max()),
            "missing_pct": float(daily[col].isna().mean() * 100),
        }
        for col in price_cols
    }
    report: dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": "Yahoo Finance (yfinance)",
        "tickers": TICKERS,
        "daily_rows": int(len(daily)),
        "hourly_rows": int(len(hourly)),
        "daily_start": str(daily["ts_day"].min()),
        "daily_end": str(daily["ts_day"].max()),
        "hourly_start": str(hourly["ts_hour"].min()),
        "hourly_end": str(hourly["ts_hour"].max()),
        "price_stats": stats,
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# International Commodity Prices Report",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Source: Yahoo Finance (yfinance)",
        "",
        f"- Daily rows: `{report['daily_rows']}`",
        f"- Hourly rows: `{report['hourly_rows']}`",
        f"- Daily range: `{report['daily_start']}` → `{report['daily_end']}`",
        "",
        "## Price Stats",
        "",
        "| Kolon | Ortalama | Min | Max | Eksik % |",
        "|---|---:|---:|---:|---:|",
    ]
    for col, s in stats.items():
        lines.append(f"| `{col}` | {s['mean']:.2f} | {s['min']:.2f} | {s['max']:.2f} | {s['missing_pct']:.1f}% |")
    lines += [
        "",
        "## Kolonlar",
        "",
        "- `brent_usd`: Brent ham petrol (USD/barrel)",
        "- `henry_hub_usd`: Henry Hub doğalgaz (USD/MMBtu)",
        "- `ttf_eur_mwh`: TTF Avrupa doğalgaz spot (EUR/MWh)",
        "- `coal_api2_usd`: API2 Rotterdam kömür (USD/ton)",
        "- `*_try`: TCMB kuru ile TRY dönüşümü (varsa)",
        "- `*_lag_7d`, `*_change_7d`, `*_pct_change_7d`, `*_roll_mean_30d`: türetilmiş özellikler",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start, end = _ptf_date_range(args.start_date)
    if args.end_date:
        end = pd.Timestamp(args.end_date).to_pydatetime()
    _log(f"Tarih aralığı: {start.date()} → {end.date()}")

    daily_new = build_daily(start, end)
    daily_new = add_try_equivalents(daily_new)
    daily = merge_with_existing(daily_new)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(RAW_CSV, index=False)
    _log(f"Raw CSV: {RAW_CSV} rows={len(daily)}")

    DAILY_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(DAILY_PATH, index=False)
    _log(f"Daily parquet: {DAILY_PATH} rows={len(daily)}")

    hourly = align_to_hourly(daily, datetime(2020, 1, 1), end)
    hourly.to_parquet(HOURLY_PATH, index=False)
    _log(f"Hourly parquet: {HOURLY_PATH} rows={len(hourly)}")

    write_report(daily, hourly)
    _log(f"Report: {REPORT_MD}")


if __name__ == "__main__":
    main()

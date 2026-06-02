#!/usr/bin/env python3
"""
Fetch plant-level KGUP (UEVÇB) for a date range and build YEKDEM must-run features.

Goal:
  Produce hourly must-run proxies aligned with the DAM curve reconstruction window.

Important semantics:
  EPİAŞ dpp-bulk does not expose a reliable publication timestamp for historical
  point-in-time joins. Therefore the produced must-run features are marked as:
    - strict_point_in_time_safe = False
    - structural_market_proxy = True

This script is resumable, rate-limit aware, and writes partial outputs.
No model training is performed.
"""

from __future__ import annotations

import argparse
import json
from datetime import timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_plant_level_kgup_pipeline import (
    attach_yekdem_mapping,
    build_must_run_features,
    latest_asof_rows,
    plant_name_key,
)
from fetch_plant_level_kgup_archive import (
    RAW_ARCHIVE_ROOT,
    STATE_BASENAME,
    collect_uevcb_map,
    discover_powerplants,
    fetch_day_batch,
    load_state,
    login,
    parse_date,
    save_state,
    utc_now,
    write_json,
)
from market_aware_ptf_pipeline_skeleton import YEKDEM_Registry

PROJECT_ROOT = Path(__file__).resolve().parent

CURVE_GLOB = "reconstructed_weekly_curve_features_*.parquet"
LOAD_FORECAST_PATH = PROJECT_ROOT / "data" / "load_forecast.csv"


def to_naive(ts: pd.Series) -> pd.Series:
    out = pd.to_datetime(ts, errors="coerce")
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out


def build_hour_index(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DatetimeIndex:
    start = start_date.normalize().tz_localize(None)
    end = end_date.normalize().tz_localize(None) + pd.Timedelta(hours=23)
    return pd.date_range(start=start, end=end, freq="h")


def load_load_forecast() -> pd.DataFrame:
    if not LOAD_FORECAST_PATH.exists():
        return pd.DataFrame(columns=["delivery_hour", "load_forecast"])
    load = pd.read_csv(LOAD_FORECAST_PATH)
    load["delivery_hour"] = pd.to_datetime(load["date"], errors="coerce")
    load["delivery_hour"] = to_naive(load["delivery_hour"])
    load["load_forecast"] = pd.to_numeric(load.get("lep"), errors="coerce")
    load = load.dropna(subset=["delivery_hour"]).drop_duplicates("delivery_hour", keep="last")
    return load[["delivery_hour", "load_forecast"]].sort_values("delivery_hour")


def load_curve_hours() -> set[pd.Timestamp]:
    curve_files = sorted((PROJECT_ROOT / "data" / "features").glob(CURVE_GLOB))
    if not curve_files:
        return set()
    frames = []
    for path in curve_files:
        df = pd.read_parquet(path)
        if "delivery_hour" in df.columns:
            frames.append(pd.DataFrame({"delivery_hour": to_naive(df["delivery_hour"])}))
    if not frames:
        return set()
    hours = pd.concat(frames, ignore_index=True)["delivery_hour"].dropna().drop_duplicates()
    return set(pd.to_datetime(hours).to_list())


def enrich_must_run(features: pd.DataFrame, load: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    out["delivery_hour"] = to_naive(out["delivery_hour"])
    load = load.copy()
    load["delivery_hour"] = to_naive(load["delivery_hour"])

    if "load_forecast" not in out.columns:
        out = out.merge(load, on="delivery_hour", how="left")
    else:
        out = out.merge(load, on="delivery_hour", how="left", suffixes=("", "_from_csv"))
        out["load_forecast"] = out["load_forecast"].fillna(out["load_forecast_from_csv"])
        out = out.drop(columns=[c for c in ["load_forecast_from_csv"] if c in out.columns])

    out["must_run_share_of_load"] = out["must_run_supply"] / out["load_forecast"].replace(0, np.nan)
    out["residual_load_after_must_run"] = out["load_forecast"] - out["must_run_supply"]

    out = out.sort_values("delivery_hour").reset_index(drop=True)
    out["must_run_ramp_1h"] = out["must_run_supply"].diff(1)
    out["must_run_ramp_3h"] = out["must_run_supply"].diff(3)

    renew = out[["must_run_wind", "must_run_solar", "must_run_biomass", "must_run_geothermal"]].sum(axis=1)
    out["renewable_must_run_pressure"] = renew / out["load_forecast"].replace(0, np.nan)
    out["hydro_must_run_pressure"] = out["must_run_hydro"] / out["load_forecast"].replace(0, np.nan)
    out["solar_must_run_pressure"] = out["must_run_solar"] / out["load_forecast"].replace(0, np.nan)
    return out


def write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def write_report(md_path: Path, json_path: Path, payload: dict[str, Any]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")

    lines = [
        "# Must-Run Range Coverage",
        "",
        f"- Range: `{payload['start_date']}` → `{payload['end_date']}`",
        f"- Expected hours: `{payload['expected_hours']}`",
        f"- Produced hours: `{payload['produced_hours']}`",
        f"- Curve overlap hours: `{payload['curve_overlap_hours']}`",
        f"- Curve overlap ratio: `{payload['curve_overlap_ratio']}`",
        "",
        "## YEKDEM Matching",
        "",
        f"- Matched plants (by code or normalized name): `{payload['matched_yekdem_plants']}`",
        f"- Unmatched plants: `{payload['unmatched_plants']}`",
        "",
        "## Leakage Status",
        "",
        "- `strict_point_in_time_safe = False`",
        "- `structural_market_proxy = True`",
        "",
        "## Fuel Breakdown (MWh)",
        "",
    ]
    for k, v in payload.get("fuel_breakdown_mwh", {}).items():
        lines.append(f"- `{k}`: `{v}`")
    lines += [
        "",
        "## Missing Hours (sample)",
        "",
    ]
    missing = payload.get("missing_hours_sample", [])
    if missing:
        for x in missing:
            lines.append(f"- `{x}`")
    else:
        lines.append("- None")
    md_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2026-05-18", help="YYYY-MM-DD")
    parser.add_argument("--end-date", default="2026-05-31", help="YYYY-MM-DD")
    parser.add_argument("--sleep-seconds", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-missing", action="store_true")
    args = parser.parse_args()

    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    if end_date < start_date:
        raise ValueError("--end-date must be >= --start-date")

    expected_hours = int(len(build_hour_index(start_date, end_date)))

    run_dir = RAW_ARCHIVE_ROOT / f"{start_date.date().isoformat()}_{end_date.date().isoformat()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / f"{STATE_BASENAME}_{start_date.date().isoformat()}_{end_date.date().isoformat()}.json"
    lock_path = run_dir / ".lock_must_run"
    if lock_path.exists():
        raise RuntimeError(f"Lock file exists (another run may be active): {lock_path}")
    lock_path.write_text(f"locked_at={utc_now().isoformat()}\n")

    feature_out = PROJECT_ROOT / "data" / "features" / f"must_run_supply_features_{args.start_date}_{args.end_date}.parquet"
    report_md = PROJECT_ROOT / "reports" / f"must_run_range_coverage_{args.start_date}_{args.end_date}.md"
    report_json = PROJECT_ROOT / "reports" / f"must_run_range_coverage_{args.start_date}_{args.end_date}.json"

    try:
        state = load_state(state_path) if args.resume else {
            "completed_dates": [],
            "completed_chunks": [],
            "discovered_plants": 0,
            "discovered_uevcbs": 0,
            "last_updated_at": None,
        }
        completed_dates = set(state.get("completed_dates", []))
        completed_chunks = set(state.get("completed_chunks", []))

        registry_files = YEKDEM_Registry.discover_files(PROJECT_ROOT)
        registry = YEKDEM_Registry(registry_files)

        tgt = login()
        plants = discover_powerplants(tgt, start_date, end_date)
        if not plants:
            raise RuntimeError("No power plants discovered.")
        uevcb_map, discovery_debug = collect_uevcb_map(tgt, plants, start_date, max_plants=0)
        if not uevcb_map:
            raise RuntimeError("No UEVÇB map discovered.")

        write_json(run_dir / "powerplant_list.json", [p.__dict__ for p in plants])
        write_json(run_dir / "uevcb_discovery_debug.json", discovery_debug)
        write_json(run_dir / "uevcb_map.json", {k: v.__dict__ for k, v in uevcb_map.items()})

        # Fetch only YEKDEM-relevant UEVÇB ids to keep the range run tractable.
        years = pd.date_range(start=start_date.normalize(), end=end_date.normalize(), freq="D").year.unique().tolist()
        registry_ids: set[str] = set()
        for y in years:
            registry_ids |= set(registry.plant_ids_for_year(int(y)))

        candidate_uevcb_ids = [
            int(k)
            for k, plant in uevcb_map.items()
            if plant and plant.entsoe_code and str(plant.entsoe_code).strip() in registry_ids
        ]
        selected_by = "entsoe_code"
        # If ENTSO-E discovery is sparse, expand scope using normalized plant name keys.
        if len(candidate_uevcb_ids) < 50:
            registry_name_keys = set(registry.registry[registry.registry_name_col].map(plant_name_key).dropna().astype(str))
            by_name = [
                int(k)
                for k, plant in uevcb_map.items()
                if plant and plant.plant_name and plant_name_key(plant.plant_name) in registry_name_keys
            ]
            candidate_uevcb_ids = list(set(candidate_uevcb_ids) | set(by_name))
            selected_by = "entsoe_code+name_key"
        if not candidate_uevcb_ids:
            # Last resort: if both fail, run full universe (slow).
            candidate_uevcb_ids = [int(k) for k in uevcb_map.keys()]
            selected_by = "full_universe_fallback"
        uevcb_ids = sorted(set(candidate_uevcb_ids))
        write_json(
            run_dir / "uevcb_fetch_scope.json",
            {
                "registry_years": [int(x) for x in years],
                "registry_entsoe_codes_count": int(len(registry_ids)),
                "uevcb_ids_total_discovered": int(len(uevcb_map)),
                "uevcb_ids_selected_for_fetch": int(len(uevcb_ids)),
                "selected_by": selected_by,
                "note": "Selected UEVÇB ids are those whose discovered ENTSO-E codes appear in the YEKDEM registry for the given year(s).",
            },
        )
        requested_chunks = 0
        successful_chunks = 0
        failed_chunks: list[dict[str, Any]] = []

        current = start_date.normalize()
        end = end_date.normalize()
        while current <= end:
            day_key = current.date().isoformat()
            day_dir = run_dir / day_key
            day_dir.mkdir(parents=True, exist_ok=True)

            for chunk_idx, i in enumerate(range(0, len(uevcb_ids), args.batch_size)):
                chunk_ids = uevcb_ids[i : i + args.batch_size]
                chunk_key = f"{day_key}::{chunk_idx:04d}::{','.join(map(str, chunk_ids))}"
                if args.only_missing and chunk_key in completed_chunks:
                    continue
                requested_chunks += 1
                try:
                    rows, raw_path, meta_path, normalized_path = fetch_day_batch(
                        tgt=tgt,
                        day=current,
                        uevcb_ids=chunk_ids,
                        uevcb_to_plant=uevcb_map,
                        out_dir=day_dir,
                        batch_idx=chunk_idx,
                        sleep_seconds=args.sleep_seconds,
                    )
                    if rows > 0:
                        successful_chunks += 1
                    else:
                        failed_chunks.append(
                            {
                                "day": day_key,
                                "chunk_idx": chunk_idx,
                                "chunk_ids": chunk_ids,
                                "reason": f"zero rows or non-200; raw={raw_path.name if raw_path else None} meta={meta_path.name if meta_path else None}",
                            }
                        )
                    completed_chunks.add(chunk_key)
                    save_state(
                        state_path,
                        {
                            **state,
                            "completed_dates": sorted(completed_dates | {day_key}),
                            "completed_chunks": sorted(completed_chunks),
                            "discovered_plants": len(plants),
                            "discovered_uevcbs": len(uevcb_map),
                            "requested_chunks": requested_chunks,
                            "successful_chunks": successful_chunks,
                            "failed_chunks_count": len(failed_chunks),
                        },
                    )
                except Exception as exc:
                    failed_chunks.append(
                        {"day": day_key, "chunk_idx": chunk_idx, "chunk_ids": chunk_ids, "error": str(exc)}
                    )
                    save_state(
                        state_path,
                        {
                            **state,
                            "completed_dates": sorted(completed_dates),
                            "completed_chunks": sorted(completed_chunks),
                            "discovered_plants": len(plants),
                            "discovered_uevcbs": len(uevcb_map),
                            "requested_chunks": requested_chunks,
                            "successful_chunks": successful_chunks,
                            "failed_chunks_count": len(failed_chunks),
                        },
                    )

            completed_dates.add(day_key)
            save_state(
                state_path,
                {
                    **state,
                    "completed_dates": sorted(completed_dates),
                    "completed_chunks": sorted(completed_chunks),
                    "discovered_plants": len(plants),
                    "discovered_uevcbs": len(uevcb_map),
                    "requested_chunks": requested_chunks,
                    "successful_chunks": successful_chunks,
                    "failed_chunks_count": len(failed_chunks),
                },
            )
            current += pd.Timedelta(days=1)

        chunk_csvs = sorted(run_dir.rglob("chunk_*.csv"))
        if not chunk_csvs:
            raise RuntimeError(f"No normalized chunk CSVs found under {run_dir}")
        archive = pd.concat((pd.read_csv(p) for p in chunk_csvs), ignore_index=True)
        archive["delivery_hour"] = pd.to_datetime(archive["delivery_hour"], errors="coerce")
        archive["delivery_hour"] = archive["delivery_hour"].dt.tz_localize(None) if getattr(archive["delivery_hour"].dt, "tz", None) is not None else archive["delivery_hour"]
        archive["forecast_timestamp"] = pd.to_datetime(archive["forecast_timestamp"], errors="coerce", utc=True)
        if "publication_timestamp" in archive.columns:
            archive["publication_timestamp"] = pd.to_datetime(archive["publication_timestamp"], errors="coerce", utc=True)
        if "archive_snapshot_timestamp" not in archive.columns:
            archive["archive_snapshot_timestamp"] = archive["forecast_timestamp"]
        archive["archive_snapshot_timestamp"] = pd.to_datetime(archive["archive_snapshot_timestamp"], errors="coerce", utc=True).fillna(archive["forecast_timestamp"])

        mapped, mapping_audit = attach_yekdem_mapping(archive, registry)
        latest, leakage_audit = latest_asof_rows(mapped)

        # Build must-run features (structural proxy due missing publication ts).
        features = build_must_run_features(latest)
        if not features.empty:
            # Enforce flags as requested.
            features["strict_point_in_time_safe"] = False
            features["structural_market_proxy"] = True
            load = load_load_forecast()
            features = enrich_must_run(features, load)

        # Conform to expected hour index and report missing hours.
        hours = build_hour_index(start_date, end_date)
        produced_hours = int(features["delivery_hour"].nunique()) if not features.empty else 0
        produced_set = set(pd.to_datetime(features["delivery_hour"]).to_list()) if not features.empty else set()
        missing_hours = [h for h in hours.to_list() if h not in produced_set]

        curve_hours = load_curve_hours()
        overlap = len([h for h in hours.to_list() if h in curve_hours and h in produced_set])
        overlap_ratio = None if produced_hours == 0 else float(overlap / expected_hours)

        if not features.empty:
            write_parquet_atomic(feature_out, features)
        else:
            write_parquet_atomic(feature_out, pd.DataFrame())

        fuel_breakdown = {}
        if not features.empty:
            for col in [
                "must_run_supply",
                "must_run_wind",
                "must_run_solar",
                "must_run_hydro",
                "must_run_biomass",
                "must_run_geothermal",
            ]:
                if col in features.columns:
                    fuel_breakdown[col] = float(np.nan_to_num(features[col]).sum())

        report = {
            "generated_at": utc_now().isoformat(),
            "start_date": start_date.date().isoformat(),
            "end_date": end_date.date().isoformat(),
            "expected_hours": expected_hours,
            "produced_hours": produced_hours,
            "missing_hours_count": int(len(missing_hours)),
            "missing_hours_sample": [str(x) for x in missing_hours[:50]],
            "curve_overlap_hours": int(overlap),
            "curve_overlap_ratio": overlap_ratio,
            "matched_yekdem_plants": int(mapping_audit.get("matched_plants", 0)),
            "unmatched_plants": int(mapping_audit.get("unmatched_plants", 0)),
            "match_method_counts": mapping_audit.get("match_method_counts", {}),
            "leakage_status": {
                "strict_point_in_time_safe": False,
                "structural_market_proxy": True,
                "note": "dpp-bulk publication timestamp missing; features are structural proxies.",
            },
            "leakage_audit": leakage_audit,
            "requested_chunks": requested_chunks,
            "successful_chunks": successful_chunks,
            "failed_chunks_count": len(failed_chunks),
            "failed_chunks_sample": failed_chunks[:20],
            "fuel_breakdown_mwh": fuel_breakdown,
            "outputs": {
                "feature_parquet": str(feature_out.relative_to(PROJECT_ROOT)),
                "raw_archive_dir": str(run_dir.relative_to(PROJECT_ROOT)),
                "state_path": str(state_path.relative_to(PROJECT_ROOT)),
            },
        }
        write_report(report_md, report_json, report)

        print(f"Wrote {feature_out}")
        print(f"Wrote {report_md}")
        print(f"Wrote {report_json}")
    finally:
        try:
            lock_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass


if __name__ == "__main__":
    main()

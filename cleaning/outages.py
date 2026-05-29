"""Expand outage events to hourly aggregates."""

from __future__ import annotations

from typing import Any

import pandas as pd

from cleaning.config import TIMEZONE
from cleaning.datetime_utils import parse_event_timestamps

FAULT_TYPE_ID = 0
MAINT_TYPE_ID = 2


def _expand_events_to_hours(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["event_start"] = parse_event_timestamps(working["caseStartDate"])
    working["event_end"] = parse_event_timestamps(working["caseEndDate"])
    working = working.dropna(subset=["event_start", "event_end"])

    working["start_hour"] = working["event_start"].dt.floor("h")
    working["end_hour"] = working["event_end"].dt.floor("h")
    working.loc[working["end_hour"] < working["start_hour"], "end_hour"] = working[
        "start_hour"
    ]

    hour_lists = [
        pd.date_range(row.start_hour, row.end_hour, freq="h", tz=TIMEZONE)
        for row in working.itertuples(index=False)
    ]
    working["ts_hour"] = hour_lists
    return working.explode("ts_hour", ignore_index=True)


def _aggregate_outages_hourly(expanded: pd.DataFrame) -> pd.DataFrame:
    fault = expanded[expanded["messageTypeId"] == FAULT_TYPE_ID]
    maint = expanded[expanded["messageTypeId"] == MAINT_TYPE_ID]

    fault_agg = (
        fault.groupby("ts_hour", as_index=False)
        .agg(
            outage_fault_event_count=("id", "count"),
            outage_fault_mw_loss_sum=("totalFaultCausedPowerLoss", "sum"),
            outage_fault_mw_loss_max=("maxFaultCausedPowerLoss", "max"),
            outage_fault_operator_power_sum=("operatorPower", "sum"),
        )
    )

    maint_agg = (
        maint.groupby("ts_hour", as_index=False)
        .agg(
            outage_maint_event_count=("id", "count"),
            outage_maint_capacity_sum=("capacityAtCaseTime", "sum"),
            outage_maint_operator_power_sum=("operatorPower", "sum"),
        )
    )

    event_rows = (
        expanded.groupby("ts_hour", as_index=False)
        .size()
        .rename(columns={"size": "outage_event_rows"})
    )

    hourly = event_rows.merge(fault_agg, on="ts_hour", how="left")
    hourly = hourly.merge(maint_agg, on="ts_hour", how="left")

    for col in (
        "outage_fault_event_count",
        "outage_maint_event_count",
        "outage_event_rows",
    ):
        hourly[col] = hourly[col].fillna(0).astype(int)

    return hourly


def clean_outages_csv(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    stats: dict[str, Any] = {"rows_in": len(df)}
    working = df.copy()

    if {"messageTypeId", "id"}.issubset(working.columns):
        stats["duplicate_key_rows"] = int(
            working.duplicated(subset=["messageTypeId", "id"], keep=False).sum()
        )
        working = working.drop_duplicates(subset=["messageTypeId", "id"], keep="last")

    expanded = _expand_events_to_hours(working)
    stats["rows_after_expand"] = len(expanded)

    hourly = _aggregate_outages_hourly(expanded)
    hourly = hourly.sort_values("ts_hour").reset_index(drop=True)

    stats["rows_out"] = len(hourly)
    stats["ts_min"] = str(hourly["ts_hour"].min()) if len(hourly) else None
    stats["ts_max"] = str(hourly["ts_hour"].max()) if len(hourly) else None
    stats["numeric_na_pct"] = {
        c: float(hourly[c].isna().mean() * 100)
        for c in hourly.columns
        if c != "ts_hour" and hourly[c].isna().any()
    }

    return hourly, stats

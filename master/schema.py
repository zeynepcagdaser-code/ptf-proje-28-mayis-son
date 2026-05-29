"""Master dataset file mapping, prefixes, and column availability metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Availability = Literal[
    "planned",
    "forecast",
    "realized",
    "balancing",
    "outage_event",
    "metadata",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DATA_DIR = PROJECT_ROOT / "data" / "clean"
MASTER_DATA_DIR = PROJECT_ROOT / "data" / "master"
REPORTS_DIR = PROJECT_ROOT / "reports"

MASTER_OUTPUT = MASTER_DATA_DIR / "master_hourly_v1.parquet"
SPINE_FILE = CLEAN_DATA_DIR / "ptf_hourly.parquet"
JOIN_KEY = "ts_hour"


@dataclass(frozen=True)
class DatasetSpec:
    """One cleaned parquet source joined onto the PTF spine."""

    name: str
    filename: str
    prefix: str
    default_availability: Availability
    column_availability: dict[str, Availability] = field(default_factory=dict)
    skip_prefix_if_starts_with: str | None = None

    @property
    def path(self) -> Path:
        return CLEAN_DATA_DIR / self.filename


def _col(name: str, availability: Availability | None = None) -> tuple[str, Availability | None]:
    return (name, availability)


# Join order after spine (PTF is loaded separately as spine).
JOIN_DATASETS: list[DatasetSpec] = [
    DatasetSpec(
        name="kgup",
        filename="kgup_hourly.parquet",
        prefix="kgup_",
        default_availability="planned",
        column_availability=dict(
            [_col("source_type", "metadata")]
        ),
    ),
    DatasetSpec(
        name="load_forecast",
        filename="load_forecast_hourly.parquet",
        prefix="load_",
        default_availability="forecast",
    ),
    DatasetSpec(
        name="realtime_generation",
        filename="realtime_generation_hourly.parquet",
        prefix="gen_",
        default_availability="realized",
        column_availability=dict([_col("was_sun_clipped", "metadata")]),
    ),
    DatasetSpec(
        name="real_consumption",
        filename="real_consumption_hourly.parquet",
        prefix="cons_",
        default_availability="realized",
    ),
    DatasetSpec(
        name="smf",
        filename="smf_hourly.parquet",
        prefix="smf_",
        default_availability="balancing",
        column_availability=dict(
            [
                _col("is_systemMarginalPrice_zero", "metadata"),
                _col("is_systemMarginalPrice_capped", "metadata"),
            ]
        ),
    ),
    DatasetSpec(
        name="yal_yat",
        filename="yal_yat_hourly.parquet",
        prefix="yal_yat_",
        default_availability="balancing",
    ),
    DatasetSpec(
        name="wind",
        filename="wind_hourly.parquet",
        prefix="wind_",
        default_availability="forecast",
        column_availability={
            "generation_mean": "realized",
            "generation_min": "realized",
            "generation_max": "realized",
            "was_generation_clipped": "metadata",
            "is_partial_hour": "metadata",
            "wind_interval_count": "metadata",
            "quarter1_mean": "forecast",
            "quarter2_mean": "forecast",
            "quarter3_mean": "forecast",
            "quarter4_mean": "forecast",
            "forecast_mean": "forecast",
            "forecast_min": "forecast",
            "forecast_max": "forecast",
            "forecast_std": "forecast",
        },
    ),
    DatasetSpec(
        name="outages",
        filename="outages_hourly.parquet",
        prefix="outage_",
        default_availability="outage_event",
        skip_prefix_if_starts_with="outage_",
    ),
]

SPINE_SPEC = DatasetSpec(
    name="ptf",
    filename="ptf_hourly.parquet",
    prefix="ptf_",
    default_availability="realized",
    column_availability={
        "price": "realized",
        "priceUsd": "realized",
        "priceEur": "realized",
        "is_price_zero": "metadata",
        "is_price_capped": "metadata",
        "is_priceUsd_zero": "metadata",
        "is_priceUsd_capped": "metadata",
        "is_priceEur_zero": "metadata",
        "is_priceEur_capped": "metadata",
    },
)


def prefixed_name(spec: DatasetSpec, column: str) -> str:
    if spec.skip_prefix_if_starts_with and column.startswith(spec.skip_prefix_if_starts_with):
        return column
    if column.startswith(spec.prefix):
        return column
    return f"{spec.prefix}{column}"


def column_availability(spec: DatasetSpec, column: str) -> Availability:
    return spec.column_availability.get(column, spec.default_availability)


def build_column_registry(
    spine_columns: list[str],
    joined_specs: list[tuple[DatasetSpec, list[str]]],
) -> dict[str, Availability]:
    """Map final master column name -> availability label."""
    registry: dict[str, Availability] = {JOIN_KEY: "metadata"}

    for col in spine_columns:
        if col == JOIN_KEY:
            continue
        master_col = prefixed_name(SPINE_SPEC, col)
        registry[master_col] = column_availability(SPINE_SPEC, col)

    for spec, columns in joined_specs:
        for col in columns:
            if col == JOIN_KEY:
                continue
            master_col = prefixed_name(spec, col)
            registry[master_col] = column_availability(spec, col)

    return registry

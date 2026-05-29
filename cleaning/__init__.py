"""Data cleaning pipeline: raw CSV → hourly parquet in data/clean/."""

from cleaning.pipeline import run_pipeline

__all__ = ["run_pipeline"]

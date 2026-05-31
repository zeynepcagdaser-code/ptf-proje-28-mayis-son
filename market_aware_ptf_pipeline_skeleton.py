#!/usr/bin/env python3
"""
Leakage-safe, regime-aware PTF forecasting architecture skeleton.

This file is intentionally a design scaffold, not a production training script.
It shows how to isolate YEKDEM must-run supply, enforce as-of joins, route
hours through regime-aware experts, and audit every prediction input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd


REGIME_ORDER = ["zero_pressure", "normal", "tight", "spike"]


@dataclass(frozen=True)
class AuditRecord:
    delivery_hour: pd.Timestamp
    source_name: str
    max_publication_timestamp: pd.Timestamp | None
    rows_used: int
    status: str
    detail: str


@dataclass
class AuditLog:
    records: list[AuditRecord] = field(default_factory=list)

    def add(
        self,
        delivery_hour: pd.Timestamp,
        source_name: str,
        max_publication_timestamp: pd.Timestamp | None,
        rows_used: int,
        status: str,
        detail: str,
    ) -> None:
        self.records.append(
            AuditRecord(
                delivery_hour=delivery_hour,
                source_name=source_name,
                max_publication_timestamp=max_publication_timestamp,
                rows_used=rows_used,
                status=status,
                detail=detail,
            )
        )

    def assert_clean(self) -> None:
        bad = [record for record in self.records if record.status != "pass"]
        if bad:
            messages = [f"{r.source_name}: {r.detail}" for r in bad]
            raise ValueError("Leakage audit failed: " + " | ".join(messages))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([record.__dict__ for record in self.records])


class AsOfJoin:
    """
    Point-in-time join helper.

    Rule:
        A row may be used only if publication_timestamp <= forecast_as_of.

    forecast_as_of is usually the timestamp when the prediction is made. For
    day-ahead PTF it should be before delivery_hour and must match the research
    protocol, not the final settlement timestamp.
    """

    def __init__(self, audit_log: AuditLog):
        self.audit_log = audit_log

    def latest_available(
        self,
        data: pd.DataFrame,
        delivery_hour: pd.Timestamp,
        forecast_as_of: pd.Timestamp,
        source_name: str,
        delivery_col: str = "delivery_hour",
        publication_col: str = "publication_timestamp",
    ) -> pd.DataFrame:
        frame = data.copy()
        frame[delivery_col] = pd.to_datetime(frame[delivery_col], errors="coerce")
        frame[publication_col] = pd.to_datetime(frame[publication_col], errors="coerce")

        eligible = frame[
            (frame[delivery_col] == delivery_hour)
            & (frame[publication_col].notna())
            & (frame[publication_col] <= forecast_as_of)
        ].copy()

        if eligible.empty:
            self.audit_log.add(
                delivery_hour=delivery_hour,
                source_name=source_name,
                max_publication_timestamp=None,
                rows_used=0,
                status="missing",
                detail="No point-in-time eligible rows found.",
            )
            return eligible

        max_pub = eligible[publication_col].max()
        latest = eligible[eligible[publication_col] == max_pub].copy()
        self.audit_log.add(
            delivery_hour=delivery_hour,
            source_name=source_name,
            max_publication_timestamp=max_pub,
            rows_used=len(latest),
            status="pass",
            detail=f"Using rows published as of {max_pub}.",
        )
        return latest


def _slug(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text


class YEKDEM_Registry:
    """
    Registry for isolating YEKDEM plants from plant-level KGUP.

    Expected Excel columns are project-specific. Configure registry_id_col and
    kgup_plant_id_col to the stable plant identifier available in both sources.
    """

    DEFAULT_ID_CANDIDATES = [
        "ENTSO-E Kodu [1]",
        "ENTSO-E Kodu",
        "entsoe_code",
        "eic",
        "plant_id",
    ]
    DEFAULT_NAME_CANDIDATES = ["Tesis Adı", "plant_name", "name"]
    DEFAULT_CAPACITY_CANDIDATES = ["YEKDEM'e Esas Güç (MWe) [8]", "YEKDEM'e Esas Güç (MWe)"]
    DEFAULT_SOURCE_CANDIDATES = ["Ana Kaynak Türü [2]", "Ana Kaynak Türü"]

    def __init__(
        self,
        excel_path: str | Path | list[str | Path],
        registry_id_col: str | None = None,
        registry_name_col: str | None = None,
        active_col: str | None = None,
    ):
        self.excel_paths = [Path(path) for path in (excel_path if isinstance(excel_path, list) else [excel_path])]
        self.registry_id_col = registry_id_col
        self.registry_name_col = registry_name_col
        self.active_col = active_col
        self.registry = self._load_registry()

    @staticmethod
    def discover_files(root: str | Path = ".") -> list[Path]:
        root_path = Path(root)
        candidates = sorted(
            path
            for path in root_path.glob("*")
            if path.suffix.lower() in {".xls", ".xlsx", ".xlsm"}
        )
        files: list[Path] = []
        for path in candidates:
            if "yek" in path.name.lower():
                files.append(path)
                continue
            try:
                sheet_names = pd.ExcelFile(path).sheet_names
            except Exception:
                continue
            if any("yek" in sheet.lower() for sheet in sheet_names):
                files.append(path)
        return sorted(set(files))

    @staticmethod
    def _detect_col(columns: pd.Index, candidates: list[str]) -> str | None:
        direct = {str(col): str(col) for col in columns}
        for candidate in candidates:
            if candidate in direct:
                return candidate
        slug_map = {_slug(col): str(col) for col in columns}
        for candidate in candidates:
            found = slug_map.get(_slug(candidate))
            if found:
                return found
        return None

    @staticmethod
    def _extract_year(path: Path, sheet_name: str | None = None) -> int | None:
        text = f"{path.name} {sheet_name or ''}"
        match = re.search(r"(20\d{2})", text)
        return int(match.group(1)) if match else None

    def _read_one_excel(self, path: Path) -> pd.DataFrame:
        sheets = pd.ExcelFile(path).sheet_names
        frames = []
        for sheet in sheets:
            # These EPİAŞ files have a title row followed by the real header row.
            frame = pd.read_excel(path, sheet_name=sheet, header=1)
            frame = frame.dropna(how="all")
            frame["registry_year"] = self._extract_year(path, sheet)
            frame["registry_source_file"] = path.name
            frame["registry_source_sheet"] = sheet
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def _load_registry(self) -> pd.DataFrame:
        if not self.excel_paths:
            raise ValueError("No YEKDEM registry Excel files provided.")
        missing = [path for path in self.excel_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing YEKDEM registry files: {missing}")

        registry = pd.concat([self._read_one_excel(path) for path in self.excel_paths], ignore_index=True)
        id_col = self.registry_id_col or self._detect_col(registry.columns, self.DEFAULT_ID_CANDIDATES)
        if id_col is None:
            raise ValueError(
                "YEKDEM registry missing plant id / ENTSO-E column. "
                f"Available columns: {list(registry.columns)}"
            )
        self.registry_id_col = id_col
        self.registry_name_col = self.registry_name_col or self._detect_col(
            registry.columns, self.DEFAULT_NAME_CANDIDATES
        )
        registry = registry.copy()
        registry[self.registry_id_col] = registry[self.registry_id_col].astype(str).str.strip()
        registry = registry.dropna(subset=[self.registry_id_col]).drop_duplicates(
            subset=[self.registry_id_col, "registry_year"], keep="last"
        )
        registry = registry[registry[self.registry_id_col].str.lower() != "nan"]
        if self.active_col and self.active_col in registry.columns:
            registry = registry[registry[self.active_col].astype(bool)]
        return registry

    @property
    def plant_ids(self) -> set[str]:
        return set(self.registry[self.registry_id_col].astype(str))

    def plant_ids_for_year(self, year: int | None) -> set[str]:
        if year is None or "registry_year" not in self.registry.columns:
            return self.plant_ids
        subset = self.registry[self.registry["registry_year"] == year]
        if subset.empty:
            eligible = self.registry[self.registry["registry_year"].fillna(0) <= year]
            if eligible.empty:
                return self.plant_ids
            latest_year = eligible["registry_year"].max()
            subset = eligible[eligible["registry_year"] == latest_year]
        return set(subset[self.registry_id_col].astype(str))

    def filter_kgup(
        self,
        kgup_plant_level: pd.DataFrame,
        kgup_plant_id_col: str = "ENTSO-E Kodu",
        delivery_col: str = "delivery_hour",
    ) -> pd.DataFrame:
        kgup = kgup_plant_level.copy()
        if kgup_plant_id_col not in kgup.columns:
            raise ValueError(f"KGUP frame missing plant id column `{kgup_plant_id_col}`.")
        kgup[kgup_plant_id_col] = kgup[kgup_plant_id_col].astype(str).str.strip()
        if delivery_col in kgup.columns:
            years = pd.to_datetime(kgup[delivery_col], errors="coerce").dt.year
            masks = []
            for year, index in years.groupby(years).groups.items():
                masks.append(
                    pd.Series(
                        kgup.loc[index, kgup_plant_id_col].isin(self.plant_ids_for_year(int(year))),
                        index=index,
                    )
                )
            mask = pd.concat(masks).reindex(kgup.index).fillna(False) if masks else pd.Series(False, index=kgup.index)
        else:
            mask = kgup[kgup_plant_id_col].isin(self.plant_ids)
        return kgup[mask].copy()

    def hourly_must_run_supply(
        self,
        kgup_plant_level: pd.DataFrame,
        delivery_col: str = "delivery_hour",
        value_col: str = "kgup_mw",
        kgup_plant_id_col: str = "ENTSO-E Kodu",
    ) -> pd.DataFrame:
        yekdem_kgup = self.filter_kgup(kgup_plant_level, kgup_plant_id_col, delivery_col)
        yekdem_kgup[delivery_col] = pd.to_datetime(yekdem_kgup[delivery_col], errors="coerce")
        hourly = (
            yekdem_kgup.dropna(subset=[delivery_col])
            .groupby(delivery_col, as_index=False)[value_col]
            .sum()
            .rename(columns={delivery_col: "delivery_hour", value_col: "must_run_supply"})
        )
        return hourly.sort_values("delivery_hour")

    def summary(self) -> pd.DataFrame:
        rows = []
        for year, group in self.registry.groupby("registry_year", dropna=False):
            rows.append(
                {
                    "registry_year": None if pd.isna(year) else int(year),
                    "plant_count": int(group[self.registry_id_col].nunique()),
                    "source_files": ", ".join(sorted(group["registry_source_file"].dropna().unique())),
                }
            )
        return pd.DataFrame(rows).sort_values("registry_year")


class ProbabilisticModel(Protocol):
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        ...


class RegressionModel(Protocol):
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        ...


class RegimeClassifier:
    """Gating network wrapper."""

    def __init__(self, model: ProbabilisticModel | None = None):
        self.model = model

    def predict_regime_probabilities(self, features: pd.DataFrame) -> pd.Series:
        if self.model is None:
            residual_load = float(features["residual_load"].iloc[0])
            must_run = float(features["must_run_supply"].iloc[0])
            load = float(features["load_forecast"].iloc[0])
            renewable_pressure = must_run / max(load, 1.0)
            spike_pressure = max(residual_load - 35000, 0) / 15000
            tight_pressure = max(residual_load - 25000, 0) / 15000
            raw = pd.Series(
                {
                    "zero_pressure": renewable_pressure,
                    "normal": 1.0,
                    "tight": tight_pressure,
                    "spike": spike_pressure,
                }
            ).clip(lower=0.01)
            return raw / raw.sum()

        proba = self.model.predict_proba(features)[0]
        return pd.Series(proba, index=REGIME_ORDER).clip(lower=0)


class ZeroPressureExpert:
    def predict(self, features: pd.DataFrame) -> float:
        must_run = float(features["must_run_supply"].iloc[0])
        load = float(features["load_forecast"].iloc[0])
        pressure = must_run / max(load, 1.0)
        return float(np.clip(50 * (1.05 - pressure), 0, 250))


class NormalTightExpert:
    def __init__(self, model: RegressionModel | None = None):
        self.model = model

    def predict(self, features: pd.DataFrame) -> float:
        if self.model is not None:
            return float(self.model.predict(features)[0])
        residual_load = float(features["residual_load"].iloc[0])
        return float(np.clip(500 + 0.06 * max(residual_load - 20000, 0), 0, 4000))


class SpikeExpert:
    def __init__(self, model: RegressionModel | None = None):
        self.model = model

    def score_spike_risk(self, features: pd.DataFrame) -> float:
        ramp = float(features.get("residual_load_ramp", pd.Series([0])).iloc[0])
        outage = float(features.get("outage_stress_index", pd.Series([0])).iloc[0])
        risk = 1 / (1 + np.exp(-(ramp / 3000 + outage * 3 - 2)))
        return float(np.clip(risk, 0, 1))

    def predict(self, features: pd.DataFrame) -> float:
        if self.model is not None:
            return float(self.model.predict(features)[0])
        risk = self.score_spike_risk(features)
        return float(2500 + 2000 * risk)


class PTFRegimeAwarePipeline:
    def __init__(
        self,
        yekdem_registry: YEKDEM_Registry,
        regime_classifier: RegimeClassifier,
        zero_expert: ZeroPressureExpert,
        normal_tight_expert: NormalTightExpert,
        spike_expert: SpikeExpert,
    ):
        self.yekdem_registry = yekdem_registry
        self.regime_classifier = regime_classifier
        self.zero_expert = zero_expert
        self.normal_tight_expert = normal_tight_expert
        self.spike_expert = spike_expert

    def build_features(
        self,
        delivery_hour: pd.Timestamp,
        forecast_as_of: pd.Timestamp,
        kgup_plant_level: pd.DataFrame,
        load_forecast: pd.DataFrame,
        outages: pd.DataFrame,
        audit_log: AuditLog,
    ) -> pd.DataFrame:
        asof = AsOfJoin(audit_log)

        must_run = self.yekdem_registry.hourly_must_run_supply(kgup_plant_level)
        kgup_asof = asof.latest_available(
            data=must_run.assign(publication_timestamp=forecast_as_of),
            delivery_hour=delivery_hour,
            forecast_as_of=forecast_as_of,
            source_name="yekdem_must_run_supply",
        )
        load_asof = asof.latest_available(
            data=load_forecast,
            delivery_hour=delivery_hour,
            forecast_as_of=forecast_as_of,
            source_name="load_forecast",
        )
        outage_asof = asof.latest_available(
            data=outages,
            delivery_hour=delivery_hour,
            forecast_as_of=forecast_as_of,
            source_name="outages",
        )

        must_run_supply = float(kgup_asof["must_run_supply"].sum()) if not kgup_asof.empty else 0.0
        load_value = float(load_asof["load_forecast"].iloc[-1]) if not load_asof.empty else np.nan
        outage_stress = (
            float(outage_asof["outage_stress_index"].max()) if not outage_asof.empty else 0.0
        )
        residual_load = load_value - must_run_supply

        return pd.DataFrame(
            [
                {
                    "delivery_hour": delivery_hour,
                    "forecast_as_of": forecast_as_of,
                    "must_run_supply": must_run_supply,
                    "load_forecast": load_value,
                    "residual_load": residual_load,
                    "residual_load_ramp": np.nan,
                    "outage_stress_index": outage_stress,
                }
            ]
        )

    def predict(
        self,
        delivery_hour: pd.Timestamp,
        forecast_as_of: pd.Timestamp,
        kgup_plant_level: pd.DataFrame,
        load_forecast: pd.DataFrame,
        outages: pd.DataFrame,
    ) -> tuple[float, pd.Series, AuditLog]:
        audit_log = AuditLog()
        features = self.build_features(
            delivery_hour=delivery_hour,
            forecast_as_of=forecast_as_of,
            kgup_plant_level=kgup_plant_level,
            load_forecast=load_forecast,
            outages=outages,
            audit_log=audit_log,
        )
        audit_log.assert_clean()

        regime_probs = self.regime_classifier.predict_regime_probabilities(features)
        expert_preds = pd.Series(
            {
                "zero_pressure": self.zero_expert.predict(features),
                "normal": self.normal_tight_expert.predict(features),
                "tight": self.normal_tight_expert.predict(features),
                "spike": self.spike_expert.predict(features),
            }
        )
        final_prediction = float((regime_probs * expert_preds).sum())
        return final_prediction, regime_probs, audit_log


def example_usage() -> None:
    registry = YEKDEM_Registry("Nihai_YEKDEM_Santral_Listesi.xlsx", registry_id_col="plant_id")
    pipeline = PTFRegimeAwarePipeline(
        yekdem_registry=registry,
        regime_classifier=RegimeClassifier(),
        zero_expert=ZeroPressureExpert(),
        normal_tight_expert=NormalTightExpert(),
        spike_expert=SpikeExpert(),
    )
    _ = pipeline


if __name__ == "__main__":
    print("This is an architecture skeleton. Import the classes into a training/inference script.")

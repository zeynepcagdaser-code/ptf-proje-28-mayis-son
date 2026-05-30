#!/usr/bin/env python3
"""Data discovery audit: kesinleşmemiş (interim) PTF availability (reports only)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PTF_CSV = DATA_DIR / "ptf_dataset.csv"
UPDATE_SCRIPT = PROJECT_ROOT / "update_dataset.py"
MASTER_PARQUET = DATA_DIR / "master" / "master_hourly_v1.parquet"
CLEAN_PTF = DATA_DIR / "clean" / "ptf_hourly.parquet"

METRICS_JSON = PROJECT_ROOT / "reports" / "unfinalized_ptf_data_audit.json"
METRICS_MD = PROJECT_ROOT / "reports" / "unfinalized_ptf_data_audit.md"

EPIAS_MCP_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/markets/dam/data/mcp"
EPIAS_INTERIM_MCP_URL = (
    "https://seffaflik.epias.com.tr/electricity-service/v1/markets/dam/data/interim-mcp"
)
EPIAS_INTERIM_STATUS_URL = (
    "https://seffaflik.epias.com.tr/electricity-service/v1/markets/dam/data/interim-mcp-published-status"
)

KEYWORDS_SCANNED = [
    "mcp",
    "market clearing price",
    "dam",
    "day ahead market",
    "provisional",
    "unfinalized",
    "finalized",
    "kesinleşmemiş",
    "kesinleşmiş",
    "piyasa takas fiyatı",
    "interim",
    "k.ptf",
]


def scan_data_csv_files() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        cols = pd.read_csv(path, nrows=0).columns.tolist()
        lower_cols = [c.lower() for c in cols]
        hits = [
            c
            for c in cols
            if any(
                k in c.lower()
                for k in [
                    "ptf",
                    "mcp",
                    "kesin",
                    "final",
                    "unfinal",
                    "interim",
                    "provis",
                    "markettrade",
                ]
            )
        ]
        rows.append(
            {
                "file": str(path.relative_to(PROJECT_ROOT)),
                "columns": cols,
                "ptf_related_columns": hits,
                "has_unfinalized_column": any(
                    k in " ".join(lower_cols)
                    for k in ["interim", "unfinal", "kesinlesmemis", "k_ptf", "markettrade"]
                ),
            }
        )
    return rows


def audit_ptf_dataset() -> dict:
    if not PTF_CSV.exists():
        return {"exists": False}
    df = pd.read_csv(PTF_CSV)
    return {
        "exists": True,
        "path": str(PTF_CSV.relative_to(PROJECT_ROOT)),
        "columns": list(df.columns),
        "row_count": int(len(df)),
        "date_min": str(df["date"].min()) if "date" in df.columns else None,
        "date_max": str(df["date"].max()) if "date" in df.columns else None,
        "price_columns_only": list(df.columns) == ["date", "hour", "price", "priceUsd", "priceEur"],
        "unfinalized_or_final_flag_column": False,
        "notes": (
            "Only DAM MCP listing fields (price/priceUsd/priceEur). "
            "No interim/K.PTF/marketTradePrice column."
        ),
    }


def audit_update_dataset_source() -> dict:
    text = UPDATE_SCRIPT.read_text(encoding="utf-8") if UPDATE_SCRIPT.exists() else ""
    return {
        "script": str(UPDATE_SCRIPT.relative_to(PROJECT_ROOT)),
        "endpoint_url": EPIAS_MCP_URL,
        "method": "POST",
        "auth": "CAS TGT (giris.epias.com.tr)",
        "csv_output": "data/ptf_dataset.csv",
        "rolling_refresh_hours": 48,
        "forward_look_days": 7,
        "inferred_price_type": "PTF Listeleme (MCP) — not interim-mcp",
        "code_excerpt_endpoint": "ptf_url = .../markets/dam/data/mcp",
    }


def epias_endpoint_catalog() -> dict:
    return {
        "electricity_service_v1": {
            "unfinalized_k_ptf_list": {
                "name": "Kesinleşmemiş Piyasa Takas Fiyatı (K.PTF) Listeleme",
                "method": "POST",
                "path": "/v1/markets/dam/data/interim-mcp",
                "full_url": EPIAS_INTERIM_MCP_URL,
                "response_price_field": "marketTradePrice",
                "unit": "TL/MWh",
                "description": (
                    "DAM match price before objection period completes (itiraz süreci tamamlanmamış)."
                ),
            },
            "unfinalized_publish_status": {
                "name": "K.PTF yayınlanma durumu",
                "method": "GET",
                "path": "/v1/markets/dam/data/interim-mcp-published-status",
                "full_url": EPIAS_INTERIM_STATUS_URL,
            },
            "ptf_mcp_list_current_ingestion": {
                "name": "Piyasa Takas Fiyatı (PTF) Listeleme",
                "method": "POST",
                "path": "/v1/markets/dam/data/mcp",
                "full_url": EPIAS_MCP_URL,
                "response_price_fields": ["price", "priceUsd", "priceEur"],
                "description": (
                    "Standard hourly DAM MCP/PTF; EPİAŞ documents as post-match clearing price. "
                    "Typically treated as published PTF series (updates after objections finalize)."
                ),
            },
            "export_variants": {
                "interim_export": "POST /v1/markets/dam/export/interim-mcp",
                "mcp_export": "POST /v1/markets/dam/export/mcp",
            },
        },
        "legacy_transparency_api": {
            "interim_mcp": "GET /market/day-ahead-interim-mcp",
            "day_ahead_mcp": "GET /market/day-ahead-mcp",
            "mcp_smp_with_state": {
                "path": "GET /market/mcp-smp",
                "mcpState_enum": ["INTERIM", "FINAL"],
                "note": "Can expose whether MCP row is interim vs finalized on combined feed.",
            },
        },
        "documentation_sources": [
            "https://seffaflik.epias.com.tr/electricity-service/technical/tr/index.html",
            "https://seffaflik.epias.com.tr/transparency/technical/tr/",
        ],
    }


def d_plus_one_alignment_proposal() -> dict:
    return {
        "goal": "Baseline = unfinalized PTF known on day D; target = day D+1 hourly PTF (prefer finalized MCP).",
        "recommended_anchor": "End of day D (e.g. ts_hour = D 23:00 Europe/Istanbul) or last hour when K.PTF for day D is published.",
        "baseline_column_candidate": "interim_mcp.marketTradePrice (K.PTF) for delivery hour on D+1",
        "baseline_rule_option_a": (
            "For target D+1 hour h: baseline = K.PTF(D+1, hour=h) if published on anchor evening; "
            "else K.PTF(D, hour=h) as same-clock proxy."
        ),
        "baseline_rule_option_b": (
            "Persistence-style: baseline(D+1,h) = K.PTF(D, hour=h) using kesinleşmemiş day-D prices only."
        ),
        "target_column": "MCP price (finalized PTF) for (D+1, hour=h) from /mcp endpoint",
        "residual_target": "target_ptf(D+1,h) - baseline(D+1,h)",
        "leakage_checks": [
            "Do not use finalized MCP(D+1) in features available at anchor on D.",
            "Use interim-mcp-published-status to gate hours not yet published.",
            "Separate train features: lag finalized PTF for history vs interim for baseline only.",
        ],
        "current_pipeline_gap": (
            "master.ptf_price and persistence shift(24) use finalized MCP from ptf_dataset.csv; "
            "no interim series joined."
        ),
    }


def conclusions() -> dict:
    return {
        "has_unfinalized_ptf_in_repo": False,
        "current_ptf_column_is": "Finalized/published MCP (price) from POST /markets/dam/data/mcp",
        "required_new_ingestion": {
            "endpoint": EPIAS_INTERIM_MCP_URL,
            "suggested_file": "data/ptf_interim_dataset.csv",
            "suggested_master_columns": [
                "ptf_interim_price",
                "ptf_interim_published",
            ],
            "price_field": "marketTradePrice",
        },
        "keep_existing_mcp_for": "Final targets, backtest realized PTF, and post-finalization evaluation",
        "answer_summary_tr": (
            "Elimizde kesinleşmemiş PTF yok; yalnızca kesinleşmiş/yayınlanmış PTF (mcp→price) var. "
            "Kesinleşmemiş için interim-mcp endpoint ve marketTradePrice kolonu çekilmeli."
        ),
    }


def build_payload() -> dict:
    csv_scan = scan_data_csv_files()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "audit_type": "data_discovery_only",
        "no_training": True,
        "no_feature_changes": True,
        "keywords_searched": KEYWORDS_SCANNED,
        "data_csv_scan": csv_scan,
        "ptf_dataset": audit_ptf_dataset(),
        "update_dataset": audit_update_dataset_source(),
        "master_mapping": {
            "clean_spine": str(CLEAN_PTF.relative_to(PROJECT_ROOT)) if CLEAN_PTF.exists() else None,
            "master_parquet": str(MASTER_PARQUET.relative_to(PROJECT_ROOT))
            if MASTER_PARQUET.exists()
            else None,
            "ptf_price_source_chain": [
                "update_dataset.py → data/ptf_dataset.csv (mcp)",
                "cleaning → data/clean/ptf_hourly.parquet",
                "master spine → ptf_price, ptf_priceUsd, ptf_priceEur",
            ],
        },
        "epias_endpoints": epias_endpoint_catalog(),
        "d_plus_one_alignment": d_plus_one_alignment_proposal(),
        "conclusions": conclusions(),
    }


def write_md(payload: dict) -> None:
    c = payload["conclusions"]
    u = payload["update_dataset"]
    p = payload["ptf_dataset"]
    lines = [
        "# Kesinleşmemiş PTF veri keşfi (audit)",
        "",
        f"Oluşturulma: {payload['generated_at']}",
        "",
        "## Özet cevaplar",
        "",
        f"1. **Elimizde kesinleşmemiş PTF var mı?** → **Hayır** (ayrı kolon/dosya yok).",
        f"2. **Mevcut `price` kolonu ne?** → EPİAŞ **PTF Listeleme** (`POST /v1/markets/dam/data/mcp`), alanlar: `price`, `priceUsd`, `priceEur`.",
        f"3. **Kesinleşmemiş için hangi endpoint?** → `POST /v1/markets/dam/data/interim-mcp`, fiyat alanı: **`marketTradePrice`** (K.PTF).",
        f"4. **Durum servisi (opsiyonel):** `GET /v1/markets/dam/data/interim-mcp-published-status`.",
        "",
        c["answer_summary_tr"],
        "",
        "## 1) `data/*.csv` taraması",
        "",
        "| Dosya | PTF ile ilgili kolonlar | Kesinleşmemiş kolonu? |",
        "|-------|-------------------------|----------------------|",
    ]
    for row in payload["data_csv_scan"]:
        flag = "evet" if row["has_unfinalized_column"] else "hayır"
        cols = ", ".join(row["ptf_related_columns"]) if row["ptf_related_columns"] else "—"
        lines.append(f"| `{row['file']}` | {cols} | {flag} |")
    lines.extend(
        [
            "",
            "## 2) `ptf_dataset.csv`",
            "",
            f"- Kolonlar: `{p.get('columns', [])}`",
            f"- Satır: {p.get('row_count', 'N/A')}, tarih: {p.get('date_min')} → {p.get('date_max')}",
            f"- Kesinleşmemiş/final ayrımı: **yok**",
            "",
            "## 3) `update_dataset.py`",
            "",
            f"- Endpoint: `{u['endpoint_url']}`",
            f"- Yöntem: {u['method']}, çıktı: `{u['csv_output']}`",
            f"- Rolling refresh: {u['rolling_refresh_hours']} saat, forward look: {u['forward_look_days']} gün",
            f"- **Yorum:** Bu servis **MCP/PTF listeleme** (kesinleşmiş/yayınlanmış seri); **interim-mcp değil**.",
            "",
            "## 4) EPİAŞ servis karşılaştırması",
            "",
            "| Servis | Path | Fiyat alanı | Tip |",
            "|--------|------|-------------|-----|",
            "| K.PTF (kesinleşmemiş) | `/v1/markets/dam/data/interim-mcp` | `marketTradePrice` | İtiraz öncesi |",
            "| PTF (mevcut ingestion) | `/v1/markets/dam/data/mcp` | `price` | Yayınlanmış MCP/PTF |",
            "| K.PTF yayın durumu | `/v1/markets/dam/data/interim-mcp-published-status` | — | Metadata |",
            "",
            "Legacy API (referans): `GET /market/day-ahead-interim-mcp`, `GET /market/day-ahead-mcp`, "
            "`GET /market/mcp-smp` (`mcpState`: INTERIM | FINAL).",
            "",
            "## 5) D günü K.PTF → D+1 hedef hizalama (öneri, uygulanmadı)",
            "",
        ]
    )
    align = payload["d_plus_one_alignment"]
    for k, v in align.items():
        if isinstance(v, list):
            lines.append(f"**{k}:**")
            for item in v:
                lines.append(f"- {item}")
        else:
            lines.append(f"- **{k}:** {v}")
    lines.extend(
        [
            "",
            "## 6) Sonraki adım (kod değişikliği bu audit kapsamında yapılmadı)",
            "",
            f"- Yeni CSV/parquet: `{c['required_new_ingestion']['suggested_file']}`",
            f"- Endpoint: `{c['required_new_ingestion']['endpoint']}`",
            "- Master’da `ptf_interim_price` + finalized `ptf_price` ayrı tutulmalı",
            "- Baseline: K.PTF; hedef: D+1 finalized MCP; model residual sapmayı tahmin eder",
            "",
        ]
    )
    METRICS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    payload = build_payload()
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(payload)
    print(f"Wrote {METRICS_JSON}")
    print(f"Wrote {METRICS_MD}")
    print(payload["conclusions"]["answer_summary_tr"])


if __name__ == "__main__":
    main()

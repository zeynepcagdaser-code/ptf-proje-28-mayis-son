# Plant-Level KGUP Audit

Generated: `2026-05-31T22:53:39.081479+00:00`

## Source Status

- Raw directory: `data/plant_level_kgup/raw`
- Raw normalized rows: `48192`
- Leakage-safe latest rows: `45192`
- Feature rows: `24`

## API Semantics

- EPİAŞ documentation lists `POST /v1/generation/data/dpp-bulk` for bulk UEVÇB KGÜP by day.
- Plant/UEVÇB ID discovery should use `powerplant-list`, `uevcb-list`, or `uevcb-list-bulk` before historical fetching.
- This run did not start broad API fetching; it ingested local raw files only.

## YEKDEM Matching

- Matched plants: `648`
- Unmatched plants: `1235`
- Duplicate mappings: `0`
- Low-confidence matches: `0`
- YEKDEM share of leakage-safe plant KGUP: `0.13841226789616143`

## Leakage Audit

- Eligible as-of rows: `45192`
- Rows failing publication <= forecast rule or missing timestamps: `0`
- Missing publication timestamp: `48192`
- Missing forecast timestamp: `0`
- Missing archive snapshot timestamp: `0`

## Coverage By Registry Year

| registry_year | plant_count | source_files |
| --- | --- | --- |
| 2021 | 894 | _PortalAdmin_Uploads_Content_News_49ffec2852371 (1).xls |
| 2022 | 1022 | _PortalAdmin_Uploads_Content_News_1bbe9be893217.xls |
| 2023 | 859 | _PortalAdmin_Uploads_Content_News_b2e2dbd832616.xlsx |
| 2024 | 753 | _PortalAdmin_Uploads_Content_News_47ef17c255990.xlsx |
| 2025 | 725 | 2025-Nihai-YEK-Listesi (1).xls |
| 2026 | 622 | _PortalAdmin_Uploads_Content_News_8024708e54184.xlsx |


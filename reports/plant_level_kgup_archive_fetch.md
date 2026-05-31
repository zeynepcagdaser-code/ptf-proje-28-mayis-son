# Plant-Level KGUP Archive Fetch

- Period: `2026-05-31` → `2026-05-31`
- Power plants discovered: `716`
- UEVÇB ids discovered: `2115`
- Requested chunks: `85`
- Successful chunks: `73`
- Failed chunks: `12`
- Combined archive rows: `48192`

## Notes

- EPİAŞ dpp-bulk gives hourly plant-level KGÜP by UEVÇB.
- Power plant discovery uses `powerplant-list` and then `uevcb-list-by-power-plant-id`.
- If the source does not expose a publication timestamp, the field is preserved as null and flagged in the audit.

## Paths

- Raw archive root: `data/plant_level_kgup/raw_archive`
- Combined parquet: `data/plant_level_kgup/plant_level_kgup_archive.parquet`
- State file: `data/plant_level_kgup/raw_archive/2026-05-31_2026-05-31/plant_level_kgup_archive_state_2026-05-31_2026-05-31.json`

## Next Step

Run `python3 build_plant_level_kgup_pipeline.py` to convert the archive into leakage-audited must-run features.


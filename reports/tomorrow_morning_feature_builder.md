# Tomorrow Morning Feature Builder

- Requested rows: `13`
- Produced rows: `0`
- Feature store max ts: `2026-05-31 23:00:00`
- Load forecast max ts: `2026-05-31 23:00:00`
- Regime labels max ts: `2026-05-31 23:00:00`
- Must-run rows: `0`

## Why 0 rows happened before

The earlier builder only looked for an exact tomorrow-date match inside historical tables. Those tables stop at the latest observed day, so tomorrow had no direct rows to copy.

## Fallbacks

- None

## Notes

The table is leakage-safe because it only uses lag-24 labels and available forecast/history rows.

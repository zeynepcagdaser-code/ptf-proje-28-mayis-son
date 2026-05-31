# Tomorrow Must-Run Enrichment Audit

Generated: `2026-05-31T21:52:49.925516+00:00`

- Tomorrow rows: `13`
- Proxy rows available: `24`
- Proxy rows matched: `13`
- Fallback rows: `0`
- strict_point_in_time_safe rows: `0`
- structural_market_proxy rows: `13`
- Leakage risk: `medium`

## Missing Features

- None

## Can this be used for tomorrow?

Yes, for analysis and directional feature enrichment. The merge is usable for tomorrow morning inference, but not strict point-in-time safe because the proxy source is structural and comes from smoke data without publication timestamps.

## Leakage Notes

The enrichment preserves `strict_point_in_time_safe` and `structural_market_proxy`. The risk is structural rather than accidental label leakage: the proxy comes from smoke raw generation with no publication timestamp, so it should not be treated as a historical live-as-of feature.

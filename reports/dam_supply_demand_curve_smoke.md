# DAM Supply-Demand Curve Smoke Test

Generated: `2026-05-31T19:08:40.974157+00:00`

## Endpoint Status

- Supply-demand endpoint status: `200`
- PTF endpoint status: `200`
- Supply-demand rows/items: `718`
- PTF rows/items: `1`
- Normalized rows: `719`

## Coverage

- Smoke date: `2026-06-01`
- Smoke hour: `00:00`
- Supply-demand columns: `['amount', 'supplyPrice', 'demandPrice']`
- PTF columns: `['date', 'mcpPrice', 'matchingQuantity']`

## Normalized Columns

- `delivery_hour, price, supply_mwh, demand_mwh, source_endpoint`

## Findings

- Clearing price column found: `True`
- Supply column present: `True`
- Demand column present: `True`
- Curve extraction usable: `True`

## Raw Artifacts

- Directory: `data/raw/dam_supply_demand_curve_smoke`
- PTF raw body: `data/raw/dam_supply_demand_curve_smoke/ptf_attempt_01.body.txt`
- Supply-demand raw body: `data/raw/dam_supply_demand_curve_smoke/supply_demand_attempt_01.body.txt`

## Notes

The smoke test only hits one date/hour. Raw response bodies and headers are preserved so the schema can be inspected before building any full historical pipeline.

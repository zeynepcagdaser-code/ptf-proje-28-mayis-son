# Kesinleşmemiş PTF veri keşfi (audit)

Oluşturulma: 2026-05-30T12:18:40.067932+00:00

## Özet cevaplar

1. **Elimizde kesinleşmemiş PTF var mı?** → **Hayır** (ayrı kolon/dosya yok).
2. **Mevcut `price` kolonu ne?** → EPİAŞ **PTF Listeleme** (`POST /v1/markets/dam/data/mcp`), alanlar: `price`, `priceUsd`, `priceEur`.
3. **Kesinleşmemiş için hangi endpoint?** → `POST /v1/markets/dam/data/interim-mcp`, fiyat alanı: **`marketTradePrice`** (K.PTF).
4. **Durum servisi (opsiyonel):** `GET /v1/markets/dam/data/interim-mcp-published-status`.

Elimizde kesinleşmemiş PTF yok; yalnızca kesinleşmiş/yayınlanmış PTF (mcp→price) var. Kesinleşmemiş için interim-mcp endpoint ve marketTradePrice kolonu çekilmeli.

## 1) `data/*.csv` taraması

| Dosya | PTF ile ilgili kolonlar | Kesinleşmemiş kolonu? |
|-------|-------------------------|----------------------|
| `data/kgup_combined.csv` | — | hayır |
| `data/load_forecast.csv` | — | hayır |
| `data/outages.csv` | — | hayır |
| `data/ptf_dataset.csv` | — | hayır |
| `data/real_consumption.csv` | — | hayır |
| `data/realtime_generation.csv` | — | hayır |
| `data/smf.csv` | — | hayır |
| `data/wind_forecast.csv` | — | hayır |
| `data/yal_yat.csv` | — | hayır |

## 2) `ptf_dataset.csv`

- Kolonlar: `['date', 'hour', 'price', 'priceUsd', 'priceEur']`
- Satır: 56208, tarih: 2020-01-01T00:00:00+03:00 → 2026-05-30T23:00:00+03:00
- Kesinleşmemiş/final ayrımı: **yok**

## 3) `update_dataset.py`

- Endpoint: `https://seffaflik.epias.com.tr/electricity-service/v1/markets/dam/data/mcp`
- Yöntem: POST, çıktı: `data/ptf_dataset.csv`
- Rolling refresh: 48 saat, forward look: 7 gün
- **Yorum:** Bu servis **MCP/PTF listeleme** (kesinleşmiş/yayınlanmış seri); **interim-mcp değil**.

## 4) EPİAŞ servis karşılaştırması

| Servis | Path | Fiyat alanı | Tip |
|--------|------|-------------|-----|
| K.PTF (kesinleşmemiş) | `/v1/markets/dam/data/interim-mcp` | `marketTradePrice` | İtiraz öncesi |
| PTF (mevcut ingestion) | `/v1/markets/dam/data/mcp` | `price` | Yayınlanmış MCP/PTF |
| K.PTF yayın durumu | `/v1/markets/dam/data/interim-mcp-published-status` | — | Metadata |

Legacy API (referans): `GET /market/day-ahead-interim-mcp`, `GET /market/day-ahead-mcp`, `GET /market/mcp-smp` (`mcpState`: INTERIM | FINAL).

## 5) D günü K.PTF → D+1 hedef hizalama (öneri, uygulanmadı)

- **goal:** Baseline = unfinalized PTF known on day D; target = day D+1 hourly PTF (prefer finalized MCP).
- **recommended_anchor:** End of day D (e.g. ts_hour = D 23:00 Europe/Istanbul) or last hour when K.PTF for day D is published.
- **baseline_column_candidate:** interim_mcp.marketTradePrice (K.PTF) for delivery hour on D+1
- **baseline_rule_option_a:** For target D+1 hour h: baseline = K.PTF(D+1, hour=h) if published on anchor evening; else K.PTF(D, hour=h) as same-clock proxy.
- **baseline_rule_option_b:** Persistence-style: baseline(D+1,h) = K.PTF(D, hour=h) using kesinleşmemiş day-D prices only.
- **target_column:** MCP price (finalized PTF) for (D+1, hour=h) from /mcp endpoint
- **residual_target:** target_ptf(D+1,h) - baseline(D+1,h)
**leakage_checks:**
- Do not use finalized MCP(D+1) in features available at anchor on D.
- Use interim-mcp-published-status to gate hours not yet published.
- Separate train features: lag finalized PTF for history vs interim for baseline only.
- **current_pipeline_gap:** master.ptf_price and persistence shift(24) use finalized MCP from ptf_dataset.csv; no interim series joined.

## 6) Sonraki adım (kod değişikliği bu audit kapsamında yapılmadı)

- Yeni CSV/parquet: `data/ptf_interim_dataset.csv`
- Endpoint: `https://seffaflik.epias.com.tr/electricity-service/v1/markets/dam/data/interim-mcp`
- Master’da `ptf_interim_price` + finalized `ptf_price` ayrı tutulmalı
- Baseline: K.PTF; hedef: D+1 finalized MCP; model residual sapmayı tahmin eder


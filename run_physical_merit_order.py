#!/usr/bin/env python3
"""
Yaklaşım A: Fiziksel Merit Order PTF Tahmini

Yöntem:
  1. KGUP yakıt tipine göre → maliyet sıralı arz adımları
  2. LEP yük tahmini → price-inelastic talep
  3. Arz eğrisi × talep kesişimi → takas fiyatı (LP değil, doğrudan merit order)

Sızdırmazlık:
  - KGUP: D-1 öğleden önce yayımlanan D+1 planı → güvenli
  - Load forecast (LEP): D-1 yayımlanan → güvenli
  - GRF: grf_tl_lag_24 (bugünün GRF'i değil, dün yayımlananı) → güvenli
  - TTF/API2/Brent: _lag_7d versiyonları → güvenli

Kullanım:
    python run_physical_merit_order.py               # 2020–2026 backtest
    python run_physical_merit_order.py --year 2026   # sadece 2026
    python run_physical_merit_order.py --plot        # grafik göster
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
MASTER_PATH  = PROJECT_ROOT / "data" / "master" / "master_hourly_v1.parquet"
REPORTS_DIR  = PROJECT_ROOT / "reports"

# ─── Fiziksel sabitler ────────────────────────────────────────────────────────

# Isıl verimlilik
_HR_CCGT        = 7.5    # mmbtu/MWh
_HR_SIMPLE      = 11.0   # mmbtu/MWh
_HR_COAL_T      = 0.385  # tonne/MWh  (26 GJ/t)
_HR_LIGNITE_T   = 2.17   # tonne/MWh  (6 GJ/t)
_HR_FUELOIL_T   = 0.22   # tonne/MWh

# Değişken O&M (TL/MWh)
_VOM_GAS     = 8.0
_VOM_COAL    = 5.0
_VOM_LIGNITE = 4.0
_VOM_FUELOIL = 12.0

# Yerli linyit sabit maliyet (USD/tonne) — piyasa verisi yok
_LIGNITE_FIXED_USD_T = 22.0

# KGUP'ta gaz santrallerinin CCGT payı (yaklaşık)
_CCGT_SHARE = 0.65   # kalan 0.35 basit çevrim

# Fiyat tavanı değişim tarihi
_CEILING_CHANGE = pd.Timestamp("2026-04-01")
_CEILING_OLD    = 3_400.0
_CEILING_NEW    = 4_500.0


# ─── Marginal maliyet hesabı ──────────────────────────────────────────────────

def compute_mc(row: pd.Series) -> dict[str, float]:
    """
    Tek saat için yakıt tipine göre marginal maliyet (TL/MWh).
    Look-ahead güvenli versiyon:
      - Gaz: grf_tl_lag_24 (önceki günün GRF'i)
      - Kömür/Fuel-oil: _lag_7d emtia fiyatları
    """
    usd = max(row.get("usd_try_buy_lag_1d", row.get("usd_try_buy", 30.0)), 1.0)

    # Doğalgaz (GRF TL/1000sm3 → TL/MWh)
    grf_tl_1000sm3 = row.get("grf_tl_lag_24", row.get("grf_tl_1000sm3", 5000.0))
    grf_tl_mmbtu   = grf_tl_1000sm3 / 36.6   # 1 mmbtu ≈ 0.0366 ksm3
    mc_gas_ccgt    = grf_tl_mmbtu * _HR_CCGT   + _VOM_GAS
    mc_gas_simple  = grf_tl_mmbtu * _HR_SIMPLE + _VOM_GAS

    # İthal kömür (USD/tonne → TL/MWh)
    coal_usd_t = row.get("coal_api2_usd_lag_7d", row.get("coal_api2_usd", 80.0))
    mc_coal    = coal_usd_t * usd * _HR_COAL_T + _VOM_COAL

    # Yerli linyit
    mc_lignite = _LIGNITE_FIXED_USD_T * usd * _HR_LIGNITE_T + _VOM_LIGNITE

    # Fuel-oil (Brent proxy, USD/bbl → USD/tonne ≈ /7.3)
    brent_usd  = row.get("brent_usd_lag_7d", row.get("brent_usd", 70.0))
    fo_usd_t   = brent_usd / 7.3 * 1000.0
    mc_fueloil = fo_usd_t * usd * _HR_FUELOIL_T + _VOM_FUELOIL

    return {
        "mc_ren":      0.0,          # yenilenebilir + hidro
        "mc_lignite":  mc_lignite,
        "mc_coal":     mc_coal,
        "mc_gas_ccgt": mc_gas_ccgt,
        "mc_gas_sim":  mc_gas_simple,
        "mc_fueloil":  mc_fueloil,
    }


# ─── Merit order takas ────────────────────────────────────────────────────────

def merit_order_price(row: pd.Series, mc: dict[str, float]) -> tuple[float, str]:
    """
    Rejim tabanlı marjinal yakıt tahmini.

    KGUP toplam ile yük arasında yapısal bir boşluk var (ithalat + ölçüm farkı).
    Bu nedenle arz-talep balance kurmak yerine hangi yakıt marjinal sorusunu soruyoruz:

      Adım 1 → Yenilenebilir baskısı var mı?  (ren_share > eşik → PTF ≈ 0)
      Adım 2 → Kömür mü, gaz mı marjinal?
                gas_share yüksek  → gas_ccgt marjinal
                coal_share yüksek → coal marjinal
                gas düşük + hydro yüksek → lignite/coal marjinal

    Fiyat tahmininde tek düzelme: gaz marjinal saatlerde kGÜP gaz yükü
    ile GRF arasındaki etkileşimi de hesaba kat.
    """
    ceiling = _CEILING_NEW if pd.Timestamp(row.get("ts_hour", pd.NaT)) >= _CEILING_CHANGE else _CEILING_OLD

    load = max(row.get("load_forecast", 1.0), 1.0)

    # Yakıt payları (yük bazında)
    ren_gen  = (row.get("kgup_wind", 0) + row.get("kgup_solar", 0) +
                row.get("kgup_river", 0) + row.get("kgup_geothermal", 0) +
                row.get("kgup_biomass", 0))
    hydro    = row.get("kgup_dammed_hydro", 0)
    gas      = row.get("kgup_gas", 0)
    coal     = row.get("kgup_import_coal", 0) + row.get("kgup_black_coal", 0)
    lignite  = row.get("kgup_lignite", 0)

    total_zero   = ren_gen + hydro   # sıfır+düşük maliyetli üretim
    total_thermal = gas + coal + lignite

    ren_share   = ren_gen / load        # sadece RES (hidro hariç)
    zero_share  = total_zero / load     # RES + hidro
    gas_share   = gas / max(total_thermal, 1)
    coal_share  = coal / max(total_thermal, 1)

    # ── Rejim 1: Yenilenebilir fazlası ────────────────────────────────────────
    # Gündüz solar surge veya yüksek rüzgar → sıfır fiyat
    if ren_share > 0.85 or zero_share > 1.05:
        return max(0.0, mc["mc_lignite"] * 0.1), "ren_surplus"

    if ren_share > 0.60:
        # RES baskılı ama marjinal saat: düşük fiyat bölgesi
        blend = (ren_share - 0.60) / 0.25   # 0→1 arasında
        price = mc["mc_lignite"] * (1 - blend) + 0.0 * blend
        return max(0.0, min(price, ceiling)), "ren_transition"

    # ── Rejim 2: Kömür marjinal ────────────────────────────────────────────────
    # Düşük gaz payı + yeterli kömür kapasitesi
    if gas_share < 0.30 and coal_share > 0.35:
        return min(mc["mc_coal"], ceiling), "coal"

    if gas_share < 0.15:
        return min(mc["mc_lignite"], ceiling), "lignite"

    # ── Rejim 3: Gaz marjinal (varsayılan) ────────────────────────────────────
    # Gaz payına göre CCGT ve basit çevrim arasında ağırlıklı ortalama
    # Yüksek gaz payı → CCGT ağırlıklı (verimli santral) → daha düşük mc
    ccgt_w = min(gas_share / 0.50, 1.0)
    gas_mc = mc["mc_gas_ccgt"] * ccgt_w + mc["mc_gas_sim"] * (1 - ccgt_w)

    # Yüksek yük → yüksek gaz talebi → fiyat baskısı (talep elastikiyeti proxy)
    load_pressure = max((load / 50_000) - 0.6, 0.0) * 200.0   # 50GWh üstü her %10 için +20 TL

    price = gas_mc + load_pressure
    return min(max(price, 0.0), ceiling), "gas"


# ─── Ana backtest ─────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tüm satırlar için merit order tahminini hesapla.
    """
    results = []
    for _, row in df.iterrows():
        mc = compute_mc(row)
        pred, fuel = merit_order_price(row, mc)
        results.append({
            "ts_hour":        row["ts_hour"],
            "actual_ptf":     row.get("ptf", np.nan),
            "predicted_ptf":  pred,
            "marginal_fuel":  fuel,
            "load_forecast":  row.get("load_forecast", np.nan),
        })
    return pd.DataFrame(results)


def print_report(res: pd.DataFrame, label: str = "Tüm dönem") -> None:
    valid = res.dropna(subset=["actual_ptf", "predicted_ptf"])
    err   = valid["predicted_ptf"] - valid["actual_ptf"]
    mae   = err.abs().mean()
    bias  = err.mean()
    wape  = (err.abs().sum() / valid["actual_ptf"].clip(lower=1).sum()) * 100
    pers  = (valid["actual_ptf"].diff().abs().mean())   # basit persistence proxy

    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    print(f"  Saat sayısı : {len(valid):,}")
    print(f"  MAE         : {mae:,.1f} TL/MWh")
    print(f"  Bias        : {bias:+,.1f} TL/MWh")
    print(f"  WAPE        : {wape:.1f}%")

    if "marginal_fuel" in valid.columns:
        fuel_dist = valid["marginal_fuel"].value_counts(normalize=True) * 100
        print(f"  Marjinal yakıt dağılımı:")
        for fuel, pct in fuel_dist.items():
            print(f"    {fuel:<15}: {pct:.1f}%")


# ─── Entrypoint ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",  type=int, default=None, help="Sadece bu yılı çalıştır")
    parser.add_argument("--split", default="test",
                        choices=["train","val","test","all"],
                        help="Hangi split")
    parser.add_argument("--plot",  action="store_true")
    args = parser.parse_args()

    print(f"Master dataset yükleniyor: {MASTER_PATH}")
    df = pd.read_parquet(MASTER_PATH)
    df["ts_hour"] = pd.to_datetime(df["ts_hour"])

    # Yıl filtresi
    if args.year:
        df = df[df["ts_hour"].dt.year == args.year]
        label = f"{args.year} backtest"
    elif args.split != "all":
        df = df[df["split"] == args.split]
        label = f"{args.split} split"
    else:
        label = "Tüm dönem (2020–2026)"

    print(f"  {len(df):,} saat işleniyor...")

    res = run_backtest(df)

    # Genel rapor
    print_report(res, label)

    # Yıllık kırılım
    res["year"] = res["ts_hour"].dt.year
    print(f"\n{'─'*55}")
    print("  Yıllık MAE kırılımı")
    print(f"{'─'*55}")
    for yr, grp in res.dropna(subset=["actual_ptf"]).groupby("year"):
        e   = (grp["predicted_ptf"] - grp["actual_ptf"]).abs()
        bias = (grp["predicted_ptf"] - grp["actual_ptf"]).mean()
        print(f"  {yr}  MAE={e.mean():>7.1f}  Bias={bias:>+7.1f}  n={len(grp):,}")

    # Kaydet
    out_csv  = REPORTS_DIR / "physical_merit_order_backtest.csv"
    out_json = REPORTS_DIR / "physical_merit_order_backtest.json"
    res.to_csv(out_csv, index=False)

    import json
    summary = {
        "label":       label,
        "n_hours":     len(res.dropna(subset=["actual_ptf"])),
        "mae":         round(float((res["predicted_ptf"] - res["actual_ptf"]).abs().mean()), 2),
        "bias":        round(float((res["predicted_ptf"] - res["actual_ptf"]).mean()), 2),
        "wape":        round(float(
            (res["predicted_ptf"] - res["actual_ptf"]).abs().sum() /
            res["actual_ptf"].clip(lower=1).sum() * 100
        ), 2),
        "yearly": {
            str(yr): round(float((g["predicted_ptf"] - g["actual_ptf"]).abs().mean()), 1)
            for yr, g in res.dropna(subset=["actual_ptf"]).groupby(res["ts_hour"].dt.year)
        },
    }
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n  Rapor kaydedildi:")
    print(f"    {out_csv}")
    print(f"    {out_json}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sample = res.dropna(subset=["actual_ptf"]).tail(720)  # son 30 gün
        plt.figure(figsize=(14, 4))
        plt.plot(sample["ts_hour"], sample["actual_ptf"],    label="Gerçek", lw=1, alpha=0.8)
        plt.plot(sample["ts_hour"], sample["predicted_ptf"], label="Merit Order", lw=1, alpha=0.8, ls="--")
        plt.title("Fiziksel Merit Order — Son 30 Gün")
        plt.ylabel("TL/MWh")
        plt.legend()
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "physical_merit_order_last30d.png", dpi=120)
        print(f"    {REPORTS_DIR / 'physical_merit_order_last30d.png'}")


if __name__ == "__main__":
    main()

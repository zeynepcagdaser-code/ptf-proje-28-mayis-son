#!/usr/bin/env python3
"""
Yaklaşım B: Fiziksel + ML Hibrit PTF Tahmini

Fiziksel merit order tahmini + marjinal maliyet sinyalleri → LightGBM feature olarak ekle.
Model artık hem PTF geçmişini hem de piyasanın fiziksel durumunu anlıyor.

Akış:
  1. Master dataset'e fiziksel feature'lar ekle (vektörize, hızlı)
  2. LightGBM'i bu ek feature'larla eğit (aynı split: 2020-2025 Q1/Q3/Q4 train, 2025 Q2 val, 2026 test)
  3. MAE'yi Yaklaşım A (465 TL) ve eski LightGBM (359 TL) ile karşılaştır
"""

from __future__ import annotations

import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
MASTER_PATH  = PROJECT_ROOT / "data" / "master" / "master_hourly_v1.parquet"
REPORTS_DIR  = PROJECT_ROOT / "reports"

# ─── Fiziksel sabitler ────────────────────────────────────────────────────────
_HR_CCGT       = 7.5
_HR_SIMPLE     = 11.0
_HR_COAL_T     = 0.385
_HR_LIGNITE_T  = 2.17
_VOM_GAS       = 8.0
_VOM_COAL      = 5.0
_VOM_LIGNITE   = 4.0
_LIGNITE_USD_T = 22.0
_CCGT_SHARE    = 0.65

_CEILING_CHANGE = pd.Timestamp("2026-04-01")
_CEILING_OLD    = 3_400.0
_CEILING_NEW    = 4_500.0


# ─── Vektörize fiziksel feature hesabı ───────────────────────────────────────

def add_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master DataFrame'e fiziksel piyasa feature'larını ekle.
    Tüm hesaplamalar vektörize — row-by-row döngüsü yok.
    """
    d = df.copy()
    ts = pd.to_datetime(d["ts_hour"])

    usd = d.get("usd_try_buy_lag_1d", d.get("usd_try_buy", 30.0)).clip(lower=1.0)

    # ── Marjinal maliyetler (TL/MWh) ──────────────────────────────────────────
    grf_lag  = d.get("grf_tl_lag_24", d.get("grf_tl_1000sm3", 5000.0)).fillna(5000.0)
    grf_mmbtu = grf_lag / 36.6   # TL/mmbtu

    d["phys_mc_gas_ccgt"]   = (grf_mmbtu * _HR_CCGT   + _VOM_GAS).clip(lower=0)
    d["phys_mc_gas_simple"] = (grf_mmbtu * _HR_SIMPLE + _VOM_GAS).clip(lower=0)

    coal_usd = d.get("coal_api2_usd_lag_7d", d.get("coal_api2_usd", 80.0)).fillna(80.0)
    d["phys_mc_coal"] = (coal_usd * usd * _HR_COAL_T + _VOM_COAL).clip(lower=0)

    d["phys_mc_lignite"] = (_LIGNITE_USD_T * usd * _HR_LIGNITE_T + _VOM_LIGNITE).clip(lower=0)

    brent_usd = d.get("brent_usd_lag_7d", d.get("brent_usd", 70.0)).fillna(70.0)
    d["phys_mc_fueloil"]  = (brent_usd / 7.3 * 1000.0 * usd * 0.22 + 12.0).clip(lower=0)

    # ── Yakıt geçiş marjı ─────────────────────────────────────────────────────
    d["phys_fuel_switch_spread"] = d["phys_mc_gas_ccgt"] - d["phys_mc_coal"]
    d["phys_gas_coal_ratio"]     = (d["phys_mc_gas_ccgt"] / d["phys_mc_coal"].clip(lower=1)).clip(0, 10)

    # ── Yük ve yenilenebilir payları ──────────────────────────────────────────
    load = d["load_forecast"].clip(lower=1.0)

    ren_gen = (
        d.get("kgup_wind",       0).fillna(0) +
        d.get("kgup_solar",      0).fillna(0) +
        d.get("kgup_river",      0).fillna(0) +
        d.get("kgup_geothermal", 0).fillna(0) +
        d.get("kgup_biomass",    0).fillna(0)
    )
    hydro    = d.get("kgup_dammed_hydro",  0).fillna(0)
    gas_kgup = d.get("kgup_gas",           0).fillna(0)
    coal_kgup = (d.get("kgup_import_coal", 0).fillna(0) +
                 d.get("kgup_black_coal",  0).fillna(0))
    lignite  = d.get("kgup_lignite",       0).fillna(0)

    thermal  = (gas_kgup + coal_kgup + lignite).clip(lower=1)

    d["phys_ren_share"]   = (ren_gen / load).clip(0, 2)
    d["phys_zero_share"]  = ((ren_gen + hydro) / load).clip(0, 2)
    d["phys_gas_share_t"] = (gas_kgup  / thermal).clip(0, 1)
    d["phys_coal_share_t"]= (coal_kgup / thermal).clip(0, 1)
    d["phys_net_load"]    = (load - ren_gen - hydro).clip(lower=0)
    d["phys_net_load_ratio"] = (d["phys_net_load"] / load).clip(0, 1)

    # ── Rejim tabanlı fiziksel fiyat tahmini ──────────────────────────────────
    ceiling = np.where(ts >= _CEILING_CHANGE, _CEILING_NEW, _CEILING_OLD)

    ren_share   = d["phys_ren_share"].values
    gas_share_t = d["phys_gas_share_t"].values
    coal_share_t= d["phys_coal_share_t"].values
    mc_gas_ccgt = d["phys_mc_gas_ccgt"].values
    mc_gas_sim  = d["phys_mc_gas_simple"].values
    mc_coal     = d["phys_mc_coal"].values
    mc_lig      = d["phys_mc_lignite"].values
    load_v      = load.values
    z_share     = d["phys_zero_share"].values

    ccgt_w  = np.clip(gas_share_t / 0.50, 0, 1)
    gas_mc  = mc_gas_ccgt * ccgt_w + mc_gas_sim * (1 - ccgt_w)
    load_p  = np.maximum((load_v / 50_000) - 0.6, 0) * 200.0

    # Rejim 1: RES fazlası
    cond_ren_surplus    = ren_share > 0.85
    cond_ren_transition = (ren_share > 0.60) & ~cond_ren_surplus
    # Rejim 2: kömür marjinal
    cond_coal = (gas_share_t < 0.30) & (coal_share_t > 0.35) & ~cond_ren_surplus & ~cond_ren_transition
    # Rejim 3: linyit marjinal
    cond_lig  = (gas_share_t < 0.15) & ~cond_ren_surplus & ~cond_ren_transition & ~cond_coal

    phys_ptf = gas_mc + load_p   # varsayılan: gaz marjinal

    blend = np.clip((ren_share - 0.60) / 0.25, 0, 1)
    phys_ptf = np.where(cond_ren_transition, mc_lig * (1 - blend), phys_ptf)
    phys_ptf = np.where(cond_ren_surplus,    mc_lig * 0.1, phys_ptf)
    phys_ptf = np.where(cond_coal, mc_coal, phys_ptf)
    phys_ptf = np.where(cond_lig,  mc_lig,  phys_ptf)
    phys_ptf = np.clip(phys_ptf, 0, ceiling)

    d["phys_ptf"]      = phys_ptf
    d["phys_ptf_lag24"]= d["phys_ptf"].shift(24).bfill()

    # ── Rejim sınıflandırması (integer + one-hot) ─────────────────────────────
    # 0=gaz, 1=kömür, 2=linyit, 3=ren_geçiş, 4=ren_fazlası
    regime = np.zeros(len(d), dtype=np.int8)
    regime[cond_lig]           = 2
    regime[cond_coal]          = 1
    regime[cond_ren_transition] = 3
    regime[cond_ren_surplus]   = 4
    d["phys_regime"] = regime

    # One-hot (LightGBM nominal daha iyi öğrenir)
    for r, name in [(0,"gas"),(1,"coal"),(2,"lignite"),(3,"ren_trans"),(4,"ren_surplus")]:
        d[f"phys_regime_{name}"] = (regime == r).astype(np.int8)

    # ── Rejim × Maliyet etkileşim feature'ları ───────────────────────────────
    # Model rejim içinde fiyatın neden yüksek/düşük olduğunu anlasın
    d["phys_gas_mc_x_regime"]  = d["phys_mc_gas_ccgt"] * d["phys_regime_gas"]
    d["phys_coal_mc_x_regime"] = d["phys_mc_coal"]     * d["phys_regime_coal"]
    d["phys_ren_share_x_ren"]  = d["phys_ren_share"]   * (d["phys_regime_ren_trans"] + d["phys_regime_ren_surplus"])

    # ── Fiziksel modelin PTF geçmişinden residual'ı ───────────────────────────
    if "ptf" in d.columns:
        d["phys_residual"] = d["ptf"] - d["phys_ptf"]

    return d


# ─── Model eğitimi ────────────────────────────────────────────────────────────

PHYSICAL_FEATURES = [
    # Marjinal maliyet sinyalleri
    "phys_mc_gas_ccgt",
    "phys_mc_gas_simple",
    "phys_mc_coal",
    "phys_mc_lignite",
    "phys_fuel_switch_spread",
    "phys_gas_coal_ratio",
    # Yük ve üretim karışımı
    "phys_ren_share",
    "phys_zero_share",
    "phys_gas_share_t",
    "phys_coal_share_t",
    "phys_net_load",
    "phys_net_load_ratio",
    # Rejim (marjinal yakıt tahmini)
    "phys_regime",
    "phys_regime_gas",
    "phys_regime_coal",
    "phys_regime_lignite",
    "phys_regime_ren_trans",
    "phys_regime_ren_surplus",
    # Rejim × maliyet etkileşimleri
    "phys_gas_mc_x_regime",
    "phys_coal_mc_x_regime",
    "phys_ren_share_x_ren",
]


def build_feature_set(df: pd.DataFrame) -> list[str]:
    """Mevcut feature'lar + fiziksel feature'lar (look-ahead riski olanlar hariç)."""
    # Ham gerçekleşen değerler (lag/roll olmadan) → D+1 tahminde bilinmez
    REALISED_PREFIXES = (
        "gen_",          # gerçekleşen üretim
        "real_con",      # gerçekleşen tüketim
        "smf",           # sistem marjinal fiyatı (teslimat sonrası)
        "yal_yat",       # yük alma/atma talimatları
        "upRegulation",  # düzenleme talimatları
        "downRegulation",
        "idm_price",     # GİP fiyatı (teslimat sonrası, lag olmadan)
    )

    def _is_short_lag(col: str) -> bool:
        """D+1 için 24h'ten kısa lag/roll feature'ları leak taşır."""
        import re
        # ptf_lag_N, idm_price_lag_N, smf_lag_N → N < 24 ise unsafe
        m = re.search(r"_lag_(\d+)$", col)
        if m and int(m.group(1)) < 24:
            return True
        # roll_mean_N veya roll_max_N → N <= 24 ise unsafe (anchor 23:00)
        m = re.search(r"_roll_(?:mean|max|min|std)_(\d+)", col)
        if m and int(m.group(1)) <= 24:
            return True
        return False

    exclude = {
        "ts_hour", "ptf", "split",
        # Post-hoc ham değerler
        *[c for c in df.columns
          if any(c.startswith(p) for p in REALISED_PREFIXES)
          and "lag" not in c and "roll" not in c],
        # 24h'ten kısa lag/roll → D+1 anı için bilinmez
        *[c for c in df.columns if _is_short_lag(c)],
        # Look-ahead (teslimat günü GÖP sonuçları)
        "delivery_grf_tl_1000sm3",
        "delivery_dam_bid_volume", "delivery_dam_ask_volume",
        "delivery_dam_block_buy_volume", "delivery_dam_block_sell_volume",
        "delivery_smf",
        # Hedeften türetilmiş
        "ptf_priceUsd", "ptf_priceEur",
        "ptf_is_price_zero", "ptf_is_price_capped",
        "ptf_is_priceUsd_zero", "ptf_is_priceUsd_capped",
        "ptf_is_priceEur_zero", "ptf_is_priceEur_capped",
        "phys_residual",
        "phys_ptf",       # bileşik fiziksel tahmin — test setinde 1161 MAE, noise
        "phys_ptf_lag24",
    }
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in exclude]


def train_hybrid(df: pd.DataFrame) -> dict:
    """Hibrit modeli eğit, test MAE döndür."""
    train = df[df["split"] == "train"].copy()
    val   = df[df["split"].isin(["val", "validation"])].copy()
    test  = df[df["split"] == "test"].copy()

    feature_cols = build_feature_set(df)
    # Sadece mevcut sütunlar
    feature_cols = [c for c in feature_cols if c in df.columns]

    print(f"  Feature sayısı: {len(feature_cols)}")
    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    X_tr = train[feature_cols].fillna(0).values.astype(np.float32)
    y_tr = train["ptf"].values
    X_vl = val[feature_cols].fillna(0).values.astype(np.float32)
    y_vl = val["ptf"].values
    X_te = test[feature_cols].fillna(0).values.astype(np.float32)
    y_te = test["ptf"].values

    ds_train = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols, free_raw_data=False)
    ds_val   = lgb.Dataset(X_vl, label=y_vl, feature_name=feature_cols, free_raw_data=False)

    params = {
        "objective":        "regression_l1",
        "learning_rate":    0.02,
        "num_leaves":       63,
        "min_data_in_leaf": 50,
        "verbose":          -1,
        "n_jobs":           -1,
    }

    callbacks = [
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(period=200),
    ]

    print("  Eğitim başlıyor...")
    model = lgb.train(
        params, ds_train,
        num_boost_round=3000,
        valid_sets=[ds_val],
        callbacks=callbacks,
    )

    val_pred  = model.predict(X_vl)
    test_pred = model.predict(X_te)

    val_mae  = mean_absolute_error(y_vl, val_pred)
    test_mae = mean_absolute_error(y_te, test_pred)

    # Fiziksel modelin tek başına test MAE'si (karşılaştırma için)
    if "phys_ptf" in test.columns:
        phys_mask = test["phys_ptf"].notna() & pd.Series(y_te).notna().values
        phys_test_mae = mean_absolute_error(
            np.array(y_te)[phys_mask], test["phys_ptf"].values[phys_mask]
        ) if phys_mask.any() else None
    else:
        phys_test_mae = None

    # Yıllık kırılım
    test["pred"] = test_pred
    test["ts"]   = pd.to_datetime(test["ts_hour"])
    yearly = {}
    for yr, g in test.groupby(test["ts"].dt.year):
        yearly[yr] = round(float(mean_absolute_error(g["ptf"], g["pred"])), 1)

    return {
        "model":          model,
        "feature_cols":   feature_cols,
        "val_mae":        val_mae,
        "test_mae":       test_mae,
        "phys_test_mae":  phys_test_mae,
        "yearly":         yearly,
        "best_iter":      model.best_iteration,
        "test_pred_df":   test[["ts_hour", "ptf", "pred", "phys_ptf"]],
    }


# ─── Entrypoint ───────────────────────────────────────────────────────────────

def main() -> None:
    import json

    print(f"[1/3] Master dataset yükleniyor...")
    df = pd.read_parquet(MASTER_PATH)
    df["ts_hour"] = pd.to_datetime(df["ts_hour"])

    # Rolling modelin en iyi split'i: 2020-2025 (Q2 hariç) train | 2025 Q2 val | 2026 test
    def assign_split(ts):
        yr, q = ts.year, ts.quarter
        if yr == 2026: return "test"
        if yr == 2025 and q == 2: return "val"
        return "train"   # 2020-2024 full + 2025 Q1/Q3/Q4
    df["split"] = df["ts_hour"].apply(assign_split)

    print(f"[2/3] Fiziksel feature'lar hesaplanıyor ({len(df):,} saat)...")
    df = add_physical_features(df)

    # Fiziksel modelin baseline MAE'si
    valid = df.dropna(subset=["ptf", "phys_ptf"])
    phys_all_mae = mean_absolute_error(valid["ptf"], valid["phys_ptf"])
    phys_test_mae_all = mean_absolute_error(
        valid[valid["split"]=="test"]["ptf"],
        valid[valid["split"]=="test"]["phys_ptf"]
    )
    print(f"\n  Fiziksel model baseline:")
    print(f"    Tüm dönem MAE : {phys_all_mae:,.1f} TL/MWh")
    print(f"    Test (2026)   : {phys_test_mae_all:,.1f} TL/MWh")

    print(f"\n[3/3] Hibrit LightGBM eğitimi...")
    results = train_hybrid(df)

    print(f"\n{'═'*55}")
    print(f"  HİBRİT MODEL SONUÇLARI")
    print(f"{'═'*55}")
    print(f"  Best iter      : {results['best_iter']}")
    print(f"  Val MAE        : {results['val_mae']:,.1f} TL/MWh")
    print(f"  Test MAE       : {results['test_mae']:,.1f} TL/MWh")
    print(f"\n  Karşılaştırma (Test / 2026):")
    print(f"    Fiziksel merit order : {phys_test_mae_all:>8,.1f} TL/MWh")
    print(f"    Eski LightGBM        : {359.3:>8,.1f} TL/MWh")
    print(f"    Hibrit LightGBM      : {results['test_mae']:>8,.1f} TL/MWh  ← bu model")
    print(f"\n  Yıllık test MAE:")
    for yr, mae in results["yearly"].items():
        print(f"    {yr}: {mae:,.1f}")

    # Rapor kaydet
    summary = {
        "val_mae":           round(results["val_mae"], 2),
        "test_mae":          round(results["test_mae"], 2),
        "phys_baseline_mae": round(phys_test_mae_all, 2),
        "old_lgbm_mae":      359.34,
        "best_iter":         results["best_iter"],
        "n_features":        len(results["feature_cols"]),
        "yearly":            results["yearly"],
    }
    out = REPORTS_DIR / "hybrid_model_results.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n  Rapor: {out}")

    # Feature importance — fiziksel feature'ların katkısı
    imp = pd.Series(
        results["model"].feature_importance("gain"),
        index=results["feature_cols"],
    ).sort_values(ascending=False)
    phys_imp = imp[[c for c in imp.index if c.startswith("phys_")]].sum()
    total_imp = imp.sum()
    print(f"\n  Fiziksel feature toplam gain katkısı: {phys_imp/total_imp*100:.1f}%")
    print(f"  En önemli fiziksel feature'lar:")
    for feat, val in imp[[c for c in imp.index if c.startswith("phys_")]].head(5).items():
        print(f"    {feat:<35}: {val/total_imp*100:.2f}%")


if __name__ == "__main__":
    main()

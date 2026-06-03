#!/usr/bin/env python3
"""
Leave-One-Group-Out (LOGO) Ablation Study for PTF Forecasting

Feature gruplarını tek tek kaldırıp MAE değişimini ölçer.
ΔMAE = baseline_MAE - ablated_MAE  → pozitif = o grup faydalı
                                   → negatif = o grup zararlı / gürültü

Kullanım:
    python run_logo_ablation.py
    python run_logo_ablation.py --quick   # hızlı mod (daha az estimator)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

# ──────────────────────────────────────────────────
# Project imports
# ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from rolling_ptf_forecast_system import (
    PROFILES,
    build_supervised_dataset,
    assign_split,
    log,
)

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────
# Feature Group Definitions
# ──────────────────────────────────────────────────
# Her grup: (grup_adı, açıklama, eşleşme kuralı fonksiyonu)
# Kural fonksiyonu → feature name → True ise o feature bu gruba dahil

def _matches(name: str, keywords: list[str], prefix: str = "", exclude: list[str] | None = None) -> bool:
    if prefix and not name.startswith(prefix):
        return False
    matched = any(k in name for k in keywords)
    if matched and exclude:
        if any(e in name for e in exclude):
            return False
    return matched


FEATURE_GROUPS: dict[str, dict[str, Any]] = {
    "PTF_history": {
        "description": "Anchor saatteki PTF geçmişi: lag, rolling stats, momentum",
        "english": "PTF price lags, rolling mean/std/min/max (6h→168h), zero/spike ratios, D1/week momentum",
        "match": lambda n: n.startswith("anchor_ptf_") or (n == "anchor_ptf"),
    },
    "Baseline_persistence": {
        "description": "Persistence baselines: dünün aynı saati, 2 gün önce, hafta önce",
        "english": "D-1 / D-2 / D-7 persistence prices and their momentum differences",
        "match": lambda n: "baseline" in n,
    },
    "Load_demand": {
        "description": "Yük tahmini ve net yük türevleri",
        "english": "TEİAŞ load forecast, net load after wind/solar/renewable, imbalance vs KGUP, ramp deltas",
        "match": lambda n: (
            ("load" in n and "coal" not in n and "block" not in n)
            or "net_load" in n
        ) and "grf" not in n and "kgup_import" not in n,
    },
    "KGUP_generation_plan": {
        "description": "KGÜP (Kesinleşmiş Gün Öncesi Üretim Programı): yakıt bazlı üretim planları",
        "english": "Day-ahead generation schedule by fuel: gas, coal, hydro, wind, solar + mix shares",
        "match": lambda n: (
            "kgup" in n
            or n in {"delivery_gas_share", "delivery_coal_share", "delivery_hydro_share",
                     "delivery_thermal_share", "delivery_renewable_share",
                     "delivery_gas_vs_renewable", "delivery_renewable_load_share",
                     "delivery_solar_load_share", "delivery_thermal_tightness_pressure"}
        ) and "delta" not in n,
    },
    "Ramp_dynamics": {
        "description": "Saatlik değişim (delta) ve rampa baskısı",
        "english": "1-hour deltas for load/KGUP/renewable, ramp tightness, ramp flags (morning/evening/night)",
        "match": lambda n: (
            "delta_1h" in n
            or "ramp" in n
            or n in {"delivery_morning_ramp_flag", "delivery_evening_ramp_flag", "delivery_night_block_flag"}
        ),
    },
    "GRF_gas_price": {
        "description": "GRF (Günlük Referans Fiyatı): doğalgaz maliyeti",
        "english": "Turkish natural gas daily reference price (TL/1000Sm³ + USD/MMBtu), lags, 7d/30d trends",
        "match": lambda n: "grf" in n,
    },
    "Exchange_rates": {
        "description": "Döviz kurları: USD/TRY, EUR/TRY",
        "english": "TCMB USD/TRY and EUR/TRY exchange rates, 1d lags, 7-day change/rolling mean",
        "match": lambda n: (
            "usd_try" in n or "eur_try" in n or "eur_usd" in n
        ) and "brent" not in n and "ttf" not in n,
    },
    "International_commodities": {
        "description": "Uluslararası emtia fiyatları: Brent, TTF, kömür, Henry Hub",
        "english": "Brent crude, TTF European gas, API2 coal, Henry Hub, 7d change / 30d rolling mean",
        "match": lambda n: (
            "brent" in n or ("ttf" in n and "grf" not in n) or "coal_api2" in n or "henry_hub" in n
        ) and "share" not in n,
    },
    "Temperature": {
        "description": "Sıcaklık ve ısı/soğutma derece günleri",
        "english": "Air temp, apparent temp, cooling/heating degree-days, load×temp interactions, 24h lags/deltas",
        "match": lambda n: (
            "temp" in n or "cool" in n or "heat" in n
            or n in {"delivery_temperature_2m", "delivery_apparent_temperature"}
        ) and "brent" not in n and "ttf" not in n,
    },
    "DAM_orderbook": {
        "description": "DAM emir defteri: alış/satış hacim, fiyatsız emirler, blok emirler",
        "english": "DAM bid/offer volumes, price-independent buy/sell, block buy volume, bid-sell balance/ratio",
        "match": lambda n: (
            "dam_bid" in n or "dam_sell" in n or "dam_block" in n or "dam_price_independent" in n
            or "dam_matched" in n or "price_independent" in n or "block_buy" in n or "block_to_matched" in n
            or "night_block_pressure" in n or "bid_sell" in n
        ),
    },
    "Calendar_time": {
        "description": "Takvim ve zaman özellikleri: saat, gün, ay, tatil, Ramazan",
        "english": "Hour/DOW/month (linear + sin/cos), weekend flag, public holiday flags, Ramadan proxy",
        "match": lambda n: (
            any(x in n for x in ["_hour", "_dow", "_month", "weekend", "holiday", "ramadan",
                                  "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"])
            and "ramp" not in n and "delta" not in n and "grf" not in n
        ) or n == "horizon",
    },
    "Lagged_realized": {
        "description": "Gecikmeli gerçekleşen veriler: SMF, gerçek tüketim, üretim, IDM, YAL/YAT",
        "english": "Realized post-settlement data lagged 24h/168h: SMF, real consumption, generation mix, IDM price, YAL/YAT regulation",
        "match": lambda n: (
            n.startswith("anchor_") and (
                "smf" in n or "gen_" in n or "real_consumption" in n
                or "yal_yat" in n or "idm" in n or "dam_matched" in n
            )
        ),
    },
    "Composite_interactions": {
        "description": "Türetilmiş kompozit özellikler: arz baskısı endeksleri, cross-market",
        "english": "Engineered: cheap_supply_pressure, gas_cost_pressure, load×gas/renewable, TTF×gas_share, TTF-GRF premium, Brent/TTF ratio",
        "match": lambda n: (
            n in {"delivery_cheap_supply_pressure", "delivery_cheap_minus_thermal",
                  "delivery_netload_x_thermal", "delivery_load_x_renewable", "delivery_load_x_gas",
                  "delivery_ttf_x_gas_share", "delivery_ttf_vs_grf_premium",
                  "delivery_brent_ttf_try_ratio", "delivery_gas_cost_pressure",
                  "delivery_thermal_cost_pressure", "delivery_gas_marginal_cost_pressure",
                  "delivery_wind_load_share"}
            or "gas_cost" in n or "thermal_cost" in n or "gas_marginal" in n
        ),
    },
}


def assign_group(feature_name: str, groups: dict) -> str | None:
    for gname, gmeta in groups.items():
        if gmeta["match"](feature_name):
            return gname
    return None


def build_group_map(feature_cols: list[str]) -> dict[str, list[str]]:
    group_map: dict[str, list[str]] = {g: [] for g in FEATURE_GROUPS}
    group_map["UNGROUPED"] = []
    for col in feature_cols:
        gname = assign_group(col, FEATURE_GROUPS)
        if gname:
            group_map[gname].append(col)
        else:
            group_map["UNGROUPED"].append(col)
    return group_map


def print_group_map(group_map: dict[str, list[str]], feat_imp: dict[str, float]) -> None:
    print("\n=== Feature Group Membership ===")
    for gname, cols in group_map.items():
        total_imp = sum(feat_imp.get(c, 0) for c in cols)
        print(f"  {gname}: n={len(cols)}, total_importance={total_imp:.0f}")
        for c in cols[:5]:
            print(f"    {c}: {feat_imp.get(c, 0):.0f}")
        if len(cols) > 5:
            print(f"    ... +{len(cols)-5} more")


# ──────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────

LGBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=127,
    max_depth=-1,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_jobs=-1,
    verbose=-1,
    random_state=42,
)

QUICK_PARAMS = dict(
    n_estimators=150,
    learning_rate=0.08,
    num_leaves=63,
    max_depth=8,
    min_child_samples=80,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    verbose=-1,
    random_state=42,
)


def train_and_eval(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    ablated_cols: list[str],
    params: dict,
    label: str,
) -> dict[str, Any]:
    """Train a model with ablated_cols zeroed out; return MAE metrics."""
    active_cols = [c for c in feature_cols if c not in ablated_cols]

    def prep(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = df[active_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        y_res = df["residual_target"].to_numpy(float)
        y_true = df["target_ptf"].to_numpy(float)
        base = df["baseline_d1_ptf"].to_numpy(float)
        return X, y_res, y_true, base

    X_tr, y_res_tr, _, _ = prep(train)
    X_val, y_res_val, y_val, base_val = prep(val)
    X_te, y_res_te, y_te, base_te = prep(test)

    model = LGBMRegressor(**params)
    model.fit(X_tr, y_res_tr)

    pred_val = model.predict(X_val) + base_val
    pred_te = model.predict(X_te) + base_te

    val_mae = float(mean_absolute_error(y_val, pred_val))
    te_mae = float(mean_absolute_error(y_te, pred_te))

    # Per-hour MAE on validation
    val_df = val.copy()
    val_df["_pred"] = pred_val
    val_df["_true"] = y_val
    hour_mae: dict[int, float] = {}
    for h in range(24):
        mask = val_df["delivery_ts_hour"].dt.hour == h if "delivery_ts_hour" in val_df.columns else pd.Series(True, index=val_df.index)
        if mask.sum() == 0:
            continue
        hour_mae[h] = float(mean_absolute_error(val_df.loc[mask, "_true"], val_df.loc[mask, "_pred"]))

    log(f"  [{label}] features={len(active_cols)} val_MAE={val_mae:.1f} test_MAE={te_mae:.1f}")
    return {
        "label": label,
        "n_features": len(active_cols),
        "val_mae": val_mae,
        "test_mae": te_mae,
        "hour_mae": hour_mae,
    }


# ──────────────────────────────────────────────────
# Main ablation loop
# ──────────────────────────────────────────────────

def run_ablation(quick: bool = False) -> None:
    params = QUICK_PARAMS if quick else LGBM_PARAMS

    # ── 1. Build supervised dataset ──────────────────
    log("Building supervised dataset (full_market profile)...")
    profile = next(p for p in PROFILES if p.name == "full_market")
    data, feature_cols, meta = build_supervised_dataset(profile)
    log(f"Dataset: {len(data):,} rows × {len(feature_cols)} features")

    train = data[data["split"] == "train"].copy()
    val = data[data["split"] == "validation"].copy()
    test = data[data["split"] == "test"].copy()
    log(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    # ── 2. Map features to groups ────────────────────
    group_map = build_group_map(feature_cols)
    ungrouped = group_map.pop("UNGROUPED", [])
    if ungrouped:
        log(f"UNGROUPED features ({len(ungrouped)}): {ungrouped}")

    # ── 3. Baseline (all features) ───────────────────
    log("\n=== BASELINE (all features) ===")
    baseline_result = train_and_eval(train, val, test, feature_cols, [], params, "BASELINE")
    baseline_val_mae = baseline_result["val_mae"]
    baseline_te_mae = baseline_result["test_mae"]
    baseline_hour_mae = baseline_result["hour_mae"]

    # ── 4. Leave-One-Group-Out ────────────────────────
    results = []
    for gname, cols in group_map.items():
        if not cols:
            log(f"[SKIP] {gname} — no features matched")
            continue
        log(f"\n=== ABLATE: {gname} ({len(cols)} features) ===")
        r = train_and_eval(train, val, test, feature_cols, cols, params, f"NO_{gname}")
        # ΔMAE = ablated - baseline: pozitif = kaldırılınca MAE arttı = grup faydalı
        delta_val = r["val_mae"] - baseline_val_mae
        delta_te = r["test_mae"] - baseline_te_mae
        # Per-hour delta (ablated - baseline: pozitif = grup o saatte faydalı)
        hour_delta = {}
        for h in range(24):
            bh = baseline_result["hour_mae"].get(h, np.nan)
            rh = r["hour_mae"].get(h, np.nan)
            if not (np.isnan(bh) or np.isnan(rh)):
                hour_delta[h] = round(rh - bh, 2)

        results.append({
            "group": gname,
            "description": FEATURE_GROUPS[gname]["description"],
            "english": FEATURE_GROUPS[gname]["english"],
            "n_features": len(cols),
            "feature_list": cols,
            "val_mae_ablated": round(r["val_mae"], 2),
            "val_mae_baseline": round(baseline_val_mae, 2),
            "delta_val_mae": round(delta_val, 2),
            "test_mae_ablated": round(r["test_mae"], 2),
            "test_mae_baseline": round(baseline_te_mae, 2),
            "delta_test_mae": round(delta_te, 2),
            "hour_delta_val_mae": hour_delta,
        })

    # ── 5. Sort by contribution ───────────────────────
    results.sort(key=lambda x: -x["delta_val_mae"])  # büyükten küçüğe: en faydalı önce

    # ── 6. Save JSON ──────────────────────────────────
    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "quick_mode": quick,
        "baseline": {
            "val_mae": round(baseline_val_mae, 2),
            "test_mae": round(baseline_te_mae, 2),
            "n_features": len(feature_cols),
            "hour_mae": {str(h): round(v, 2) for h, v in baseline_hour_mae.items()},
        },
        "groups": results,
    }
    json_path = REPORT_DIR / "logo_ablation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    log(f"\nJSON saved: {json_path}")

    # ── 7. Save Markdown report ───────────────────────
    md_path = REPORT_DIR / "logo_ablation_results.md"
    _write_markdown(output, md_path)
    log(f"Markdown saved: {md_path}")

    # ── 8. Print summary ─────────────────────────────
    print("\n" + "=" * 72)
    print("LEAVE-ONE-GROUP-OUT ABLATION SUMMARY")
    print(f"Baseline Val MAE: {baseline_val_mae:.1f} | Test MAE: {baseline_te_mae:.1f}")
    print("=" * 72)
    print(f"{'Group':<30} {'N':>4} {'ΔVAL':>8} {'ΔTEST':>8}  {'En kritik saat'}")
    print("-" * 72)
    for r in results:
        best_hours = sorted(r["hour_delta_val_mae"].items(), key=lambda x: -x[1])[:3]
        hours_str = ", ".join(f"s{h}(Δ{d:+.0f})" for h, d in best_hours)
        marker = "★" if r["delta_val_mae"] > 50 else ("▲" if r["delta_val_mae"] > 0 else "▼")
        print(f"{marker} {r['group']:<28} {r['n_features']:>4} {r['delta_val_mae']:>+8.1f} {r['delta_test_mae']:>+8.1f}  {hours_str}")
    print("=" * 72)
    print("★ = çok kritik (ΔMAE>50) | ▲ = faydalı (kaldırılınca MAE artar) | ▼ = zararlı/nötr")


def _write_markdown(output: dict, path: Path) -> None:
    baseline = output["baseline"]
    groups = output["groups"]

    lines = [
        "# Leave-One-Group-Out (LOGO) Ablation Raporu",
        "",
        f"**Oluşturulma:** `{output['generated_at']}`  ",
        f"**Mod:** {'Hızlı (quick)' if output['quick_mode'] else 'Tam'}  ",
        f"**Baseline Val MAE:** `{baseline['val_mae']}` TL/MWh  ",
        f"**Baseline Test MAE:** `{baseline['test_mae']}` TL/MWh  ",
        f"**Toplam feature:** `{baseline['n_features']}`",
        "",
        "---",
        "",
        "## Özet Tablo",
        "",
        "ΔMAE = Baseline − Ablated. **Pozitif** = grup kaldırılınca MAE arttı → grup faydalı.",
        "",
        "| Grup | N | ΔVAL MAE | ΔTEST MAE | Kritik Saatler |",
        "|------|--:|---------:|----------:|----------------|",
    ]
    for r in groups:
        worst = sorted(r["hour_delta_val_mae"].items(), key=lambda x: -abs(x[1]))[:3]
        hrs = " · ".join(f"s{h}(Δ{d:+.0f})" for h, d in worst)
        sign = "🔴" if r["delta_val_mae"] > 100 else ("🟠" if r["delta_val_mae"] > 30 else ("🟡" if r["delta_val_mae"] > 0 else "⚪"))
        lines.append(
            f"| {sign} **{r['group']}** | {r['n_features']} | "
            f"**{r['delta_val_mae']:+.1f}** | {r['delta_test_mae']:+.1f} | {hrs} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Grup Detayları",
        "",
    ]

    for r in groups:
        lines += [
            f"### {r['group']}",
            "",
            f"**Açıklama:** {r['description']}  ",
            f"**İngilizce:** {r['english']}  ",
            f"**Feature sayısı:** {r['n_features']}  ",
            f"**Val MAE ablated:** {r['val_mae_ablated']} (baseline: {r['val_mae_baseline']}) → ΔMAE = **{r['delta_val_mae']:+.1f}** ({'✅ faydalı' if r['delta_val_mae']>0 else '⚠️ zararlı/nötr'})  ",
            f"**Test MAE ablated:** {r['test_mae_ablated']} (baseline: {r['test_mae_baseline']}) → ΔMAE = **{r['delta_test_mae']:+.1f}**",
            "",
            "**Saate göre ΔMAE (validation):**",
            "",
            "| Saat | ΔMAE | Yorum |",
            "|------|-----:|-------|",
        ]
        for h in range(24):
            d = r["hour_delta_val_mae"].get(h, None)
            if d is None:
                continue
            if d > 20:
                comment = "🔴 Kritik katkı"
            elif d > 5:
                comment = "🟠 Faydalı"
            elif d > -5:
                comment = "⚪ Nötr"
            else:
                comment = "🟡 Negatif etki"
            lines.append(f"| {h:02d}:00 | {d:+.1f} | {comment} |")

        lines += [
            "",
            "**Features:**",
            "",
        ]
        for col in r["feature_list"][:20]:
            lines.append(f"- `{col}`")
        if len(r["feature_list"]) > 20:
            lines.append(f"- _...+{len(r['feature_list'])-20} more_")
        lines.append("")

    # Baseline saatlik MAE
    lines += [
        "---",
        "",
        "## Baseline Saatlik MAE (Validation)",
        "",
        "| Saat | MAE |",
        "|------|----:|",
    ]
    for h_str, mae in sorted(baseline["hour_mae"].items(), key=lambda x: int(x[0])):
        lines.append(f"| {int(h_str):02d}:00 | {mae:.1f} |")

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LOGO Ablation Study")
    parser.add_argument("--quick", action="store_true", help="Hızlı mod: daha az estimator")
    args = parser.parse_args()
    t0 = time.time()
    run_ablation(quick=args.quick)
    log(f"\nToplam süre: {(time.time()-t0)/60:.1f} dakika")

#!/usr/bin/env python3
"""Walk-forward monthly retrain — 2026 için gerçekçi canlı performans ölçümü.

Her ay:
  1. O aya kadar olan tüm veriyle model eğit
  2. Son 60 günü validation olarak ayır (alpha kalibrasyonu)
  3. Hedef ayı tahmin et
  4. Gerçek PTF ile karşılaştır

Bu, production'da "her ay retrain" yapıldığında elde edeceğimiz
gerçek performansı simüle eder.

Kullanım:
  python3 run_walk_forward_2026.py            # tüm aylar, quick mode
  python3 run_walk_forward_2026.py --full     # 700 estimator (yavaş ama doğru)
  python3 run_walk_forward_2026.py --month 2  # sadece Şubat
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parent

# 2026'nın her ayı: (tahmin başlangıcı, tahmin sonu, ay adı)
MONTHS_2026 = [
    ("2026-01-01", "2026-02-01", "Ocak 2026"),
    ("2026-02-01", "2026-03-01", "Şubat 2026"),
    ("2026-03-01", "2026-04-01", "Mart 2026"),
    ("2026-04-01", "2026-05-01", "Nisan 2026"),
    ("2026-05-01", "2026-06-01", "Mayıs 2026"),
    ("2026-06-01", "2026-06-04", "Haziran 2026 (kısmi)"),
]

VAL_WINDOW_DAYS = 60  # son 60 gün = validation (alpha kalibrasyonu)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def make_split_fn(target_start: str, target_end: str):
    """Her ay için dinamik split fonksiyonu üretir."""
    t_start = pd.Timestamp(target_start)
    t_end = pd.Timestamp(target_end)
    val_end = t_start  # validation = target ayından hemen önce
    val_start = t_start - pd.Timedelta(days=VAL_WINDOW_DAYS)  # son 60 gün

    def assign_split(ts: pd.Series) -> pd.Series:
        split = pd.Series("ignore", index=ts.index, dtype="object")
        # Train: 2020 başından val_start'a kadar
        split[(ts >= "2020-01-01") & (ts < val_start)] = "train"
        # Val: son 60 gün (alpha kalibrasyonu için)
        split[(ts >= val_start) & (ts < val_end)] = "validation"
        # Test: hedef ay
        split[(ts >= t_start) & (ts < t_end)] = "test"
        return split

    return assign_split


def run_month(target_start: str, target_end: str, month_name: str, quick: bool) -> dict:
    """Tek bir ay için retrain + tahmin + metrik."""
    import rolling_ptf_forecast_system as rps

    log(f"{'='*55}")
    log(f"{month_name} — eğitim başlıyor")
    log(f"  Train sonu : {pd.Timestamp(target_start) - pd.Timedelta(days=VAL_WINDOW_DAYS+1):%Y-%m-%d}")
    log(f"  Val        : {pd.Timestamp(target_start) - pd.Timedelta(days=VAL_WINDOW_DAYS):%Y-%m-%d} → {pd.Timestamp(target_start):%Y-%m-%d}")
    log(f"  Test       : {target_start} → {target_end}")

    # assign_split'i bu ay için dinamik olarak override et
    rps.assign_split = make_split_fn(target_start, target_end)

    # Eğit
    profile = next(p for p in rps.PROFILES if p.name == "full_market")
    result = rps.fit_profile(profile, quick=quick)

    metrics = result["metrics"]
    vm = metrics["validation_model"]
    tm = metrics["test_model"]
    vp = metrics["validation_persistence"]
    tp = metrics["test_persistence"]

    # Test satır sayısını da alalım
    test_rows = result["meta"].get("split_counts", {}).get("test", "?")

    log(f"  Val  MAE: {vm['mae']:.1f}  (persistence: {vp['mae']:.1f})")
    log(f"  Test MAE: {tm['mae']:.1f}  (persistence: {tp['mae']:.1f}, iyileşme: {100*(1-tm['mae']/tp['mae']):.1f}%)")

    return {
        "month": month_name,
        "target_start": target_start,
        "target_end": target_end,
        "val_mae": round(vm["mae"], 2),
        "val_persistence": round(vp["mae"], 2),
        "test_mae": round(tm["mae"], 2),
        "test_persistence": round(tp["mae"], 2),
        "test_improvement_pct": round((1 - tm["mae"] / tp["mae"]) * 100, 1) if tp["mae"] > 0 else 0,
        "test_rows": test_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="700 estimator (daha doğru ama yavaş)")
    ap.add_argument("--month", type=int, default=None, help="Sadece belirli bir ay (1-6)")
    args = ap.parse_args()

    quick = not args.full
    months = MONTHS_2026
    if args.month:
        months = [MONTHS_2026[args.month - 1]]

    log(f"Walk-forward 2026 başlıyor — {'quick' if quick else 'full'} mode, {len(months)} ay")

    results = []
    for target_start, target_end, month_name in months:
        try:
            r = run_month(target_start, target_end, month_name, quick=quick)
            results.append(r)
        except Exception as e:
            log(f"HATA — {month_name}: {e}")
            import traceback; traceback.print_exc()

    if not results:
        log("Hiç sonuç yok.")
        return

    # Özet tablo
    print("\n" + "="*70)
    print("WALK-FORWARD 2026 SONUÇLARI")
    print("="*70)
    print(f"{'Ay':<28} {'Test MAE':>9} {'Persistence':>12} {'İyileşme':>9}")
    print("-"*70)
    for r in results:
        print(f"{r['month']:<28} {r['test_mae']:>9.1f} {r['test_persistence']:>12.1f} {r['test_improvement_pct']:>8.1f}%")

    print("-"*70)

    # Ağırlıklı ortalama (satır sayısına göre)
    # (test_rows None ise basit ortalama)
    all_mae = [r["test_mae"] for r in results]
    all_pers = [r["test_persistence"] for r in results]
    avg_mae = np.mean(all_mae)
    avg_pers = np.mean(all_pers)
    overall_improvement = (1 - avg_mae / avg_pers) * 100 if avg_pers > 0 else 0

    print(f"{'ORTALAMA (tüm aylar)':<28} {avg_mae:>9.1f} {avg_pers:>12.1f} {overall_improvement:>8.1f}%")
    print("="*70)
    print()

    # Sabit model (single train) ile karşılaştırma
    print("KARŞILAŞTIRMA:")
    print(f"  Sabit model (2020-24 train):   Test MAE = 381.3 TL/MWh")
    print(f"  Walk-forward (aylık retrain):  Test MAE = {avg_mae:.1f} TL/MWh")
    diff = avg_mae - 381.26
    print(f"  Fark: {diff:+.1f} TL/MWh  ({'iyileşme' if diff < 0 else 'kötüleşme'})")

    # JSON kaydet
    out = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "mode": "quick" if quick else "full",
        "val_window_days": VAL_WINDOW_DAYS,
        "results": results,
        "summary": {
            "avg_test_mae": round(avg_mae, 2),
            "avg_persistence": round(avg_pers, 2),
            "avg_improvement_pct": round(overall_improvement, 1),
            "baseline_single_train_mae": 381.26,
        }
    }
    out_path = PROJECT_ROOT / "reports" / "walk_forward_2026_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    log(f"Sonuçlar kaydedildi: {out_path}")


if __name__ == "__main__":
    main()

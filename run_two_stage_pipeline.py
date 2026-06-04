#!/usr/bin/env python3
"""
İki Aşamalı PTF Tahmin Pipeline'ı — Günlük Operasyonel Çalıştırıcı.

Sabah koşusu (11:30):
    1. data_loader → sabah snapshot (PTF/SMF/RES/kesinti/baraj/emtia)
    2. physical_features → MC matrisi + SDI + NRM özellik tablosu
    3. TwoStagePTFForecaster → 24 saatlik eğri tahmini + Pyomo LP takas
    4. RTCorrectionModule → Kalman bias düzeltmesi
    5. data/live/two_stage_forecast_latest.csv yaz

Akşam koşusu (14:30 — GÖP sonuçları açıklandıktan sonra):
    6. Gerçek PTF çek
    7. RTCorrectionModule.update_day() → P matrislerini güncelle
    8. data/live/rt_correction_state.json kaydet

Kullanım:
    python run_two_stage_pipeline.py forecast          # sabah: sadece tahmin
    python run_two_stage_pipeline.py update            # akşam: Kalman güncelle
    python run_two_stage_pipeline.py full              # ikisi birden
    python run_two_stage_pipeline.py warmstart         # Kalman ilk başlatma
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
LIVE_DIR     = PROJECT_ROOT / "data" / "live"
MODEL_DIR    = PROJECT_ROOT / "models" / "two_stage_v1"
STATE_PATH   = LIVE_DIR / "rt_correction_state.json"
FORECAST_CSV = LIVE_DIR / "two_stage_forecast_latest.csv"
HISTORY_CSV  = LIVE_DIR / "forecast_vs_actual_all.csv"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Adım 1–4: Tahmin koşusu ──────────────────────────────────────────────────

def run_forecast(target_date: str | None = None) -> pd.DataFrame:
    """
    Sabah 11:30 koşusu: snapshot → özellik → tahmin → Kalman düzeltme.

    Returns:
        24 saatlik tahmin DataFrame'i (FORECAST_CSV'ye de yazılır)
    """
    target = target_date or str(date.today() + timedelta(days=1))
    log.info(f"=== TAHMIN KOŞUSU: {target} ===")

    # 1. Sabah snapshot
    log.info("Adım 1/4: Sabah snapshot çekiliyor...")
    snapshot = _safe_fetch_snapshot(target)

    # 2. Model yükle veya fallback
    log.info("Adım 2/4: Model yükleniyor...")
    forecaster = _load_or_build_forecaster()

    # 3. Tahmin
    log.info("Adım 3/4: 24 saatlik tahmin üretiliyor...")
    try:
        pred_df = forecaster.predict_day(target)
        pred_df["lp_price"] = pred_df["predicted_ptf"]
    except Exception as exc:
        log.warning(f"Stage 1+2 tahmin hatası ({exc}), fallback moda geçiliyor...")
        pred_df = _fallback_forecast(target)

    # 4. Kalman düzeltmesi
    log.info("Adım 4/4: Kalman düzeltmesi uygulanıyor...")
    from realtime_correction import RTCorrectionModule
    module = RTCorrectionModule()
    module.load(STATE_PATH)

    corrections = module.corrections_for_tomorrow()
    pred_df["kalman_bias"] = pred_df["hour"].map(corrections).fillna(0.0)
    pred_df["corrected_ptf"] = pred_df["predicted_ptf"] + pred_df["kalman_bias"]
    pred_df["predicted_ptf"] = pred_df["corrected_ptf"]   # ana sütunu güncelle

    # Teslimat zaman damgası
    pred_df["delivery_ts_hour"] = pred_df["hour"].apply(
        lambda h: pd.Timestamp(f"{target}T{h:02d}:00:00", tz="Europe/Istanbul")
    )
    pred_df["anchor_ts_hour"] = pd.Timestamp.now(tz="Europe/Istanbul")

    # Persistence (önceki günün PTF'si)
    pred_df["baseline_d1_ptf"] = _get_persistence_baseline(target)

    # Yaz
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(FORECAST_CSV, index=False)
    log.info(f"Tahmin kaydedildi → {FORECAST_CSV.relative_to(PROJECT_ROOT)}")
    log.info(f"24 saat: ort={pred_df['predicted_ptf'].mean():.0f} TL  "
             f"peak={pred_df['predicted_ptf'].max():.0f} TL  "
             f"bias_avg={pred_df['kalman_bias'].mean():.1f} TL")
    return pred_df


# ─── Adım 5–7: Kalman güncelleme koşusu ──────────────────────────────────────

def run_kalman_update(delivery_date: str | None = None) -> None:
    """
    Akşam 14:30 koşusu: gerçek GÖP sonuçlarıyla Kalman P matrisini güncelle.

    Args:
        delivery_date: GÖP sonuçlarının ait olduğu teslimat günü (varsayılan: bugün)
    """
    delivery = delivery_date or str(date.today())
    log.info(f"=== KALMAN GÜNCELLEME: {delivery} ===")

    # Gerçek PTF
    actual = _fetch_actual_ptf(delivery)
    if actual is None or len(actual) == 0:
        log.warning(f"{delivery} için gerçek PTF verisi bulunamadı. Güncelleme atlandı.")
        return

    # Mevcut tahmini yükle (dün sabah yazılan CSV)
    predicted = _load_predicted_ptf(delivery)
    if predicted is None:
        log.warning("Önceki tahmin CSV'si bulunamadı. Güncelleme atlandı.")
        return

    # Kalman güncelle
    from realtime_correction import RTCorrectionModule
    module = RTCorrectionModule()
    module.load(STATE_PATH)
    module.update_day(actual, predicted, delivery_date=delivery)
    module.save(STATE_PATH)

    # Geçmiş arşivine ekle
    _append_to_history(delivery, actual, predicted)

    # Özet
    em = module.error_matrix()
    mae = em.apply(lambda r: abs(r["current_bias_tl"]), axis=1).mean() if not em.empty else 0
    log.info(f"Kalman güncellendi. Ort. bias = {mae:.1f} TL | {module.summary(top_n=3)[:150]}")


# ─── Warm-start ───────────────────────────────────────────────────────────────

def run_warmstart() -> None:
    """Kalman filtrelerini tarihsel parquet verisiyle ısıt."""
    log.info("=== KALMAN WARM-START ===")
    from realtime_correction import RTCorrectionModule, warmstart_from_curve_features
    module = RTCorrectionModule()
    warmstart_from_curve_features(module)
    module.save(STATE_PATH)
    log.info("Warm-start tamamlandı.")
    print(module.summary(top_n=10))


# ─── Yardımcılar ──────────────────────────────────────────────────────────────

def _safe_fetch_snapshot(target_date: str) -> dict:
    """data_loader.fetch_morning_snapshot() — hata toleranslı."""
    try:
        from data_loader import fetch_morning_snapshot
        return fetch_morning_snapshot(target_date)
    except Exception as exc:
        log.warning(f"Snapshot çekilemedi ({exc}), boş döndürülüyor.")
        return {}


def _load_or_build_forecaster():
    """Kaydedilmiş modeli yükle; yoksa veri setinden eğit."""
    from models import TwoStagePTFForecaster

    forecaster = TwoStagePTFForecaster(backend="lgbm", solver_name="appsi_highs")

    if (MODEL_DIR / "supply_model.pkl").exists():
        log.info(f"Mevcut model yükleniyor: {MODEL_DIR}")
        forecaster.load_model(MODEL_DIR)
    else:
        log.info("Kaydedilmiş model bulunamadı — eğitim başlıyor...")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        forecaster.fit_from_files(save_to=MODEL_DIR)
        log.info(f"Model kaydedildi: {MODEL_DIR}")

    return forecaster


def _fallback_forecast(target_date: str) -> pd.DataFrame:
    """
    Tahmin başarısız olursa basit persistence (dünün değerleri) kullan.
    Sütunlar: hour, predicted_ptf
    """
    log.info("Fallback: dünün PTF verileri kullanılıyor (persistence).")
    prev = _get_persistence_24h(target_date)
    return pd.DataFrame({
        "hour": list(range(24)),
        "predicted_ptf": prev if prev is not None else [500.0] * 24,
        "lp_status": "fallback_persistence",
        "clearing_volume": [0.0] * 24,
    })


def _get_persistence_baseline(target_date: str) -> list[float]:
    """Önceki günün saatlik PTF değerlerini döndür (24 eleman)."""
    vals = _get_persistence_24h(target_date)
    return vals if vals is not None else [float("nan")] * 24


def _get_persistence_24h(target_date: str) -> list[float] | None:
    """target_date - 1 günün saatlik PTF'sini master dataset'ten al."""
    try:
        master = _load_master()
        target_ts = pd.Timestamp(target_date, tz="Europe/Istanbul")
        prev_date = target_ts - pd.Timedelta(days=1)
        day_data = master[master["ts_hour"].dt.date == prev_date.date()].sort_values("ts_hour")
        ptf_col = next((c for c in ["ptf_price","ptf","marketTradePrice"] if c in day_data.columns), None)
        if ptf_col and len(day_data) >= 24:
            return day_data[ptf_col].iloc[:24].tolist()
    except Exception as exc:
        log.debug(f"Persistence yüklenemedi: {exc}")
    return None


def _load_master() -> pd.DataFrame:
    p = PROJECT_ROOT / "data" / "master" / "master_hourly_v1.parquet"
    df = pd.read_parquet(p)
    df["ts_hour"] = pd.to_datetime(df["ts_hour"], utc=True).dt.tz_convert("Europe/Istanbul")
    return df


def _fetch_actual_ptf(delivery_date: str) -> dict[int, float] | None:
    """
    Gerçekleşen PTF'yi çek.
    1. eptr2 MCP endpoint (canlı)
    2. Master dataset (arşiv)
    """
    try:
        from data_loader import fetch_mcp
        df = fetch_mcp(delivery_date, delivery_date)
        hour_col = next((c for c in ["hour","ts_hour","date"] if c in df.columns), None)
        ptf_col  = next((c for c in ["ptf","marketTradePrice","price"] if c in df.columns), None)
        if hour_col and ptf_col:
            df["h"] = pd.to_datetime(df[hour_col], errors="coerce").dt.hour
            return dict(zip(df["h"], df[ptf_col].astype(float)))
    except Exception as exc:
        log.debug(f"eptr2 MCP başarısız ({exc}), master'a geçiliyor.")

    try:
        master = _load_master()
        day = master[master["ts_hour"].dt.date == pd.Timestamp(delivery_date).date()]
        ptf_col = next((c for c in ["ptf_price","ptf"] if c in day.columns), None)
        if ptf_col:
            day = day.sort_values("ts_hour")
            return dict(enumerate(day[ptf_col].values[:24]))
    except Exception as exc:
        log.debug(f"Master PTF başarısız: {exc}")

    return None


def _load_predicted_ptf(delivery_date: str) -> dict[int, float] | None:
    """Önceden tahmin edilmiş değerleri FORECAST_CSV veya arşivden al."""
    if FORECAST_CSV.exists():
        try:
            df = pd.read_csv(FORECAST_CSV)
            if "hour" in df.columns and "predicted_ptf" in df.columns:
                return dict(zip(df["hour"].astype(int), df["predicted_ptf"].astype(float)))
        except Exception:
            pass
    return None


def _append_to_history(
    delivery_date: str,
    actual: dict[int, float],
    predicted: dict[int, float],
) -> None:
    """Gerçek vs tahmin karşılaştırmasını geçmiş arşivine ekle."""
    rows = []
    for h in range(24):
        a = actual.get(h, float("nan"))
        p = predicted.get(h, float("nan"))
        rows.append({
            "ts_hour": pd.Timestamp(f"{delivery_date}T{h:02d}:00:00", tz="Europe/Istanbul"),
            "actual_ptf": a,
            "predicted_ptf": p,
            "error": p - a,
        })
    new_df = pd.DataFrame(rows)
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    if HISTORY_CSV.exists():
        old = pd.read_csv(HISTORY_CSV)
        combined = pd.concat([old, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ts_hour"]).sort_values("ts_hour")
        combined.to_csv(HISTORY_CSV, index=False)
    else:
        new_df.to_csv(HISTORY_CSV, index=False)
    log.info(f"Geçmiş arşivi güncellendi → {HISTORY_CSV.relative_to(PROJECT_ROOT)}")


# ─── Ana giriş noktası ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="İki Aşamalı PTF Pipeline")
    parser.add_argument(
        "mode",
        choices=["forecast", "update", "full", "warmstart"],
        help="forecast=sabah tahmini | update=akşam Kalman | full=ikisi | warmstart=ilk başlatma",
    )
    parser.add_argument("--date", default=None, help="Teslimat tarihi YYYY-MM-DD (varsayılan: yarın)")
    args = parser.parse_args()

    if args.mode == "warmstart":
        run_warmstart()

    elif args.mode == "forecast":
        run_forecast(args.date)

    elif args.mode == "update":
        # Güncelleme için teslimat tarihi = bugün (dün sabah tahmin edildi)
        delivery = args.date or str(date.today())
        run_kalman_update(delivery)

    elif args.mode == "full":
        # Sabah + akşam
        target = args.date or str(date.today() + timedelta(days=1))
        pred_df = run_forecast(target)
        delivery = str(date.today())
        run_kalman_update(delivery)


if __name__ == "__main__":
    main()

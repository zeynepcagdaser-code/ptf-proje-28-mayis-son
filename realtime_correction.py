#!/usr/bin/env python3
"""
Gerçek Zamanlı PTF Düzeltme Modülü.

GÖP sonuçları açıklandıkça (14:00) veya her saat başı GİP fiyatları
gerçekleştikçe, Kalman Filtresi'nin durum kovaryans matrisini (P) dinamik
olarak günceller ve ertesi gün tahminlerine uygulanacak bias düzeltmelerini
sağlar.

İki çalışma modu:
    BATCH       : 14:00'de GÖP sonuçlarının tamamı tek seferde beslenir (24 saat).
    INCREMENTAL : Her saat başı gerçekleşen GİP fiyatı geldiğinde güncelleme yapılır.

Kalman durum vektörü (24 boyutlu):
    x = [bias_h0, bias_h1, ..., bias_h23]   TL/MWh

    bias_h  = Gerçekleşen PTF − Model Tahmini (pozitif = model alttan tahmin)

Kovaryans matrisi P (24×24):
    Köşegen   : Her saatin bias belirsizliği
    Köşegen dışı: Saatler arası bias korelasyonu (zaman içinde doğal gelişir)

Kalman parametrelerinin saate göre kalibrasyonu:
    Saatler 7, 9, 11, 13, 14 → Q_h = 0.1 TL² (kararlı, RES fazlası dönemleri)
    Saatler 17, 19, 21-22    → Q_h = 5.0 TL² (volatil, blok teklif geçiş saatleri)
    Diğerleri                → Q_h = 0.5 TL²

    R_h = tarihsel saatlik std² (LP ölçüm gürültüsü)

Günlük operasyonel akış:
    11:30 → predict_day() → 24 saatlik ham tahmin
    14:00 → module.update_day(actual_ptf, predicted_ptf)  ← bu modül
    14:05 → module.corrections_for_tomorrow()             ← bias vektörü
    15:00 → ertesi gün nihai tahmini = ham + bias
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
log = logging.getLogger(__name__)

_N_HOURS = 24

# ─── Saat bazlı Kalman parametre kalibrasyonu ─────────────────────────────────
# Tarihsel LP hata istatistiklerinden türetildi (2026-05-18/24 haftalık verisi)

_PROCESS_NOISE_BY_HOUR: dict[int, float] = {
    # Kararlı saatler (bias std ≈ 0)
    7: 0.05, 9: 0.05, 11: 0.05, 13: 0.05, 14: 0.05,
    # Volatil saatler (blok teklif, spike geçiş)
    17: 5.0, 19: 8.0, 21: 4.0, 22: 5.0,
}
_DEFAULT_PROCESS_NOISE = 0.5

# Tarihsel ölçüm gürültüsü (LP tahmin std² per saat)
_MEASUREMENT_NOISE_BY_HOUR: dict[int, float] = {
    0: 20.11**2,  1: 0.61**2,   2: 10.16**2,  3: 18.12**2,
    4: 1.74**2,   5: 0.93**2,   6: 0.38**2,   7: 0.01**2,
    8: 3.34**2,   9: 0.01**2,  10: 1.88**2,  11: 0.01**2,
   12: 0.82**2,  13: 0.01**2,  14: 0.01**2,  15: 3.15**2,
   16: 0.37**2,  17: 38.66**2, 18: 2.58**2,  19: 77.39**2,
   20: 0.09**2,  21: 24.20**2, 22: 44.75**2, 23: 6.41**2,
}
_DEFAULT_MEASUREMENT_NOISE = 4.0**2

# Spike kırpma eşiği (blok teklif kaynaklı büyük artıkları izole eder)
_MAX_RESIDUAL_BY_HOUR: dict[int, float] = {
    19: 40.0, 22: 50.0, 21: 30.0, 17: 25.0,
}
_DEFAULT_MAX_RESIDUAL = 20.0


# ─── Yardımcı: tek saatlik Kalman filtresi ────────────────────────────────────

class _SingleHourKF:
    """filterpy KalmanFilter sarmalayıcısı — tek saat için."""

    def __init__(self, hour: int) -> None:
        from filterpy.kalman import KalmanFilter
        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.x = np.array([[0.0]])
        kf.F = np.array([[1.0]])
        kf.H = np.array([[1.0]])
        q = _PROCESS_NOISE_BY_HOUR.get(hour, _DEFAULT_PROCESS_NOISE)
        r = _MEASUREMENT_NOISE_BY_HOUR.get(hour, _DEFAULT_MEASUREMENT_NOISE)
        kf.P = np.array([[r * 2]])   # başlangıç belirsizliği
        kf.Q = np.array([[q]])
        kf.R = np.array([[r]])
        self._kf = kf
        self.hour = hour
        self.max_r = _MAX_RESIDUAL_BY_HOUR.get(hour, _DEFAULT_MAX_RESIDUAL)
        self.n_updates = 0
        self.n_clipped = 0

    @property
    def bias(self) -> float:
        return float(self._kf.x[0, 0])

    @property
    def uncertainty(self) -> float:
        return float(np.sqrt(max(0, self._kf.P[0, 0])))

    @property
    def P_scalar(self) -> float:
        return float(self._kf.P[0, 0])

    def update(self, actual: float, predicted: float) -> float:
        raw = actual - predicted
        clipped = float(np.clip(raw, -self.max_r, self.max_r))
        self.n_updates += 1
        if abs(raw) > self.max_r:
            self.n_clipped += 1
        self._kf.predict()
        self._kf.update(np.array([[clipped]]))
        return self.bias

    def to_dict(self) -> dict:
        return {
            "hour": self.hour,
            "bias": self.bias,
            "uncertainty": self.uncertainty,
            "P": self._kf.P[0, 0],
            "n_updates": self.n_updates,
            "n_clipped": self.n_clipped,
        }

    def from_dict(self, d: dict) -> None:
        self._kf.x[0, 0] = d["bias"]
        self._kf.P[0, 0] = d["P"]
        self.n_updates = d.get("n_updates", 0)
        self.n_clipped = d.get("n_clipped", 0)


# ─── Ana modül ────────────────────────────────────────────────────────────────

class RTCorrectionModule:
    """
    Gerçek zamanlı PTF tahmin düzeltme sistemi.

    Her saat için bağımsız bir Kalman Filtresi çalıştırır (24 filtre).
    P matrisleri (durum kovaryansı) her güncellemede dinamik olarak değişir:
      - Büyük hata → P artar → filtre daha esnek, sonraki ölçüme daha çok güvenir
      - Tutarlı küçük hata → P azalır → filtre kararlı, bias'a güvenir

    Çapraz-saat kovaryans kanalı:
      Saatler 17-23 (spike grubu) ve saatler 0-6 (gece grubu) arasında
      ortak bir bias hareketi tespit edildiğinde P_cross artar. Bu bilgi
      tek saatin verisi az olduğunda diğer saatleri de düzeltmek için kullanılır.

    Kullanım:
        module = RTCorrectionModule()
        module.load("data/live/rt_correction_state.json")  # önceki durumu yükle

        # 14:00 GÖP sonuçları geldiğinde
        module.update_day(actual_dict, predicted_dict)

        # Ertesi gün tahminlerini düzelt
        corrections = module.corrections_for_tomorrow()     # {hour: bias}
        corrected_ptf = {h: predicted[h] + corrections[h] for h in range(24)}

        module.save("data/live/rt_correction_state.json")
    """

    def __init__(self, log_path: Path | None = None) -> None:
        try:
            from filterpy.kalman import KalmanFilter  # noqa: F401
        except ImportError as exc:
            raise ImportError("pip install 'filterpy>=1.4.5'") from exc

        self._filters: dict[int, _SingleHourKF] = {
            h: _SingleHourKF(h) for h in range(_N_HOURS)
        }

        # Çapraz-saat kovaryans (spike saatleri birlikte hareket eder)
        # _P_cross[i][j]: saat i bias'ındaki değişimin saat j'ye taşınan kısmı
        self._P_cross: np.ndarray = np.zeros((_N_HOURS, _N_HOURS))

        # Hata geçmişi (log + istatistik için)
        self._history: list[dict] = []
        self.log_path = log_path or PROJECT_ROOT / "data" / "live" / "rt_correction_log.csv"

    # ── Güncelleme fonksiyonları ───────────────────────────────────────────────

    def update_hour(
        self,
        hour: int,
        actual_ptf: float,
        predicted_ptf: float,
        delivery_date: str | date | None = None,
    ) -> float:
        """
        Tek saat için Kalman güncellemesi (incremental mod).

        GİP (gün içi piyasa) saatlik fiyatları gerçekleştikçe çağrılır.

        Returns:
            Güncellenmiş saat bias'ı (TL/MWh)
        """
        kf = self._filters[hour]
        old_P = kf.P_scalar
        bias = kf.update(actual_ptf, predicted_ptf)
        new_P = kf.P_scalar

        record = {
            "ts": datetime.now().isoformat(),
            "delivery_date": str(delivery_date or date.today()),
            "hour": hour,
            "actual_ptf": actual_ptf,
            "predicted_ptf": predicted_ptf,
            "raw_residual": actual_ptf - predicted_ptf,
            "new_bias": bias,
            "P_before": old_P,
            "P_after": new_P,
            "uncertainty": kf.uncertainty,
        }
        self._history.append(record)
        self._update_cross_covariance(hour, actual_ptf - predicted_ptf)

        log.debug(f"Saat {hour:02d}: actual={actual_ptf:.2f}, pred={predicted_ptf:.2f}, "
                  f"bias={bias:+.2f}, P={new_P:.3f}")
        return bias

    def update_day(
        self,
        actual_ptf: dict[int, float] | np.ndarray | list[float],
        predicted_ptf: dict[int, float] | np.ndarray | list[float],
        delivery_date: str | date | None = None,
    ) -> dict[int, float]:
        """
        24 saatlik GÖP sonuçlarıyla toplu Kalman güncellemesi (batch mod).

        14:00'de EPİAŞ GÖP sonuçlarını açıkladığında çağrılır.

        Args:
            actual_ptf   : Gerçekleşen saatlik fiyatlar (dict {saat: fiyat} veya 24 elemanlı dizi)
            predicted_ptf: Model tahminleri

        Returns:
            {hour: updated_bias} — güncellenmiş 24 saatlik bias vektörü
        """
        if not isinstance(actual_ptf, dict):
            actual_ptf = {h: float(actual_ptf[h]) for h in range(_N_HOURS) if h < len(actual_ptf)}
        if not isinstance(predicted_ptf, dict):
            predicted_ptf = {h: float(predicted_ptf[h]) for h in range(_N_HOURS) if h < len(predicted_ptf)}

        updated_biases = {}
        for h in range(_N_HOURS):
            a = actual_ptf.get(h, float("nan"))
            p = predicted_ptf.get(h, float("nan"))
            if np.isnan(a) or np.isnan(p):
                updated_biases[h] = self._filters[h].bias
                continue
            updated_biases[h] = self.update_hour(h, a, p, delivery_date)

        log.info(
            f"Günlük güncelleme tamamlandı ({delivery_date}) → "
            f"bias aralığı [{min(updated_biases.values()):+.1f}, {max(updated_biases.values()):+.1f}] TL"
        )
        self._flush_log()
        return updated_biases

    # ── Çapraz-saat kovaryans ─────────────────────────────────────────────────

    def _update_cross_covariance(self, hour: int, raw_residual: float) -> None:
        """
        Spike grubundaki (17-23) büyük hatalar, aynı gruptaki diğer saatlerin
        P matrislerini biraz artırır (nedensel belirsizlik yayılımı).
        """
        spike_hours = {17, 18, 19, 20, 21, 22, 23}
        off_peak_hours = {0, 1, 2, 3, 4, 5, 6}
        group: set[int] = set()
        if hour in spike_hours:
            group = spike_hours
        elif hour in off_peak_hours:
            group = off_peak_hours

        if group and abs(raw_residual) > 20:
            # Büyük hata: aynı gruptaki saatlerin belirsizliğini artır
            propagation = 0.2 * raw_residual**2
            for peer in group:
                if peer != hour:
                    kf = self._filters[peer]
                    kf._kf.P[0, 0] += propagation * 0.1   # yumuşak yayılım
                    self._P_cross[hour, peer] += propagation

    # ── Düzeltme çıktıları ────────────────────────────────────────────────────

    def corrections_for_tomorrow(self) -> dict[int, float]:
        """
        Ertesi gün tahminlerine uygulanacak bias düzeltmelerini döndür.

        Returns:
            {hour: bias_tl_mwh}  — tüm 24 saat için TL/MWh cinsinden düzeltme
        """
        return {h: self._filters[h].bias for h in range(_N_HOURS)}

    def apply_corrections(
        self,
        predictions: dict[int, float] | np.ndarray | pd.Series,
    ) -> np.ndarray:
        """
        Tahmin dizisine bias düzeltmeleri uygula.

        Args:
            predictions: 24 elemanlı saatlik PTF tahmini

        Returns:
            24 elemanlı düzeltilmiş PTF tahmini (np.ndarray)
        """
        if isinstance(predictions, dict):
            arr = np.array([predictions.get(h, 0.0) for h in range(_N_HOURS)])
        else:
            arr = np.asarray(predictions, dtype=float)

        biases = np.array([self._filters[h].bias for h in range(_N_HOURS)])
        return arr + biases

    # ── Hata matrisi ──────────────────────────────────────────────────────────

    def error_matrix(self) -> pd.DataFrame:
        """
        Tüm güncelleme geçmişinin saatlik hata matrisini döndür.

        Sütunlar: hour, mean_error, std_error, bias, uncertainty, P,
                  n_updates, n_clipped, clip_rate
        """
        rows = []
        for h in range(_N_HOURS):
            kf = self._filters[h]
            h_hist = [r for r in self._history if r["hour"] == h]
            if h_hist:
                residuals = [r["raw_residual"] for r in h_hist]
                mean_err = float(np.mean(residuals))
                std_err = float(np.std(residuals))
            else:
                mean_err = std_err = 0.0
            rows.append({
                "hour": h,
                "mean_error_tl": round(mean_err, 2),
                "std_error_tl": round(std_err, 2),
                "current_bias_tl": round(kf.bias, 2),
                "uncertainty_tl": round(kf.uncertainty, 2),
                "P_covariance": round(kf.P_scalar, 4),
                "n_updates": kf.n_updates,
                "n_clipped": kf.n_clipped,
                "clip_rate": round(kf.n_clipped / max(kf.n_updates, 1), 3),
            })
        return pd.DataFrame(rows)

    def state_covariance_matrix(self) -> np.ndarray:
        """
        24×24 durum kovaryans matrisini döndür.

        Köşegen: her saatin P değeri (bias belirsizliği)
        Köşegen dışı: çapraz-saat kovaryans (P_cross'tan)
        """
        P = np.diag([self._filters[h].P_scalar for h in range(_N_HOURS)])
        P += self._P_cross * 0.01   # off-diagonal ölçekleme
        return P

    # ── Kaydet / Yükle ────────────────────────────────────────────────────────

    def save(self, path: Path | str | None = None) -> None:
        """Modül durumunu JSON dosyasına kaydet."""
        path = Path(path or PROJECT_ROOT / "data" / "live" / "rt_correction_state.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "saved_at": datetime.now().isoformat(),
            "filters": {str(h): kf.to_dict() for h, kf in self._filters.items()},
            "P_cross": self._P_cross.tolist(),
        }
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        log.info(f"Durum kaydedildi: {path}")

    def load(self, path: Path | str | None = None) -> "RTCorrectionModule":
        """Önceden kaydedilmiş modül durumunu yükle."""
        path = Path(path or PROJECT_ROOT / "data" / "live" / "rt_correction_state.json")
        if not path.exists():
            log.info(f"Durum dosyası bulunamadı ({path}), sıfırdan başlıyor.")
            return self
        state = json.loads(path.read_text())
        for h_str, kf_dict in state.get("filters", {}).items():
            h = int(h_str)
            if h in self._filters:
                self._filters[h].from_dict(kf_dict)
        if "P_cross" in state:
            self._P_cross = np.array(state["P_cross"])
        log.info(f"Durum yüklendi: {path} ({state.get('saved_at', '?')})")
        return self

    def _flush_log(self) -> None:
        """Hata geçmişini CSV'ye yaz."""
        if not self._history:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self._history)
        write_header = not self.log_path.exists()
        df.to_csv(self.log_path, mode="a", header=write_header, index=False)
        self._history.clear()   # bellek temizliği

    def summary(self, top_n: int = 5) -> str:
        """Saatlik bias özetini yazdır."""
        em = self.error_matrix()
        em_sorted = em.reindex(em["current_bias_tl"].abs().sort_values(ascending=False).index)
        lines = [
            f"RTCorrectionModule — {em['n_updates'].sum()} toplam güncelleme",
            f"  {'Saat':>4} {'Bias':>8} {'±':>6} {'P':>8} {'N':>4} {'Kırp%':>6}",
            "  " + "-" * 44,
        ]
        for _, row in em_sorted.head(top_n).iterrows():
            lines.append(
                f"  {int(row['hour']):>4}h "
                f"{row['current_bias_tl']:>+7.2f} "
                f"{row['uncertainty_tl']:>6.2f} "
                f"{row['P_covariance']:>8.4f} "
                f"{int(row['n_updates']):>4} "
                f"{row['clip_rate']:>5.1%}"
            )
        lines.append(f"\n  State covariance trace: {self.state_covariance_matrix().trace():.2f}")
        return "\n".join(lines)


# ─── Warm-start yardımcısı ────────────────────────────────────────────────────

def warmstart_from_curve_features(
    module: RTCorrectionModule,
    parquet_paths: list[Path] | None = None,
    predicted_col: str = "reconstructed_clearing_price",
    actual_col: str = "mcpPrice",
) -> RTCorrectionModule:
    """
    Mevcut yeniden yapılandırılmış eğri parquetlerinden Kalman filtrelerini ısıt.

    Bu fonksiyon, sistemi canlıya almadan önce geçmiş LP hata vektörünü
    tüm saatlik filtrelere besler. Sonuç: filtreler doğru bias değerine
    yakın başlar, ilk birkaç günde salınım olmaz.

    Args:
        module       : Isıtılacak RTCorrectionModule örneği
        parquet_paths: Eğri özellik parquetleri (None → otomatik bul)

    Returns:
        module (yerinde güncellendi, kolaylık için döndürülür)
    """
    import glob
    if parquet_paths is None:
        parquet_paths = [
            Path(p) for p in sorted(
                glob.glob(str(PROJECT_ROOT / "data" / "features" / "reconstructed_*_curve_features_*.parquet"))
            )
        ]

    all_frames = []
    for p in parquet_paths:
        try:
            df = pd.read_parquet(p)
            all_frames.append(df)
        except Exception as exc:
            log.warning(f"Parquet yüklenemedi {p}: {exc}")

    if not all_frames:
        log.warning("Warm-start için parquet bulunamadı.")
        return module

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["delivery_hour"]).sort_values("delivery_hour")
    combined["hour"] = combined["delivery_hour"].str[11:13].astype(int)

    n_updates = 0
    for _, row in combined.iterrows():
        h = int(row["hour"])
        a = float(row.get(actual_col, float("nan")))
        p_val = float(row.get(predicted_col, float("nan")))
        if np.isnan(a) or np.isnan(p_val):
            continue
        module._filters[h].update(a, p_val)
        n_updates += 1

    log.info(f"Warm-start tamamlandı: {n_updates} saat × {len(parquet_paths)} parquet")
    return module


# ─── Komut satırı çalıştırma ─────────────────────────────────────────────────

def _run_daily_update(
    actual_csv: str,
    predicted_csv: str,
    state_path: str | None = None,
) -> None:
    """
    Komut satırından günlük güncelleme:
        python realtime_correction.py actual.csv predicted.csv [state.json]

    CSV formatı: hour, ptf
    """
    actual_df = pd.read_csv(actual_csv)
    pred_df   = pd.read_csv(predicted_csv)
    actual    = dict(zip(actual_df["hour"], actual_df["ptf"]))
    predicted = dict(zip(pred_df["hour"], pred_df["ptf"]))

    module = RTCorrectionModule()
    module.load(state_path)
    module.update_day(actual, predicted, delivery_date=str(date.today()))
    print(module.summary())
    module.save(state_path)

    print("\nErtesi gün bias düzeltmeleri:")
    corr = module.corrections_for_tomorrow()
    for h, b in sorted(corr.items()):
        if abs(b) > 0.1:
            print(f"  Saat {h:02d}: {b:+.2f} TL/MWh")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        _run_daily_update(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        print("Kullanım: python realtime_correction.py actual.csv predicted.csv [state.json]")

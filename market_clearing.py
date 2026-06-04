#!/usr/bin/env python3
"""
GÖP saatlik piyasa takas simülasyonu — Pyomo + CBC/Gurobi.

Mimari:
    1. MarketClearingModel   : Pyomo LP/MILP ile arz-talep kesişimi
    2. KalmanPriceCorrector  : filterpy Kalman Filtresi ile dinamik bias düzeltmesi

Çözücü seçimi:
    model = MarketClearingModel(solver_name="cbc")      # açık kaynak (python-mip/cbc)
    model = MarketClearingModel(solver_name="gurobi")   # ticari lisans gerektirir

Kalman geri besleme döngüsü:
    1. Saat 12:00 — model 24 saatlik PTF tahmini üretir
    2. Saat 14:30 — EPİAŞ GÖP sonuçlarını açıklar
    3. corrector.update(actual, predicted) çağrılır
    4. Ertesi gün tahminlerine corrector.correct(preds) uygulanır
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


# ─── Veri yapıları ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CurveStep:
    """Arz veya talep eğrisinin tek bir fiyat-hacim adımı."""
    price: float       # TL/MWh
    volume_mwh: float  # MWh


@dataclass
class HourlyBid:
    """
    Tek bir saat için arz ve talep teklifleri.

    supply : fiyata göre artan sıralı (merit order)
    demand : fiyata göre azalan sıralı
    """
    hour: int
    supply: list[CurveStep]
    demand: list[CurveStep]
    price_floor: float = 0.0
    price_ceiling: float = 4500.0       # Nisan 2026 EPDK kararı


@dataclass(frozen=True)
class ClearingResult:
    hour: int
    clearing_price: float    # TL/MWh
    clearing_volume: float   # MWh
    social_welfare: float    # TL — amaç fonksiyonu değeri
    status: str              # "optimal" | "infeasible" | hata mesajı


# ─── Tek saat Pyomo LP modeli ─────────────────────────────────────────────────

def _clear_single_hour(bid: HourlyBid, solver_name: str) -> ClearingResult:
    """
    Bir saat için sosyal refahı maksimize eden LP çöz.

    Model:
        max  Σ_j demand[j].price * xd[j]  -  Σ_i supply[i].price * xs[i]
        s.t. Σ_i xs[i] = Σ_j xd[j]           (denge)
             0 ≤ xs[i] ≤ supply[i].volume_mwh
             0 ≤ xd[j] ≤ demand[j].volume_mwh

    Takas fiyatı: kabul edilen en pahalı arz adımının fiyatı.
    """
    try:
        from pyomo.environ import (
            ConcreteModel,
            Constraint,
            NonNegativeReals,
            Objective,
            RangeSet,
            SolverFactory,
            Var,
            maximize,
            value,
        )
    except ImportError as exc:
        raise ImportError("pyomo kurulu değil — pip install 'pyomo>=6.0'") from exc

    supply = sorted(bid.supply, key=lambda s: s.price)
    demand = sorted(bid.demand, key=lambda d: d.price, reverse=True)
    n_s, n_d = len(supply), len(demand)

    if n_s == 0 or n_d == 0:
        return ClearingResult(bid.hour, 0.0, 0.0, 0.0, "infeasible:empty_curve")

    m = ConcreteModel()
    m.S = RangeSet(0, n_s - 1)
    m.D = RangeSet(0, n_d - 1)
    m.xs = Var(m.S, within=NonNegativeReals)
    m.xd = Var(m.D, within=NonNegativeReals)

    for i, step in enumerate(supply):
        m.xs[i].setub(step.volume_mwh)
    for j, step in enumerate(demand):
        m.xd[j].setub(step.volume_mwh)

    m.obj = Objective(
        expr=(
            sum(demand[j].price * m.xd[j] for j in range(n_d))
            - sum(supply[i].price * m.xs[i] for i in range(n_s))
        ),
        sense=maximize,
    )
    m.balance = Constraint(
        expr=sum(m.xs[i] for i in range(n_s)) == sum(m.xd[j] for j in range(n_d))
    )

    solver = SolverFactory(solver_name)
    res = solver.solve(m, tee=False)
    status = str(res.solver.termination_condition)

    if "optimal" not in status.lower():
        return ClearingResult(bid.hour, 0.0, 0.0, 0.0, f"infeasible:{status}")

    cleared_vol = float(sum(value(m.xs[i]) for i in range(n_s)))
    welfare = float(value(m.obj))

    # Takas fiyatı: kabul edilen son (en pahalı) arz adımı
    clearing_price = bid.price_floor
    for i in range(n_s - 1, -1, -1):
        if value(m.xs[i]) > 1e-6:
            clearing_price = supply[i].price
            break

    # Tavan/taban sınırları
    clearing_price = float(np.clip(clearing_price, bid.price_floor, bid.price_ceiling))
    return ClearingResult(bid.hour, clearing_price, cleared_vol, welfare, "optimal")


# ─── Günlük 24 saat modeli ────────────────────────────────────────────────────

@dataclass
class MarketClearingModel:
    """
    24 adet saatlik LP problemini sırayla çözerek GÖP takas fiyatları üretir.

    Attributes:
        solver_name: "appsi_highs" (HiGHS, varsayılan), "cbc", veya "gurobi"

    Çözücü kurulumları:
        HiGHS  : pip install highspy          (Python 3.12+, açık kaynak)
        CBC    : pip install python-mip       (Python ≤3.11, açık kaynak)
        Gurobi : pip install gurobipy         (ticari lisans gerektirir)
    """
    solver_name: str = "appsi_highs"

    def clear_day(self, hourly_bids: Sequence[HourlyBid]) -> list[ClearingResult]:
        """24 saatlik teklif listesini çöz, sonuçları döndür."""
        return [_clear_single_hour(bid, self.solver_name) for bid in hourly_bids]

    @staticmethod
    def results_to_frame(results: list[ClearingResult]) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in results])

    @staticmethod
    def from_supply_demand_arrays(
        supply_prices: np.ndarray,
        supply_volumes: np.ndarray,
        demand_prices: np.ndarray,
        demand_volumes: np.ndarray,
        hour: int,
        price_ceiling: float = 4500.0,
    ) -> HourlyBid:
        """
        NumPy dizilerinden HourlyBid oluştur.

        Kullanım:
            bid = MarketClearingModel.from_supply_demand_arrays(
                sp, sv, dp, dv, hour=14, price_ceiling=4500.0
            )
        """
        supply = [CurveStep(p, v) for p, v in zip(supply_prices, supply_volumes)]
        demand = [CurveStep(p, v) for p, v in zip(demand_prices, demand_volumes)]
        return HourlyBid(
            hour=hour,
            supply=supply,
            demand=demand,
            price_ceiling=price_ceiling,
        )


# ─── Kalman Filtresi bias düzelticisi ─────────────────────────────────────────

class KalmanPriceCorrector:
    """
    LP kısıtsız takas ile gerçek PTF arasındaki sistematik sapmayı (bias)
    öğrenen ve bir sonraki tahmin döngüsü için düzeltme sağlayan Kalman filtresi.

    Durum:  x = [bias]      (TL/MWh — blok teklif etkisi kaynaklı sistematik sapma)
    Ölçüm:  z = actual - predicted   (kırpılmış, spike outlier'larını dışlar)

    Tasarım notları:
      • Blok teklifler saatlik LP fiyatını ~3–15 TL kaydırır — bu sistematik
        sapma sabit-hızlı (random-walk) bir süreç olarak modellenir (F=1).
      • Spike saatleri (19:00, 21:00–22:00) LP'yi 50–200 TL underestimate eder.
        Bu spike'lar blok bid bias'ı DEĞİL, regime değişimidir ve ayrı spike
        sınıflandırıcı (train_spike_classifier.py) tarafından ele alınır.
      • max_residual parametresi ile gözlemler kırpılarak filtrenin spike
        outlier'larından aşırı etkilenmesi önlenir.

    Parametreler:
        process_noise   : Bias'ın ne hızlı değiştiği (Q); küçük = yavaş adaptasyon
        measurement_noise : Normal saatlerdeki LP ölçüm gürültüsü (R)
        max_residual    : Gözlem kırpma sınırı (TL); spike'ları filtreden izole eder

    Geri besleme döngüsü (saat 14:30 EPİAŞ GÖP açıklandığında):
        corrector.update(actual_ptf=300.0, predicted_ptf=296.5)
    Ertesi gün tahminine uygulamak için:
        adjusted = corrector.correct(tomorrow_predictions)
    """

    def __init__(
        self,
        process_noise: float = 0.5,
        measurement_noise: float = 4.0,
        max_residual: float = 25.0,
    ) -> None:
        try:
            from filterpy.kalman import KalmanFilter
        except ImportError as exc:
            raise ImportError(
                "filterpy kurulu değil — pip install 'filterpy>=1.4.5'"
            ) from exc

        from filterpy.kalman import KalmanFilter

        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.x = np.array([[0.0]])    # başlangıç bias tahmini: sıfır
        kf.F = np.array([[1.0]])    # bias bir adımdan sonrakine aynen geçer
        kf.H = np.array([[1.0]])    # ölçüm = bias
        kf.P = np.array([[25.0]])   # başlangıç belirsizliği
        kf.Q = np.array([[process_noise]])
        kf.R = np.array([[measurement_noise]])
        self._kf = kf
        self._max_residual = max_residual
        self._updates: int = 0
        self._clipped: int = 0

    @property
    def current_bias(self) -> float:
        """Mevcut filtre tarafından tahmin edilen fiyat sapması (TL/MWh)."""
        return float(self._kf.x[0, 0])

    @property
    def uncertainty(self) -> float:
        """Bias tahmininin standart sapması (TL/MWh)."""
        return float(np.sqrt(self._kf.P[0, 0]))

    @property
    def clip_rate(self) -> float:
        """Kırpılan güncelleme oranı (spike outlier göstergesi)."""
        return self._clipped / max(self._updates, 1)

    def update(self, actual_ptf: float, predicted_ptf: float) -> float:
        """
        GÖP sonucu açıklandığında çağrılır.

        Gözlem max_residual ile kırpılır: spike saatlerindeki büyük residuals
        (blok teklif kaynaklı değil, fiyat rejimi değişimi kaynaklı) filtreyi
        bozmaz.

        Returns:
            Güncellenmiş bias (TL/MWh)
        """
        raw_residual = actual_ptf - predicted_ptf
        residual = float(np.clip(raw_residual, -self._max_residual, self._max_residual))
        self._updates += 1
        if abs(raw_residual) > self._max_residual:
            self._clipped += 1
        self._kf.predict()
        self._kf.update(np.array([[residual]]))
        return self.current_bias

    def update_batch(
        self,
        actual_series: pd.Series,
        predicted_series: pd.Series,
    ) -> pd.Series:
        """
        Tarihsel seri üzerinden filtreyi ısıt (warm-start).

        Returns:
            Her adımda hesaplanan bias serisini döndürür.
        """
        biases = []
        for actual, predicted in zip(actual_series, predicted_series):
            bias = self.update(float(actual), float(predicted))
            biases.append(bias)
        return pd.Series(biases, index=actual_series.index)

    def correct(self, predictions: np.ndarray | pd.Series) -> np.ndarray:
        """
        Mevcut bias ile saatlik tahmin dizisini düzelt.

        predictions : shape (24,) saatlik PTF tahmini
        Returns     : shape (24,) düzeltilmiş tahmin
        """
        arr = np.asarray(predictions, dtype=float)
        return arr + self.current_bias

    def summary(self) -> dict[str, float]:
        return {
            "bias_tl_per_mwh": self.current_bias,
            "uncertainty_tl_per_mwh": self.uncertainty,
        }

#!/usr/bin/env python3
"""
İki Aşamalı PTF Tahmin Modeli.

Aşama 1 — Eğri Tahmini:
    35 fiyat kademesinde kümülatif arz ve talep miktarlarını tahmin eden
    çok çıktılı model. İki arka uç:
      A) MultiOutputLGBMCurvePredictor : küçük veri (≤1000 örnek) için tavsiye
      B) CurveNet (PyTorch)            : daha büyük veri veya GPU mevcutsa

Aşama 2 — Piyasa Takas:
    Tahmin edilen eğriler market_clearing.py Pyomo LP modeline beslenir;
    sosyal refahı maksimize eden saatlik denge fiyatı türetilir.

Fiyat ızgarası (35 kademe):
    0-4500 TL/MWh aralığında logaritmik sıkıştırma ile dağıtılmış 35 nokta.
    Sıfır fiyat bölgesi, orta fiyat ve tavan bölgeleri ayrı ayrı örneklenmiştir.

Notasyon:
    supply_cum[k] = P ≤ PRICE_GRID[k] koşulunu sağlayan kümülatif arz (MWh)
    demand_cum[k] = P ≥ PRICE_GRID[k] koşulunu sağlayan kümülatif talep (MWh)

    Her iki dizi de monotondur:
      supply_cum non-decreasing (fiyat arttıkça arz artar)
      demand_cum non-increasing  (fiyat arttıkça talep düşer)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
log = logging.getLogger(__name__)

# ─── Fiyat ızgarası ───────────────────────────────────────────────────────────

PRICE_GRID: np.ndarray = np.array([
    0, 10, 30,                                    # sıfır fiyat / neredeyse-sıfır
    50, 100, 150, 200, 250, 300, 350, 400,        # düşük fiyat (RES meriti)
    500, 600, 700, 800, 900, 1000,                # orta-düşük
    1200, 1500, 1750, 2000, 2250,                 # orta
    2500, 2750, 3000, 3300, 3600,                 # yüksek
    3900, 4100, 4200, 4300, 4400, 4450, 4500,    # tavan bölgesi (EPDK 2026: 4500)
], dtype=np.float32)

N_LEVELS: int = len(PRICE_GRID)   # 35

# Normalleştirme sabitleri
_VOL_SCALE   = 60_000.0   # MW — yaklaşık Türkiye kurulu kapasite üst sınırı
_PRICE_SCALE = 4_500.0    # TL/MWh — tavan fiyat

# ─── 1. Veri seti oluşturucu ──────────────────────────────────────────────────

class CurveDataset:
    """
    Ham EPİAŞ JSON dosyalarından ve master özellik tablosundan
    35 kademeli arz/talep eğrisi eğitim veri seti üretir.

    Çalışma mantığı:
      1. Her haftalık klasördeki `hour_XX_supply.body.txt` dosyalarını oku.
      2. Her saat için kümülatif arz ve talebi 35 fiyat kademesinde enterpolasyon ile hesapla.
      3. Master özellik tablosu (master_hourly_v1.parquet) ile `ts_hour` üzerinden birleştir.

    Sonuç:
        X         : (N, n_features) — girdi özellik matrisi
        supply_mat: (N, 35)         — kümülatif arz [MWh, normalize]
        demand_mat: (N, 35)         — kümülatif talep [MWh, normalize]
        hours_idx : pd.DatetimeIndex — her satırın teslimat saati
    """

    def __init__(
        self,
        weekly_root: Path | None = None,
        master_path: Path | None = None,
    ) -> None:
        self.weekly_root = weekly_root or PROJECT_ROOT / "data" / "raw" / "dam_supply_demand_curve_weekly"
        self.master_path = master_path or PROJECT_ROOT / "data" / "master" / "master_hourly_v1.parquet"

    @staticmethod
    def curve_vectors_from_json(supply_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Tek bir `hour_XX_supply.body.txt` dosyasından 35 kademeli vektörler üret.

        Döner:
            (supply_cum, demand_cum) her biri shape (35,) normalize float32
            Hata durumunda None
        """
        try:
            items = json.loads(supply_path.read_text(encoding="utf-8"))["items"]
        except Exception:
            return None

        sup_rows = sorted([x for x in items if "supplyPrice" in x], key=lambda r: r["supplyPrice"])
        dem_rows = sorted([x for x in items if "demandPrice" in x], key=lambda r: r["demandPrice"])

        if not sup_rows or not dem_rows:
            return None

        s_prices = np.array([r["supplyPrice"] for r in sup_rows], dtype=np.float32)
        s_vols   = np.array([r["amount"] for r in sup_rows], dtype=np.float32)
        d_prices = np.array([r["demandPrice"] for r in dem_rows], dtype=np.float32)
        d_vols   = np.array([r["amount"] for r in dem_rows], dtype=np.float32)

        # Enterpolasyon → 35 kademe
        supply_cum = np.interp(PRICE_GRID, s_prices, s_vols).astype(np.float32)
        demand_cum = np.interp(PRICE_GRID, d_prices, d_vols).astype(np.float32)

        # Monotonluk garantisi (nümerik hata temizliği)
        supply_cum = np.maximum.accumulate(supply_cum)
        demand_cum = np.minimum.accumulate(demand_cum[::-1])[::-1]

        return supply_cum / _VOL_SCALE, demand_cum / _VOL_SCALE

    def build(
        self,
        feature_cols: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Tüm mevcut haftalık verilerden eğitim matrisini oluştur.

        Returns:
            X          : (N, F) float32 — özellikler
            supply_mat : (N, 35) float32 — normalize kümülatif arz
            demand_mat : (N, 35) float32 — normalize kümülatif talep
            ts_index   : pd.DatetimeIndex uzunluğu N
        """
        # Master özellik tablosu yükle
        master = pd.read_parquet(self.master_path)
        master["ts_hour"] = pd.to_datetime(master["ts_hour"], utc=True).dt.tz_convert("Europe/Istanbul")
        master = master.sort_values("ts_hour").reset_index(drop=True)

        # Varsayılan özellik sütunları
        if feature_cols is None:
            exclude = {"ts_hour", "ptf", "split"}
            feature_cols = [c for c in master.columns if c not in exclude
                            and master[c].dtype in (np.float32, np.float64, np.int32, np.int64, "float64", "int64")]

        master_ts = master.set_index("ts_hour")

        curve_rows: list[dict] = []

        week_dirs = sorted(self.weekly_root.glob("*/"))
        for week_dir in week_dirs:
            day_dirs = sorted(week_dir.glob("*/"))
            for day_dir in day_dirs:
                date_str = day_dir.name  # YYYY-MM-DD
                for hour in range(24):
                    supply_path = day_dir / f"hour_{hour:02d}_supply.body.txt"
                    if not supply_path.exists():
                        continue

                    result = self.curve_vectors_from_json(supply_path)
                    if result is None:
                        continue

                    supply_cum, demand_cum = result
                    ts = pd.Timestamp(f"{date_str}T{hour:02d}:00:00", tz="Europe/Istanbul")
                    curve_rows.append({
                        "ts_hour": ts,
                        "supply_cum": supply_cum,
                        "demand_cum": demand_cum,
                    })

        if not curve_rows:
            raise RuntimeError(f"Eğri verisi bulunamadı: {self.weekly_root}")

        log.info(f"Eğri örnekleri: {len(curve_rows)}")

        # Feature eşleştirme
        X_list, S_list, D_list, ts_list = [], [], [], []
        for row in curve_rows:
            ts = row["ts_hour"]
            # En yakın master satırını bul (±1 saat tolerans)
            try:
                feat_row = master_ts.loc[ts]
            except KeyError:
                # Toleranslı arama (±1 saat)
                deltas_s = (master_ts.index - ts).total_seconds()
                abs_deltas = np.abs(deltas_s)
                idx_min = int(np.argmin(abs_deltas))
                if abs_deltas[idx_min] > 3600:
                    continue
                feat_row = master_ts.iloc[idx_min]

            feat_vals = pd.to_numeric(feat_row[feature_cols], errors="coerce").fillna(0.0).values.astype(np.float32)
            X_list.append(feat_vals)
            S_list.append(row["supply_cum"])
            D_list.append(row["demand_cum"])
            ts_list.append(ts)

        X = np.stack(X_list).astype(np.float32)
        supply_mat = np.stack(S_list).astype(np.float32)
        demand_mat = np.stack(D_list).astype(np.float32)
        ts_index = pd.DatetimeIndex(ts_list)

        log.info(f"Eğitim seti: X={X.shape}, supply={supply_mat.shape}, demand={demand_mat.shape}")
        return X, supply_mat, demand_mat, ts_index

    @property
    def feature_cols(self) -> list[str]:
        """Varsayılan özellik listesini döndür (build() ile aynı mantık)."""
        master = pd.read_parquet(self.master_path)
        exclude = {"ts_hour", "ptf", "split"}
        return [c for c in master.columns if c not in exclude
                and master[c].dtype in ("float64", "int64")]


# ─── 2A. Çok çıktılı LightGBM eğri tahmin modeli ────────────────────────────

class MultiOutputLGBMCurvePredictor:
    """
    35 × 2 = 70 ayrı LightGBM regresyonu ile arz ve talep eğrilerini tahmin eder.

    sklearn MultiOutputRegressor kullanılır; her çıktı (fiyat kademesi) bağımsız.
    Monotonluk tahminden sonra zorunlu kılınır (cummax / cummin).

    Küçük veri (≤1000 örnek) için LightGBM'in ağaç yapısı PyTorch'tan
    daha kararlı sonuç verir. Tavsiye edilen arka uçtur.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        num_leaves: int = 31,
        min_child_samples: int = 10,
        learning_rate: float = 0.05,
        n_jobs: int = -1,
    ) -> None:
        self.params = dict(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            learning_rate=learning_rate,
            n_jobs=n_jobs,
            verbose=-1,
        )
        self.supply_model: Any = None
        self.demand_model: Any = None
        self.feature_cols: list[str] = []
        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        supply_mat: np.ndarray,
        demand_mat: np.ndarray,
        feature_cols: list[str] | None = None,
    ) -> "MultiOutputLGBMCurvePredictor":
        """
        Args:
            X          : (N, F) özellik matrisi
            supply_mat : (N, 35) normalize kümülatif arz
            demand_mat : (N, 35) normalize kümülatif talep
        """
        try:
            from lightgbm import LGBMRegressor
            from sklearn.multioutput import MultiOutputRegressor
        except ImportError as exc:
            raise ImportError("pip install lightgbm scikit-learn") from exc

        if feature_cols:
            self.feature_cols = feature_cols

        log.info(f"LightGBM Stage 1 eğitimi: X={X.shape}, supply={supply_mat.shape}")

        base_s = LGBMRegressor(**self.params)
        base_d = LGBMRegressor(**self.params)
        self.supply_model = MultiOutputRegressor(base_s, n_jobs=1)
        self.demand_model = MultiOutputRegressor(base_d, n_jobs=1)

        self.supply_model.fit(X, supply_mat)
        self.demand_model.fit(X, demand_mat)
        self._is_fitted = True
        log.info("Eğitim tamamlandı.")
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            supply_pred: (N, 35) normalize kümülatif arz
            demand_pred: (N, 35) normalize kümülatif talep
            (her ikisi de monotonluk zorunlu kılınmış)
        """
        if not self._is_fitted:
            raise RuntimeError("Model henüz eğitilmedi. Önce fit() çağırın.")

        sup = self.supply_model.predict(X).astype(np.float32)
        dem = self.demand_model.predict(X).astype(np.float32)

        # Monotonluk
        sup = np.maximum.accumulate(sup.clip(0, 1), axis=1)
        dem_rev = np.minimum.accumulate(dem[:, ::-1].clip(0, 1), axis=1)[:, ::-1]

        return sup, dem_rev

    def predict_as_bids(
        self,
        X: np.ndarray,
        hour_indices: list[int],
        price_ceiling: float = 4_500.0,
    ) -> "list[HourlyBid]":
        """
        Tahmin edilen eğrileri market_clearing.py HourlyBid nesnelerine dönüştür.

        Kümülatif → Marjinal dönüşümü burada gerçekleşir.
        """
        from market_clearing import CurveStep, HourlyBid

        sup, dem = self.predict(X)
        bids = []
        for idx, hour in enumerate(hour_indices):
            s_cum = sup[idx] * _VOL_SCALE
            d_cum = dem[idx] * _VOL_SCALE

            # Kümülatif → marjinal
            supply_steps = []
            prev = 0.0
            for k, price in enumerate(PRICE_GRID):
                inc = float(s_cum[k]) - prev
                if inc > 1.0:
                    supply_steps.append(CurveStep(price=float(price), volume_mwh=inc))
                prev = float(s_cum[k])

            demand_steps = []
            prev = 0.0
            for k in range(len(PRICE_GRID) - 1, -1, -1):
                inc = float(d_cum[k]) - prev
                if inc > 1.0:
                    demand_steps.append(CurveStep(price=float(PRICE_GRID[k]), volume_mwh=inc))
                prev = float(d_cum[k])

            bids.append(HourlyBid(
                hour=hour,
                supply=supply_steps,
                demand=demand_steps,
                price_ceiling=price_ceiling,
            ))
        return bids

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.supply_model, path / "supply_model.pkl")
        joblib.dump(self.demand_model, path / "demand_model.pkl")
        joblib.dump({"feature_cols": self.feature_cols, "params": self.params}, path / "meta.pkl")

    @classmethod
    def load(cls, path: Path | str) -> "MultiOutputLGBMCurvePredictor":
        path = Path(path)
        meta = joblib.load(path / "meta.pkl")
        obj = cls(**meta["params"])
        obj.supply_model = joblib.load(path / "supply_model.pkl")
        obj.demand_model = joblib.load(path / "demand_model.pkl")
        obj.feature_cols = meta.get("feature_cols", [])
        obj._is_fitted = True
        return obj


# ─── 2B. PyTorch eğri tahmin modeli ──────────────────────────────────────────

class CurveNet:
    """
    35 kademeli arz ve talep eğrilerini tahmin eden tam bağlantılı sinir ağı.

    Monotonluk: çıktı katmanında kümülatif toplam (cumsum) kullanılır.
      supply_head → softplus → cumsum → normalize [0, 1]
      demand_head → softplus → cumsum → ters yönde → normalize [0, 1]

    Daha büyük veri (≥2000 örnek) veya GPU mevcutsa LightGBM yerine kullanın.
    Küçük veri için MultiOutputLGBMCurvePredictor daha kararlıdır.
    """

    def __init__(
        self,
        n_features: int = 89,
        hidden_dims: tuple[int, ...] = (256, 256, 128),
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 150,
        batch_size: int = 64,
        device: str = "auto",
    ) -> None:
        self.n_features = n_features
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device_str = device
        self._net: Any = None
        self.feature_cols: list[str] = []
        self._is_fitted = False
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None

    def _get_device(self):
        import torch
        if self.device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_str)

    def _build_net(self, n_features: int):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, n_in, hidden, drop, n_levels):
                super().__init__()
                layers = []
                prev = n_in
                for h in hidden:
                    layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(drop)]
                    prev = h
                self.encoder = nn.Sequential(*layers)
                self.supply_head = nn.Linear(prev, n_levels)
                self.demand_head = nn.Linear(prev, n_levels)
                self.softplus = nn.Softplus()

            def forward(self, x):
                h = self.encoder(x)
                # supply: monotone non-decreasing (cumsum of softplus outputs)
                s_inc = self.softplus(self.supply_head(h)) + 1e-6
                s_cum = s_inc.cumsum(dim=-1)
                s_cum = s_cum / (s_cum[:, -1:] + 1e-6)   # normalize [0, 1]

                # demand: monotone non-increasing (reversed cumsum)
                d_inc = self.softplus(self.demand_head(h)) + 1e-6
                d_cum_rev = d_inc.flip(-1).cumsum(dim=-1).flip(-1)
                d_cum = d_cum_rev / (d_cum_rev[:, 0:1] + 1e-6)

                return s_cum, d_cum

        return _Net(n_features, self.hidden_dims, self.dropout, N_LEVELS)

    def fit(
        self,
        X: np.ndarray,
        supply_mat: np.ndarray,
        demand_mat: np.ndarray,
        feature_cols: list[str] | None = None,
        val_split: float = 0.15,
    ) -> "CurveNet":
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise ImportError("pip install torch") from exc

        if feature_cols:
            self.feature_cols = feature_cols

        # Özellik standardizasyonu
        self._feature_mean = X.mean(axis=0)
        self._feature_std = X.std(axis=0) + 1e-8
        X_norm = ((X - self._feature_mean) / self._feature_std).astype(np.float32)

        device = self._get_device()
        n = len(X_norm)
        n_val = max(1, int(n * val_split))
        idx = np.random.permutation(n)
        tr_idx, val_idx = idx[n_val:], idx[:n_val]

        def to_tensor(arr): return torch.from_numpy(arr).to(device)

        ds_tr = TensorDataset(to_tensor(X_norm[tr_idx]), to_tensor(supply_mat[tr_idx]), to_tensor(demand_mat[tr_idx]))
        ds_val = TensorDataset(to_tensor(X_norm[val_idx]), to_tensor(supply_mat[val_idx]), to_tensor(demand_mat[val_idx]))
        dl_tr = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=len(ds_val))

        self.n_features = X_norm.shape[1]
        self._net = self._build_net(self.n_features).to(device)
        opt = torch.optim.AdamW(self._net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        loss_fn = nn.MSELoss()

        best_val, best_state = float("inf"), None
        log.info(f"PyTorch Stage 1 eğitimi: {len(ds_tr)} eğitim, {len(ds_val)} doğrulama")

        for epoch in range(1, self.epochs + 1):
            self._net.train()
            for xb, sb, db in dl_tr:
                sp, dm = self._net(xb)
                loss = loss_fn(sp, sb) + loss_fn(dm, db)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                opt.step()
            sched.step()

            if epoch % 10 == 0 or epoch == self.epochs:
                self._net.eval()
                with torch.no_grad():
                    xv, sv, dv = next(iter(dl_val))
                    sp_v, dm_v = self._net(xv)
                    val_loss = (loss_fn(sp_v, sv) + loss_fn(dm_v, dv)).item()
                if val_loss < best_val:
                    best_val = val_loss
                    import copy; best_state = copy.deepcopy(self._net.state_dict())
                log.info(f"  epoch {epoch:3d}/{self.epochs} — val_loss={val_loss:.6f}")

        if best_state:
            self._net.load_state_dict(best_state)
        self._is_fitted = True
        log.info(f"En iyi val_loss={best_val:.6f}")
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self._is_fitted:
            raise RuntimeError("Model henüz eğitilmedi.")
        import torch
        device = self._get_device()
        X_norm = ((X - self._feature_mean) / self._feature_std).astype(np.float32)
        with torch.no_grad():
            xb = torch.from_numpy(X_norm).to(device)
            sp, dm = self._net(xb)
        return sp.cpu().numpy(), dm.cpu().numpy()

    def predict_as_bids(
        self,
        X: np.ndarray,
        hour_indices: list[int],
        price_ceiling: float = 4_500.0,
    ) -> "list[HourlyBid]":
        from market_clearing import CurveStep, HourlyBid
        sup, dem = self.predict(X)
        bids = []
        for idx, hour in enumerate(hour_indices):
            s_cum = sup[idx] * _VOL_SCALE
            d_cum = dem[idx] * _VOL_SCALE
            supply_steps, demand_steps = [], []
            prev = 0.0
            for k, price in enumerate(PRICE_GRID):
                inc = float(s_cum[k]) - prev
                if inc > 1.0:
                    supply_steps.append(CurveStep(float(price), inc))
                prev = float(s_cum[k])
            prev = 0.0
            for k in range(len(PRICE_GRID) - 1, -1, -1):
                inc = float(d_cum[k]) - prev
                if inc > 1.0:
                    demand_steps.append(CurveStep(float(PRICE_GRID[k]), inc))
                prev = float(d_cum[k])
            bids.append(HourlyBid(hour, supply_steps, demand_steps, price_ceiling=price_ceiling))
        return bids

    def save(self, path: Path | str) -> None:
        import torch
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(), path / "net.pt")
        joblib.dump({
            "n_features": self.n_features,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "feature_cols": self.feature_cols,
            "feature_mean": self._feature_mean,
            "feature_std": self._feature_std,
        }, path / "meta.pkl")

    @classmethod
    def load(cls, path: Path | str) -> "CurveNet":
        import torch
        path = Path(path)
        meta = joblib.load(path / "meta.pkl")
        obj = cls(n_features=meta["n_features"], hidden_dims=meta["hidden_dims"], dropout=meta["dropout"])
        obj.feature_cols = meta.get("feature_cols", [])
        obj._feature_mean = meta["feature_mean"]
        obj._feature_std = meta["feature_std"]
        obj._net = obj._build_net(meta["n_features"])
        obj._net.load_state_dict(torch.load(path / "net.pt", map_location="cpu"))
        obj._net.eval()
        obj._is_fitted = True
        return obj


# ─── 3. İki Aşamalı PTF Tahmin Sistemi ───────────────────────────────────────

class TwoStagePTFForecaster:
    """
    Aşama 1 (eğri tahmini) + Aşama 2 (Pyomo LP takas) = saatlik PTF tahmini.

    Kullanım:
        forecaster = TwoStagePTFForecaster(backend="lgbm")
        forecaster.fit_from_files()                       # veri yükle + eğit
        prices = forecaster.predict_day("2026-06-05")     # 24 saatlik tahmin

    Kalman düzeltmesi:
        corrector = KalmanPriceCorrector(...)
        forecaster.kalman = corrector                     # atama yeterli
        prices = forecaster.predict_day("2026-06-05")     # otomatik düzeltilir
    """

    def __init__(
        self,
        backend: str = "lgbm",   # "lgbm" | "pytorch"
        solver_name: str = "appsi_highs",
        kalman: Any = None,
    ) -> None:
        if backend not in ("lgbm", "pytorch"):
            raise ValueError(f"backend '{backend}' geçersiz. 'lgbm' veya 'pytorch' kullanın.")
        self.backend = backend
        self.solver_name = solver_name
        self.kalman = kalman
        self._model: MultiOutputLGBMCurvePredictor | CurveNet | None = None
        self._dataset: CurveDataset | None = None
        self._feature_cols: list[str] = []
        self._master: pd.DataFrame | None = None

    def fit_from_files(
        self,
        weekly_root: Path | None = None,
        master_path: Path | None = None,
        save_to: Path | None = None,
    ) -> "TwoStagePTFForecaster":
        """Disk üzerindeki veriden veri seti oluştur ve Stage 1 modelini eğit."""
        ds = CurveDataset(weekly_root, master_path)
        X, supply_mat, demand_mat, _ = ds.build()
        self._feature_cols = ds.feature_cols
        self._dataset = ds

        if self.backend == "lgbm":
            self._model = MultiOutputLGBMCurvePredictor()
        else:
            self._model = CurveNet(n_features=X.shape[1])

        self._model.fit(X, supply_mat, demand_mat, self._feature_cols)

        if save_to:
            self._model.save(save_to)
        return self

    def load_model(self, path: Path | str) -> "TwoStagePTFForecaster":
        path = Path(path)
        if self.backend == "lgbm":
            self._model = MultiOutputLGBMCurvePredictor.load(path)
        else:
            self._model = CurveNet.load(path)
        self._feature_cols = self._model.feature_cols
        return self

    def _get_master(self) -> pd.DataFrame:
        if self._master is None:
            p = PROJECT_ROOT / "data" / "master" / "master_hourly_v1.parquet"
            self._master = pd.read_parquet(p)
            ts = pd.to_datetime(self._master["ts_hour"])
            if ts.dt.tz is None:
                ts = ts.dt.tz_localize("Europe/Istanbul", ambiguous="NaT", nonexistent="NaT")
            else:
                ts = ts.dt.tz_convert("Europe/Istanbul")
            self._master["ts_hour"] = ts
        return self._master

    def predict_day(
        self,
        for_date: str,
        price_ceiling: float = 4_500.0,
    ) -> pd.DataFrame:
        """
        Teslimat gününün 24 saatlik PTF tahminini üret.

        Akış:
          1. Master tablosundan for_date için 24 saatlik özellik vektörü al
          2. Stage 1 ile 24 × 35 kademeli arz/talep matrisi tahmin et
          3. HourlyBid dönüştür → Pyomo LP (MarketClearingModel) ile 24 fiyat bul
          4. Kalman varsa bias düzeltmesi uygula

        Returns:
            DataFrame: hour, predicted_ptf, clearing_volume, status
        """
        from market_clearing import MarketClearingModel

        if self._model is None:
            raise RuntimeError("Model yüklenmemiş. Önce fit_from_files() veya load_model() çağırın.")

        master = self._get_master()
        date_ts = pd.Timestamp(for_date, tz="Europe/Istanbul")
        day_mask = master["ts_hour"].dt.date == date_ts.date()
        day_data = master[day_mask].sort_values("ts_hour").reset_index(drop=True)

        if day_data.empty:
            # En yakın geçmiş günün verisiyle devam et
            available = master["ts_hour"].dt.date.unique()
            past = [d for d in available if d < date_ts.date()]
            if not past:
                raise ValueError(f"{for_date} için master tablosunda veri yok.")
            nearest = max(past)
            log.warning(f"{for_date} master'da yok — en yakın gün kullanılıyor: {nearest}")
            day_data = master[master["ts_hour"].dt.date == nearest].sort_values("ts_hour").reset_index(drop=True)

        cols = [c for c in self._feature_cols if c in day_data.columns]
        X_day = day_data[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
        hours = day_data["ts_hour"].dt.hour.tolist()

        bids = self._model.predict_as_bids(X_day, hours, price_ceiling=price_ceiling)

        # Stage 2: Pyomo LP
        model = MarketClearingModel(solver_name=self.solver_name)
        results = model.clear_day(bids)

        # Kalman düzeltmelerini önceden al (saat → bias sözlüğü)
        kalman_corrections: dict[int, float] = {}
        if self.kalman is not None:
            kalman_corrections = self.kalman.corrections_for_tomorrow()

        rows = []
        for r in results:
            ptf = r.clearing_price
            if kalman_corrections:
                ptf += kalman_corrections.get(r.hour, 0.0)
            rows.append({
                "hour": r.hour,
                "predicted_ptf": round(ptf, 2),
                "clearing_volume": round(r.clearing_volume, 1),
                "lp_status": r.status,
            })

        return pd.DataFrame(rows).sort_values("hour").reset_index(drop=True)

    def evaluate(
        self,
        start_date: str,
        end_date: str,
        actual_ptf_col: str = "ptf",
    ) -> pd.DataFrame:
        """
        Belirli tarih aralığında tahmin vs. gerçek karşılaştırması.
        Master tablosundaki actual_ptf_col sütunu kullanılır.
        """
        master = self._get_master()
        mask = (master["ts_hour"].dt.date >= pd.Timestamp(start_date).date()) & \
               (master["ts_hour"].dt.date <= pd.Timestamp(end_date).date())
        subset = master[mask].copy()

        all_rows = []
        for date_str in subset["ts_hour"].dt.date.unique():
            try:
                pred_df = self.predict_day(str(date_str))
                pred_df["date"] = str(date_str)
                actual = subset[subset["ts_hour"].dt.date == date_str][[actual_ptf_col, "ts_hour"]].copy()
                actual["hour"] = actual["ts_hour"].dt.hour
                pred_df = pred_df.merge(actual[["hour", actual_ptf_col]], on="hour", how="left")
                pred_df = pred_df.rename(columns={actual_ptf_col: "actual_ptf"})
                pred_df["error"] = pred_df["predicted_ptf"] - pred_df["actual_ptf"]
                pred_df["abs_error"] = pred_df["error"].abs()
                all_rows.append(pred_df)
            except Exception as exc:
                log.warning(f"{date_str} tahmin hatası: {exc}")

        if not all_rows:
            return pd.DataFrame()

        result = pd.concat(all_rows, ignore_index=True)
        mae = result["abs_error"].mean()
        mape = (result["abs_error"] / result["actual_ptf"].clip(lower=1)).mean() * 100
        log.info(f"Değerlendirme {start_date}→{end_date}: MAE={mae:.2f} TL, MAPE={mape:.2f}%")
        return result

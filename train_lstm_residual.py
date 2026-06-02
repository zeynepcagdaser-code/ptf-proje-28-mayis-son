#!/usr/bin/env python3
"""Train residual LSTM: predict persistence error, combine for final PTF forecast."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "data" / "model_residual"
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "lstm_residual_next24_v1.parquet"
OUTPUT_MODEL = PROJECT_ROOT / "models" / "lstm_residual.pt"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
METRICS_JSON = PROJECT_ROOT / "reports" / "lstm_residual_metrics.json"
METRICS_MD = PROJECT_ROOT / "reports" / "lstm_residual_metrics.md"
PERSISTENCE_METRICS = PROJECT_ROOT / "reports" / "persistence_metrics.json"
DIRECT_LSTM_METRICS = PROJECT_ROOT / "reports" / "lstm_baseline_metrics.json"

SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 80
PATIENCE = 10
LEARNING_RATE = 1e-3
HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 64
DROPOUT = 0.2
OUTPUT_HORIZON = 24
MAPE_MASK_THRESHOLD = 100.0

ARRAY_PATHS = {
    "X_train": MODEL_DIR / "X_train.npy",
    "y_train": MODEL_DIR / "y_train.npy",
    "X_val": MODEL_DIR / "X_val.npy",
    "y_val": MODEL_DIR / "y_val.npy",
    "X_test": MODEL_DIR / "X_test.npy",
    "y_test": MODEL_DIR / "y_test.npy",
}


@dataclass
class TrainConfig:
    batch_size: int = BATCH_SIZE
    max_epochs: int = MAX_EPOCHS
    patience: int = PATIENCE
    learning_rate: float = LEARNING_RATE
    smoke_test: bool = False


class MmapSequenceDataset(Dataset):
    def __init__(self, x_path: Path, y_path: Path):
        self._x = np.load(x_path, mmap_mode="r")
        self._y = np.load(y_path, mmap_mode="r")
        if len(self._x) != len(self._y):
            raise ValueError(
                f"X/y length mismatch: {len(self._x)} vs {len(self._y)} "
                f"({x_path.name}, {y_path.name})"
            )

    @property
    def feature_count(self) -> int:
        return int(self._x.shape[2])

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.array(self._x[idx], dtype=np.float32))
        y = torch.from_numpy(np.array(self._y[idx], dtype=np.float32))
        return x, y


class StackedLSTM(nn.Module):
    def __init__(self, input_size: int, output_size: int = OUTPUT_HORIZON):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=HIDDEN_SIZE_1, batch_first=True)
        self.dropout1 = nn.Dropout(DROPOUT)
        self.lstm2 = nn.LSTM(input_size=HIDDEN_SIZE_1, hidden_size=HIDDEN_SIZE_2, batch_first=True)
        self.dropout2 = nn.Dropout(DROPOUT)
        self.head = nn.Linear(HIDDEN_SIZE_2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1, _ = self.lstm1(x)
        out1 = self.dropout1(out1)
        out2, _ = self.lstm2(out1)
        last = self.dropout2(out2[:, -1, :])
        return self.head(last)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def verify_array_paths() -> None:
    for name, path in ARRAY_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {name} at {path}. Run: python run_sequence_residual.py"
            )


def make_loader(
    x_path: Path,
    y_path: Path,
    *,
    batch_size: int,
    device: torch.device,
    shuffle: bool = False,
) -> DataLoader:
    dataset = MmapSequenceDataset(x_path, y_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    n_batches = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=(device.type == "cuda"))
        yb = yb.to(device, non_blocking=(device.type == "cuda"))

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        preds = model(xb)
        loss = criterion(preds, yb)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mape(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    mask = np.abs(a) > eps
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((a[mask] - b[mask]) / a[mask])) * 100)


def masked_mape(actual: np.ndarray, pred: np.ndarray, threshold: float = MAPE_MASK_THRESHOLD) -> float:
    mask = np.abs(actual) > threshold
    if not mask.any():
        return float("nan")
    return mape(actual[mask], pred[mask])


def horizon_mae(actual: np.ndarray, pred: np.ndarray) -> dict[int, float]:
    return {h + 1: mae(actual[:, h], pred[:, h]) for h in range(actual.shape[1])}


def subset_mae(actual: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float | None:
    if not mask.any():
        return None
    return mae(actual[mask].ravel(), pred[mask].ravel())


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    for xb, _ in loader:
        preds = model(xb.to(device, non_blocking=(device.type == "cuda"))).cpu().numpy()
        chunks.append(preds)
    return np.concatenate(chunks, axis=0)


def load_price_and_persistence(anchor_path: Path) -> tuple[np.ndarray, np.ndarray]:
    price_cols = [f"target_{h}h" for h in range(1, OUTPUT_HORIZON + 1)]
    persistence_cols = [f"persistence_{h}h" for h in range(1, OUTPUT_HORIZON + 1)]

    from src.utils.io_utils import read_parquet_with_normalized_ts
    features = read_parquet_with_normalized_ts(FEATURES_PATH)
    features["ts_hour"] = pd.to_datetime(features["ts_hour"], utc=True).dt.tz_convert(
        "Europe/Istanbul"
    )

    anchors = pd.read_csv(anchor_path)
    anchors["anchor_ts_hour"] = pd.to_datetime(
        anchors["anchor_ts_hour"], utc=True
    ).dt.tz_convert("Europe/Istanbul")

    merged = anchors.merge(
        features,
        left_on="anchor_ts_hour",
        right_on="ts_hour",
        how="inner",
    )
    if len(merged) != len(anchors):
        raise ValueError(
            f"Anchor merge mismatch: {len(merged)} merged vs {len(anchors)} anchors"
        )

    return merged[price_cols].to_numpy(dtype=np.float64), merged[persistence_cols].to_numpy(
        dtype=np.float64
    )


def build_predictions_csv(
    anchor_path: Path,
    actual: np.ndarray,
    persistence: np.ndarray,
    residual_pred: np.ndarray,
    final_pred: np.ndarray,
) -> pd.DataFrame:
    anchors = pd.read_csv(anchor_path)
    rows = []
    for i in range(len(anchors)):
        anchor_ts = anchors.loc[i, "anchor_ts_hour"]
        for h in range(OUTPUT_HORIZON):
            hour = h + 1
            rows.append(
                {
                    "anchor_ts_hour": anchor_ts,
                    "target_hour": hour,
                    "actual_price": actual[i, h],
                    "persistence_price": persistence[i, h],
                    "predicted_residual": residual_pred[i, h],
                    "predicted_price": final_pred[i, h],
                    "absolute_error": abs(actual[i, h] - final_pred[i, h]),
                    "persistence_error": abs(actual[i, h] - persistence[i, h]),
                }
            )
    return pd.DataFrame(rows)


def plot_loss_curve(history: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Huber loss (scaled residual)")
    plt.title("LSTM residual — training curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_horizon_mae(horizon_errors: dict[int, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hours = sorted(horizon_errors.keys())
    values = [horizon_errors[h] for h in hours]
    plt.figure(figsize=(9, 5))
    plt.bar(hours, values, color="steelblue")
    plt.xlabel("Forecast horizon (hours)")
    plt.ylabel("MAE (TL/MWh)")
    plt.title("Residual LSTM — final prediction MAE by horizon")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_vs_persistence(
    persistence_mae: float,
    final_mae: float,
    direct_lstm_mae: float | None,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["Persistence", "LSTM residual"]
    values = [persistence_mae, final_mae]
    colors = ["#6baed6", "#2171b5"]
    if direct_lstm_mae is not None:
        labels.append("LSTM direct")
        values.append(direct_lstm_mae)
        colors.append("#969696")

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, values, color=colors)
    plt.ylabel("Test MAE (TL/MWh)")
    plt.title("Test MAE comparison")
    plt.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.1f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def assess_fit(history: dict) -> str:
    train_final = history["train_loss"][-1]
    val_final = history["val_loss"][-1]
    val_best = min(history["val_loss"])
    gap = val_final - train_final

    if val_best > train_final * 1.5 and gap > 0.05:
        return "likely_overfit (validation loss above train, widening gap)"
    if train_final > val_final * 1.2 and gap < -0.02:
        return "likely_underfit (train loss higher than validation — check scaling or capacity)"
    if gap > 0.1:
        return "mild_overfit (validation loss notably above train)"
    return "reasonable_fit (train and validation losses close at stop)"


def write_metrics_md(report: dict) -> str:
    cmp_ = report.get("persistence_comparison", {})
    lines = [
        "# LSTM Residual Metrics",
        "",
        "Final prediction: `persistence + inverse_transform(predicted_residual)`",
        "",
        f"- **Device:** {report['device']}",
        f"- **Epochs run:** {report['epochs_run']}",
        f"- **Best epoch:** {report['best_epoch']}",
        f"- **Final train loss:** {report['final_train_loss']:.6f}",
        f"- **Final validation loss:** {report['final_val_loss']:.6f}",
        f"- **Best validation loss:** {report['best_val_loss']:.6f}",
        f"- **Fit assessment:** {report['fit_assessment']}",
        "",
        "## Final prediction metrics (TL/MWh)",
        "",
        f"- MAE: {report['test_mae']:.4f}",
        f"- RMSE: {report['test_rmse']:.4f}",
        f"- MAPE (all): {report['test_mape']:.4f}%",
        f"- MAPE (actual > {MAPE_MASK_THRESHOLD}): {report['masked_mape_actual_gt_100']:.4f}%",
        f"- Zero-price MAE: {report.get('zero_price_mae', 'n/a')}",
        "",
        f"- **Worst horizon:** h{report['worst_horizon']} (MAE {report['worst_horizon_mae']:.4f})",
        "",
        "## vs persistence baseline",
        "",
        f"- Persistence MAE: {cmp_.get('persistence_mae', 'n/a')}",
        f"- Residual LSTM MAE: {cmp_.get('final_mae', report['test_mae'])}",
        f"- Improvement vs persistence: {cmp_.get('improvement_pct', 'n/a')}%",
        f"- **{cmp_.get('interpretation', '')}**",
        "",
        "## Horizon MAE (final prediction)",
        "",
        "| Hour | MAE |",
        "|-----:|----:|",
    ]
    for h, v in sorted(report["horizon_mae"].items(), key=lambda x: int(x[0])):
        lines.append(f"| {h} | {v:.4f} |")
    return "\n".join(lines)


def smoke_test(cfg: TrainConfig | None = None) -> dict:
    cfg = cfg or TrainConfig(smoke_test=True)
    cfg.max_epochs = 1
    cfg.patience = 1

    set_seed()
    device = pick_device()
    verify_array_paths()

    train_ds = MmapSequenceDataset(ARRAY_PATHS["X_train"], ARRAY_PATHS["y_train"])
    input_size = train_ds.feature_count
    print("Device:", device)
    print("Input size:", input_size)
    print("Train samples:", len(train_ds))

    model = StackedLSTM(input_size=input_size).to(device)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    train_loader = make_loader(
        ARRAY_PATHS["X_train"],
        ARRAY_PATHS["y_train"],
        batch_size=cfg.batch_size,
        device=device,
    )
    val_loader = make_loader(
        ARRAY_PATHS["X_val"],
        ARRAY_PATHS["y_val"],
        batch_size=cfg.batch_size,
        device=device,
    )

    t0 = time.time()
    train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = run_epoch(model, val_loader, criterion, None, device)

    xb, yb = next(iter(train_loader))
    with torch.no_grad():
        preds = model(xb.to(device))

    result = {
        "smoke_test": True,
        "device": str(device),
        "input_size": input_size,
        "train_loss_epoch_1": train_loss,
        "val_loss_epoch_1": val_loss,
        "elapsed_seconds": round(time.time() - t0, 2),
        "batch_x_shape": list(xb.shape),
        "batch_y_shape": list(yb.shape),
        "batch_preds_shape": list(preds.shape),
        "memory_mode": "mmap_lazy_per_sample",
    }
    print("\n=== Residual LSTM smoke test OK ===")
    print(json.dumps(result, indent=2))
    return result


def train_model(cfg: TrainConfig | None = None) -> dict:
    cfg = cfg or TrainConfig()
    if cfg.smoke_test:
        return smoke_test(cfg)

    set_seed()
    device = pick_device()
    verify_array_paths()

    train_ds = MmapSequenceDataset(ARRAY_PATHS["X_train"], ARRAY_PATHS["y_train"])
    input_size = train_ds.feature_count

    residual_scaler = joblib.load(MODEL_DIR / "residual_target_scaler.pkl")
    anchor_test = MODEL_DIR / "anchor_test.csv"

    model = StackedLSTM(input_size=input_size).to(device)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    train_loader = make_loader(
        ARRAY_PATHS["X_train"],
        ARRAY_PATHS["y_train"],
        batch_size=cfg.batch_size,
        device=device,
    )
    val_loader = make_loader(
        ARRAY_PATHS["X_val"],
        ARRAY_PATHS["y_val"],
        batch_size=cfg.batch_size,
        device=device,
    )
    test_loader = make_loader(
        ARRAY_PATHS["X_test"],
        ARRAY_PATHS["y_test"],
        batch_size=cfg.batch_size,
        device=device,
    )

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0

    print("Device:", device)
    print("Training residual LSTM (mmap lazy loading)")

    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = run_epoch(model, val_loader, criterion, None, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch:03d}/{cfg.max_epochs} "
            f"train={train_loss:.6f} val={val_loss:.6f} "
            f"({time.time() - t0:.1f}s)"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                print(f"Early stopping at epoch {epoch} (patience={cfg.patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": input_size,
            "hidden_size_1": HIDDEN_SIZE_1,
            "hidden_size_2": HIDDEN_SIZE_2,
            "dropout": DROPOUT,
            "output_horizon": OUTPUT_HORIZON,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "model_kind": "residual_lstm",
        },
        OUTPUT_MODEL,
    )

    final_train_loss = run_epoch(model, train_loader, criterion, None, device)
    final_val_loss = run_epoch(model, val_loader, criterion, None, device)

    y_residual_pred_scaled = predict_loader(model, test_loader, device)
    residual_pred = residual_scaler.inverse_transform(y_residual_pred_scaled)

    y_test_actual, persistence = load_price_and_persistence(anchor_test)
    final_pred = persistence + residual_pred

    persistence_mae = mae(y_test_actual, persistence)
    final_mae = mae(y_test_actual, final_pred)
    improvement_pct = (persistence_mae - final_mae) / persistence_mae * 100

    h_mae = horizon_mae(y_test_actual, final_pred)
    worst_h = max(h_mae, key=h_mae.get)
    zero_mask = y_test_actual == 0

    pred_df = build_predictions_csv(
        anchor_test,
        y_test_actual,
        persistence,
        residual_pred,
        final_pred,
    )
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = PREDICTIONS_DIR / "lstm_residual_test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    direct_lstm_mae = None
    if DIRECT_LSTM_METRICS.exists():
        direct_lstm_mae = json.loads(DIRECT_LSTM_METRICS.read_text())["test_mae"]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(history, FIGURES_DIR / "lstm_residual_loss_curve.png")
    plot_horizon_mae(h_mae, FIGURES_DIR / "lstm_residual_horizon_mae.png")
    plot_vs_persistence(
        persistence_mae,
        final_mae,
        direct_lstm_mae,
        FIGURES_DIR / "lstm_residual_vs_persistence.png",
    )

    interpretation = (
        "Residual LSTM beats persistence baseline"
        if final_mae < persistence_mae
        else "Persistence baseline still better than residual LSTM"
    )

    report = {
        "model_kind": "residual_lstm",
        "device": str(device),
        "input_size": input_size,
        "train_samples": len(train_ds),
        "val_samples": len(MmapSequenceDataset(ARRAY_PATHS["X_val"], ARRAY_PATHS["y_val"])),
        "test_samples": len(MmapSequenceDataset(ARRAY_PATHS["X_test"], ARRAY_PATHS["y_test"])),
        "epochs_run": len(history["train_loss"]),
        "best_epoch": best_epoch,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "best_val_loss": best_val,
        "fit_assessment": assess_fit(history),
        "test_mae": final_mae,
        "test_rmse": rmse(y_test_actual, final_pred),
        "test_mape": mape(y_test_actual, final_pred),
        "masked_mape_actual_gt_100": masked_mape(y_test_actual, final_pred),
        "horizon_mae": {str(k): v for k, v in h_mae.items()},
        "worst_horizon": worst_h,
        "worst_horizon_mae": h_mae[worst_h],
        "zero_price_mae": subset_mae(y_test_actual, final_pred, zero_mask),
        "zero_price_hours": int(zero_mask.sum()),
        "persistence_comparison": {
            "persistence_mae": persistence_mae,
            "final_mae": final_mae,
            "mae_delta_final_minus_persistence": final_mae - persistence_mae,
            "improvement_pct": improvement_pct,
            "lstm_better_than_persistence": final_mae < persistence_mae,
            "interpretation": interpretation,
            "direct_lstm_mae": direct_lstm_mae,
        },
        "model_path": str(OUTPUT_MODEL),
        "predictions_path": str(pred_path),
        "memory_mode": "mmap_lazy_per_sample",
    }

    METRICS_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    METRICS_MD.write_text(write_metrics_md(report), encoding="utf-8")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train residual LSTM for PTF forecasting")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one epoch only (memory-safe sanity check)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(smoke_test=args.smoke_test)

    if cfg.smoke_test:
        smoke_test(cfg)
        return

    report = train_model(cfg)
    cmp_ = report["persistence_comparison"]
    print("\n=== Residual LSTM training complete ===")
    print(f"Model saved: {report['model_path']}")
    print(f"Final train loss: {report['final_train_loss']:.6f}")
    print(f"Final val loss:   {report['final_val_loss']:.6f}")
    print(f"Test MAE (final): {report['test_mae']:.4f} TL/MWh")
    print(f"Persistence MAE:  {cmp_['persistence_mae']:.4f} TL/MWh")
    print(f"Improvement vs persistence: {cmp_['improvement_pct']:.2f}%")
    print(f"{cmp_['interpretation']}")
    print(f"Metrics: {METRICS_JSON}")


if __name__ == "__main__":
    main()

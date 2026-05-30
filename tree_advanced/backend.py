"""LightGBM / XGBoost / sklearn backends with sample weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np

EARLY_STOPPING_ROUNDS = 50
MAX_BOOST_ROUNDS = 1500


def pick_backend() -> str:
    try:
        import lightgbm  # noqa: F401

        return "lightgbm"
    except ImportError:
        pass
    try:
        import xgboost  # noqa: F401

        return "xgboost"
    except ImportError:
        pass
    return "sklearn"


def train_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    backend: str = "lightgbm",
) -> Any:
    if backend == "lightgbm":
        import lightgbm as lgb

        train_set = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
        params = {
            "objective": "regression",
            "metric": "mae",
            "verbosity": -1,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_data_in_leaf": 40,
            "lambda_l1": 0.2,
            "lambda_l2": 1.0,
            "feature_fraction": 0.75,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "seed": 42,
        }
        return lgb.train(
            params,
            train_set,
            num_boost_round=MAX_BOOST_ROUNDS,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(0)],
        )

    if backend == "xgboost":
        import xgboost as xgb

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="mae",
            learning_rate=0.03,
            max_depth=7,
            n_estimators=MAX_BOOST_ROUNDS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            subsample=0.8,
            colsample_bytree=0.75,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        return model

    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.05,
        max_depth=10,
        random_state=42,
    )
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    return model


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    backend: str = "lightgbm",
) -> Any:
    if backend == "lightgbm":
        import lightgbm as lgb

        train_set = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 30,
            "feature_fraction": 0.8,
            "seed": 42,
            "is_unbalance": True,
        }
        return lgb.train(
            params,
            train_set,
            num_boost_round=800,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(0)],
        )

    if backend == "xgboost":
        import xgboost as xgb

        model = xgb.XGBClassifier(
            eval_metric="logloss",
            learning_rate=0.05,
            max_depth=6,
            n_estimators=800,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        return model

    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(max_iter=300, random_state=42)
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    return model


def predict_model(model: Any, X: np.ndarray, *, backend: str) -> np.ndarray:
    if backend == "lightgbm":
        return model.predict(X)
    return model.predict(X)


def save_model(model: Any, path: Path, *, backend: str, kind: Literal["regressor", "classifier"]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if backend == "lightgbm":
        model.save_model(str(path))
    else:
        joblib.dump(model, path)


def load_model(path: Path, *, backend: str) -> Any:
    if backend == "lightgbm":
        import lightgbm as lgb

        return lgb.Booster(model_file=str(path))
    return joblib.load(path)


def model_suffix(backend: str) -> str:
    return ".txt" if backend == "lightgbm" else ".joblib"

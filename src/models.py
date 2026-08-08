from __future__ import annotations

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from .config import FEATURES, MODEL_DIR, RANDOM_STATE
from .preprocessing import build_preprocessor


def build_random_forest() -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def build_neural_network() -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0005,
            learning_rate_init=0.001,
            max_iter=1500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=RANDOM_STATE,
        )),
    ])


def metrics(y_true, y_pred) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(rmse, 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


def save_model(model, filename: str) -> None:
    joblib.dump(model, MODEL_DIR / filename)


def load_model(filename: str):
    return joblib.load(MODEL_DIR / filename)

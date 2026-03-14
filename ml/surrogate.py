"""Surrogate-model training helpers for Task 2."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

FEATURE_COLUMNS = [
    "input_temperature_C",
    "input_pressure_bar",
    "input_ghsv_h",
    "input_target_x_co",
    "input_total_flow_kmol_h",
    "input_tube_inner_diameter_m",
    "input_particle_diameter_m",
    "input_void_fraction",
    "input_purge_fraction",
]

TARGET_COLUMNS = [
    "target_rate_kgph",
    "target_fraction",
    "specific_energy_kwh_per_kg_target",
    "compressor_power_mw",
    "cooling_duty_mw",
    "delta_p_bar",
    "feasible_numeric",
]


def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    train_df = df.copy()
    train_df = train_df[train_df["run_status"] == "success"].copy()
    train_df = train_df.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=FEATURE_COLUMNS + TARGET_COLUMNS[:-1])
    train_df["feasible_numeric"] = train_df["feasible"].astype(int)
    return train_df


def train_and_save_surrogate(df: pd.DataFrame, out_dir: str | Path, metadata: dict) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = prepare_training_frame(df)
    if len(train_df) < 25:
        raise ValueError("Need at least 25 successful cases to train the surrogate reliably")

    X = train_df[FEATURE_COLUMNS]
    y = train_df[TARGET_COLUMNS]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=metadata.get("random_seed", 42))

    model = MultiOutputRegressor(
        ExtraTreesRegressor(
            n_estimators=300,
            random_state=metadata.get("random_seed", 42),
            min_samples_leaf=2,
            n_jobs=-1,
        )
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    metrics = {}
    for idx, target in enumerate(TARGET_COLUMNS):
        metrics[target] = {
            "mae": float(mean_absolute_error(y_val.iloc[:, idx], y_pred[:, idx])),
            "r2": float(r2_score(y_val.iloc[:, idx], y_pred[:, idx])),
        }

    model_path = out_dir / "surrogate_model.joblib"
    metadata_path = out_dir / "surrogate_metadata.json"
    joblib.dump(model, model_path)

    full_metadata = {
        **metadata,
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "n_training_rows": int(len(train_df)),
        "metrics": metrics,
    }
    metadata_path.write_text(json.dumps(full_metadata, indent=2), encoding="utf-8")

    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "metrics": metrics,
        "n_training_rows": int(len(train_df)),
    }

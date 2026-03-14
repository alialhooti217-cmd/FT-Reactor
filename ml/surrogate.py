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
]

OPTIONAL_TARGET_COLUMNS = [
    "feasible_numeric",
]


def _check_required_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing {label} columns in dataset: {missing}")


def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    train_df = df.copy()

    if "run_status" in train_df.columns:
        train_df = train_df[train_df["run_status"] == "success"].copy()

    if "feasible" in train_df.columns:
        train_df = train_df[train_df["feasible"] == True].copy()  # noqa: E712

    _check_required_columns(train_df, FEATURE_COLUMNS, "feature")
    _check_required_columns(train_df, TARGET_COLUMNS, "target")

    train_df = train_df.replace([float("inf"), float("-inf")], pd.NA)

    if "feasible" in train_df.columns:
        train_df["feasible_numeric"] = train_df["feasible"].astype(int)

    required_for_dropna = FEATURE_COLUMNS + TARGET_COLUMNS
    train_df = train_df.dropna(subset=required_for_dropna).copy()

    return train_df


def _safe_r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def train_and_save_surrogate(df: pd.DataFrame, out_dir: str | Path, metadata: dict) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = prepare_training_frame(df)

    if len(train_df) < 25:
        raise ValueError(
            f"Need at least 25 successful and feasible cases to train the surrogate reliably. "
            f"Only found {len(train_df)} usable rows."
        )

    X = train_df[FEATURE_COLUMNS].copy()
    y = train_df[TARGET_COLUMNS].copy()

    random_seed = int(metadata.get("random_seed", 42))

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_seed,
    )

    model = MultiOutputRegressor(
        ExtraTreesRegressor(
            n_estimators=300,
            random_state=random_seed,
            min_samples_leaf=2,
            n_jobs=-1,
        )
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    metrics: dict[str, dict[str, float]] = {}
    for idx, target in enumerate(TARGET_COLUMNS):
        y_true_col = y_val.iloc[:, idx]
        y_pred_col = y_pred[:, idx]

        metrics[target] = {
            "mae": float(mean_absolute_error(y_true_col, y_pred_col)),
            "r2": _safe_r2(y_true_col, y_pred_col),
        }

    model_path = out_dir / "surrogate_model.joblib"
    metadata_path = out_dir / "surrogate_metadata.json"
    training_data_path = out_dir / "surrogate_training_data.csv"

    joblib.dump(model, model_path)
    train_df.to_csv(training_data_path, index=False)

    full_metadata = {
        **metadata,
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "optional_target_columns": OPTIONAL_TARGET_COLUMNS,
        "model_type": "MultiOutputRegressor(ExtraTreesRegressor)",
        "n_training_rows": int(len(train_df)),
        "n_train_rows_split": int(len(X_train)),
        "n_validation_rows_split": int(len(X_val)),
        "metrics": metrics,
        "artifacts": {
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "training_data_path": str(training_data_path),
        },
    }

    metadata_path.write_text(json.dumps(full_metadata, indent=2), encoding="utf-8")

    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "training_data_path": str(training_data_path),
        "metrics": metrics,
        "n_training_rows": int(len(train_df)),
        "n_train_rows_split": int(len(X_train)),
        "n_validation_rows_split": int(len(X_val)),
    }
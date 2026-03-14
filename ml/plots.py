from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

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

DEFAULT_TARGETS_TO_PLOT = [
    "target_rate_kgph",
    "specific_energy_kwh_per_kg_target",
    "delta_p_bar",
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_feasibility_scatter(
    df: pd.DataFrame,
    out_dir: str | Path,
    x_col: str = "input_temperature_C",
    y_col: str = "input_pressure_bar",
) -> str:
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    if x_col not in df.columns:
        raise KeyError(f"Column '{x_col}' not found in dataframe.")
    if y_col not in df.columns:
        raise KeyError(f"Column '{y_col}' not found in dataframe.")
    if "feasible" not in df.columns:
        raise KeyError("Column 'feasible' not found in dataframe.")

    fig, ax = plt.subplots(figsize=(8, 6))

    feasible_df = df[df["feasible"] == True]   # noqa: E712
    infeasible_df = df[df["feasible"] == False]  # noqa: E712

    ax.scatter(feasible_df[x_col], feasible_df[y_col], alpha=0.5, label="Feasible")
    ax.scatter(infeasible_df[x_col], infeasible_df[y_col], alpha=0.5, label="Infeasible")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("Feasible vs Infeasible Design Space")
    ax.legend()
    ax.grid(True)

    out_path = out_dir / "feasibility_scatter.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    return str(out_path)


def plot_delta_p_histogram(
    df: pd.DataFrame,
    out_dir: str | Path,
    delta_p_limit: float | None = None,
) -> str:
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    if "delta_p_bar" not in df.columns:
        raise KeyError("Column 'delta_p_bar' not found in dataframe.")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df["delta_p_bar"].dropna(), bins=30)

    if delta_p_limit is not None:
        ax.axvline(delta_p_limit, linestyle="--", linewidth=2, label=f"Limit = {delta_p_limit:.2f} bar")
        ax.legend()

    ax.set_xlabel("Pressure Drop (bar)")
    ax.set_ylabel("Count")
    ax.set_title("Pressure Drop Distribution")
    ax.grid(True)

    out_path = out_dir / "delta_p_histogram.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    return str(out_path)


def plot_feature_importance(
    model_path: str | Path,
    out_dir: str | Path,
    target_index: int = 0,
) -> str:
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    model = joblib.load(model_path)

    if not hasattr(model, "estimators_"):
        raise AttributeError("Loaded model does not have 'estimators_' attribute.")

    if target_index < 0 or target_index >= len(model.estimators_):
        raise IndexError(
            f"target_index={target_index} is out of range for "
            f"{len(model.estimators_)} trained target estimators."
        )

    estimator = model.estimators_[target_index]
    if not hasattr(estimator, "feature_importances_"):
        raise AttributeError("Underlying estimator does not expose 'feature_importances_'.")

    importances = estimator.feature_importances_

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(FEATURE_COLUMNS, importances)
    ax.set_xlabel("Features")
    ax.set_ylabel("Importance")
    ax.set_title(f"Feature Importance for Target Index {target_index}")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True)

    out_path = out_dir / f"feature_importance_target_{target_index}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    return str(out_path)


def plot_predicted_vs_actual(
    model_path: str | Path,
    training_data_path: str | Path,
    out_dir: str | Path,
    target_cols: list[str] | None = None,
) -> list[str]:
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    model_path = Path(model_path)
    training_data_path = Path(training_data_path)

    model = joblib.load(model_path)
    df = pd.read_csv(training_data_path)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            raise KeyError(f"Feature column '{col}' not found in training data.")

    if target_cols is None:
        target_cols = DEFAULT_TARGETS_TO_PLOT

    for target in target_cols:
        if target not in df.columns:
            raise KeyError(f"Target column '{target}' not found in training data.")

    X = df[FEATURE_COLUMNS]
    y_true = df[target_cols]
    y_pred = model.predict(X)

    metadata_path = model_path.with_name("surrogate_metadata.json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    trained_targets = metadata.get("target_columns", [])

    output_paths: list[str] = []

    for target in target_cols:
        if target not in trained_targets:
            raise KeyError(f"Target '{target}' not found in trained target columns: {trained_targets}")

        idx = trained_targets.index(target)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(y_true[target], y_pred[:, idx], alpha=0.5)

        min_val = min(y_true[target].min(), y_pred[:, idx].min())
        max_val = max(y_true[target].max(), y_pred[:, idx].max())
        ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")

        ax.set_xlabel(f"Actual {target}")
        ax.set_ylabel(f"Predicted {target}")
        ax.set_title(f"Predicted vs Actual: {target}")
        ax.grid(True)

        out_path = out_dir / f"predicted_vs_actual_{target}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

        output_paths.append(str(out_path))

    return output_paths
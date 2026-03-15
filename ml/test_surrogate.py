from pathlib import Path
import joblib
import json
import pandas as pd

MODEL_PATH = Path("models/surrogate_model.joblib")
META_PATH = Path("models/surrogate_metadata.json")

model = joblib.load(MODEL_PATH)

metadata = json.loads(META_PATH.read_text())

features = metadata["feature_columns"]
targets = metadata["target_columns"]

# Example test input
x = {
    "input_temperature_C": 210,
    "input_pressure_bar": 25,
    "input_ghsv_h": 2000,
    "input_target_x_co": 0.70,
    "input_total_flow_kmol_h": 1300,
    "input_tube_inner_diameter_m": 0.042,
    "input_particle_diameter_m": 0.0012,
    "input_void_fraction": 0.42,
    "input_purge_fraction": 0.03,
}

X = pd.DataFrame([x])[features]

y_pred = model.predict(X)[0]

print("\nSurrogate prediction:\n")

for name, value in zip(targets, y_pred):
    print(f"{name:35s} : {value:.5f}")

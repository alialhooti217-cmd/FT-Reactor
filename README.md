# Fischer–Tropsch Reactor Simulation and Optimization

This repository contains a **Python-based modular simulation framework for a Fischer–Tropsch (FT) reactor**, developed as part of an **IR4 engineering project** at Sultan Qaboos University.

The project combines **process modeling, dataset generation, machine learning, and optimization** to explore reactor performance and operating conditions.

---

# Project Overview

The Fischer–Tropsch process converts **syngas (CO + H₂)** into hydrocarbons through catalytic reactions. These reactors are commonly used in **gas-to-liquids (GTL)** and **synthetic fuel production systems**. ([sciencedirect.com][1])

This project provides a computational framework to:

* simulate FT reactor behavior (including recycle loop convergence)
* generate large datasets from parametric studies
* train surrogate models (ML-based fast approximations)
* perform optimization of reactor operating conditions

The code is designed for **modularity, reproducibility, and integration with machine learning workflows**.

---

# Repository Structure

```
FT-Reactor/
├── README.md
├── requirements.txt                 ← Python dependencies
├── config.yaml                      ← central configuration file
├── main.py                          ← entry point for single simulations
├── run_interactive.py               ← rich terminal UI for user-defined inputs
├── visualize.py                     ← generates 2D and 3D plots from the dataset
├── app.py                           ← Streamlit web dashboard (browser UI)
├── Copy_of_FT_Reactor.ipynb         ← standalone prototype sizing notebook
│
├── src/                             ← core simulation modules
│   ├── constants.py                 ← molecular weights, ASF params, thermo data
│   ├── asf.py                       ← Anderson-Schulz-Flory distribution model
│   ├── feed.py                      ← feed composition normalization
│   ├── geometry.py                  ← reactor geometry sizing and search
│   ├── hydraulics.py                ← Ergun pressure drop calculations
│   ├── kinetics.py                  ← kinetic rate law and catalyst volume
│   ├── mass.py                      ← mass balance and stoichiometry
│   ├── thermo.py                    ← thermodynamic calculations
│   ├── utilities.py                 ← helper functions (ASF, Cp, MW)
│   └── reactor.py                   ← main FTReactor class and recycle loop
│
├── batch/
│   └── run_batch.py                 ← dataset generation from parameter ranges
│
├── ml/
│   ├── surrogate.py                 ← ML model training (ExtraTreesRegressor)
│   ├── predictor.py                 ← shared surrogate loader and prediction interface
│   ├── optimize_surrogate.py        ← optimisation via trained surrogate
│   ├── plots.py                     ← ML-specific plot utilities
│   ├── test_surrogate.py            ← surrogate smoke test
│   └── verify_optimum.py            ← verify surrogate optimum with full sim
│
├── tests/
│   └── test_sanity.py               ← unit tests
│
├── data/
│   └── processed/
│       ├── dataset.csv              ← all 21,000 simulation cases (~15 MB)
│       ├── dataset_feasible.csv     ← 10,527 feasible cases (~7 MB)
│       └── run_metadata.json        ← generation metadata and KPI definitions
│
├── models/
│   └── surrogate_metadata.json      ← ML model performance metrics
│
└── plots/                           ← generated visualisations (2D and 3D)
    ├── 01_feasibility_map.png
    ├── 02_pressure_drop_histogram.png
    ├── 03_conversion_vs_selectivity.png
    ├── 04_energy_vs_production.png
    ├── 05_kpi_correlation_heatmap.png
    ├── 06_3d_energy_landscape.png
    ├── 07_3d_selectivity_map.png
    └── 08_3d_pressure_drop.png
```

---

# Key Features

### 1. Reactor Model

The core simulation (`src/reactor.py`) implements a full multitubular fixed-bed FT reactor with recycle loop convergence. It calls:

| Module | Responsibility |
|--------|---------------|
| `src/asf.py` | Anderson-Schulz-Flory hydrocarbon product distribution |
| `src/mass.py` | Stoichiometry, recycle/purge, gas-liquid separation |
| `src/thermo.py` | Mixture MW, density, heat capacity, heat of reaction |
| `src/kinetics.py` | CO conversion rate law, catalyst volume |
| `src/geometry.py` | Tube bundle sizing, shell diameter, L/D search |
| `src/hydraulics.py` | Ergun equation pressure drop |
| `src/feed.py` | Feed composition normalization |
| `src/constants.py` | Species properties, global constants |

The recycle loop converges when stream flows stabilize to within `loop_tol = 1e-6` (configurable), with a maximum of 50 iterations.

**Feasibility criteria checked after each simulation:**

* ΔP ≤ `max_delta_p_bar` (default 4.0 bar)
* Shell diameter Ds ≤ `max_shell_diameter_m` (default 4.0 m)
* L/D in `[ld_min, ld_max]` (default 3.0–6.0)
* Superficial velocity in `[v_min, v_max]` (default 0.05–4.0 m/s)
* C8–C16 molar fraction ≥ `min_target_fraction` (default 0.08)

---

### 2. Batch Simulation Engine

Generates datasets by sampling parameter combinations and running simulations in bulk.

```bash
python batch/run_batch.py
```

Workflow:
1. Stratified random sampling from 10 parameter ranges (seed 42)
2. Each case runs a full `FTReactor` simulation
3. Results saved to `data/processed/dataset.csv` (all cases) and `data/processed/dataset_feasible.csv` (feasible only)

**Current dataset:** 21,000 simulated cases → 10,527 feasible (50.1%)

---

### 3. Machine Learning Surrogate Model

A surrogate model approximates the reactor simulation output to enable fast evaluation during optimization.

```bash
python -m ml.surrogate
```

**Architecture:** `MultiOutputRegressor(ExtraTreesRegressor(n_estimators=300))`

| | |
|---|---|
| **Input features (9)** | Temperature, pressure, GHSV, target_x_co, total flow, tube_Di, particle_Di, void_fraction, purge_fraction |
| **Output targets (6)** | target_rate_kgph, target_fraction, specific_energy_kwh_per_kg_target, compressor_power_mw, cooling_duty_mw, delta_p_bar |
| **Training split** | 80% train / 20% validation (from 10,527 feasible cases) |
| **Saved to** | `models/` |

---

### 4. Optimization Module

Uses the trained surrogate to rapidly search for operating conditions that minimize the primary objective (`specific_energy_kwh_per_kg_target` by default).

```bash
python -m ml.optimize_surrogate
```

Samples large numbers of input combinations and predicts outputs using the surrogate — far faster than running full simulations.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/alialhooti217-cmd/FT-Reactor.git
cd FT-Reactor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Simulation

Run a single simulation using the base configuration:

```bash
python main.py
```

All parameters are controlled through `config.yaml` at the project root.

### Interactive Terminal Mode

A colorful, formatted terminal UI using the `rich` library:

```bash
python run_interactive.py
```

At startup, choose the simulation mode:
* **🤖 ML Surrogate (default)** — instant predictions from the trained model, shows R² accuracy scores, optional comparison against full physics
* **⚗️ Full Physics** — runs all equations, shows geometry, recycle loop, and energy details

Features color-coded KPI bar charts, a parameter range table, feasibility banner, and an ASCII reactor diagram.

### Web Dashboard (Streamlit)

A browser-based interactive dashboard with sliders and live charts:

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Features:
* Sidebar mode selector: **ML Surrogate** (instant) or **Full Physics**
* Sidebar sliders for all 9 operating parameters
* Live KPI metric cards and feasibility badge after each run
* **ML mode tabs:** KPI charts · ML model info (R² scores, training stats) · Compare vs Physics · Dataset plots
* **Physics mode tabs:** KPI charts · Reactor geometry + ASCII diagram · Energy & recycle loop · Dataset plots

---

# Key Configuration Parameters (`config.yaml`)

| Parameter | Default | Tunable Range | Description |
|-----------|---------|---------------|-------------|
| `temperature_c` | 220 | 210–235 °C | Reactor temperature |
| `pressure_bar` | 25 | 20–28 bar | Operating pressure |
| `ghsv` | 1800 | 1500–2200 h⁻¹ | Gas hourly space velocity |
| `total_flow_kmol_h` | 1200 | 900–1800 kmol/h | Fresh feed total molar flow |
| `target_x_co` | 0.72 | 0.60–0.78 | Target single-pass CO conversion |
| `tube_Di_m` | 0.042 | 0.040–0.046 m | Tube inner diameter |
| `particle_Di_m` | 0.0012 | 0.0011–0.0015 m | Catalyst particle diameter |
| `void_fraction` | 0.42 | 0.41–0.46 | Bed void fraction |
| `purge_fraction` | 0.03 | 0.02–0.06 | Recycle purge fraction |
| `max_delta_p_bar` | 4.0 | — | Max allowed pressure drop |
| `max_shell_diameter_m` | 4.0 | — | Max reactor shell diameter |

---

# Running Batch Simulations

```bash
python batch/run_batch.py
```

This will:
1. Sample reactor operating conditions from the ranges in `config.yaml`
2. Run simulations in sequence
3. Save all results to `data/processed/dataset.csv`
4. Save feasible-only results to `data/processed/dataset_feasible.csv`

---

# Training the Surrogate Model

After generating simulation data:

```bash
python -m ml.surrogate
```

The trained model and metadata will be saved to `models/`.

---

# KPI Definitions

| KPI | Description |
|-----|-------------|
| `target_rate_kgph` | Mass flow rate of C8–C16 hydrocarbon products (kg/h) |
| `target_fraction` | Molar fraction of C8–C16 in all hydrocarbon products |
| `specific_energy_kwh_per_kg_target` | Energy consumption per kg of C8–C16 produced (primary optimization objective) |
| `compressor_power_mw` | Recycle compressor electrical power (MW) |
| `cooling_duty_mw` | Reactor cooling duty required (MW) |
| `delta_p_bar` | Pressure drop across the packed bed (bar) |

---

# Visualisations

Generate all 2D and 3D plots from the simulation dataset:

```bash
python visualize.py
```

This produces 8 plots in the `plots/` directory:

**2D plots:**
1. **Feasibility map** — Temperature vs Pressure coloured by feasible/infeasible
2. **Pressure-drop histogram** — distribution with the 4-bar constraint line
3. **Conversion vs selectivity** — CO conversion vs C8–C16 fraction, coloured by temperature
4. **Energy vs production** — specific energy vs product rate, coloured by pressure
5. **KPI correlation heatmap** — pairwise correlations between all inputs and outputs

**3D plots:**
6. **Energy landscape** — T x P x GHSV coloured by specific energy
7. **Selectivity map** — T x P x CO conversion coloured by C8–C16 fraction
8. **Pressure-drop surface** — T x P x pressure drop

---

# Testing

Run the unit tests with:

```bash
pytest
```

Tests in `tests/test_sanity.py` verify pressure drop limits, particle diameter effects, and purge fraction behavior.

---

# Prototype Notebook

An interactive sizing prototype is available in:

```
Copy_of_FT_Reactor.ipynb
```

This notebook was used to develop and validate the reactor sizing algorithm before it was refactored into the `src/` modules. It is useful for:

* manually exploring individual design cases
* verifying geometry and hydraulics calculations
* understanding the N-sensitivity scan (parallel reactors)

> **Note:** The logic in this notebook has been superseded by the modular source code in `src/`. For production use, run `main.py` or `batch/run_batch.py`.

---

# Future Improvements

Possible extensions include:

* multi-objective optimization (Pareto front for yield vs. energy)
* catalyst deactivation modeling
* advanced optimization algorithms (genetic algorithms, Bayesian optimization)
* uncertainty quantification
* `pyproject.toml` packaging and distribution

---

# Author

Dr. Ali's Team
Chemical and Process Engineering
Sultan Qaboos University

---

# License

This project is for academic and research purposes.

[1]: https://www.sciencedirect.com/topics/engineering/fischer-tropsch-reactor?utm_source=chatgpt.com "Fischer-Tropsch Reactor - an overview | ScienceDirect Topics"

# Fischer–Tropsch Reactor Simulation and Optimization

This repository contains a **Python-based modular simulation framework for a Fischer–Tropsch (FT) reactor**, developed as part of an **IR4 engineering project**.

The project combines **process modeling, dataset generation, machine learning, and optimization** to explore reactor performance and operating conditions.

---

# Project Overview

The Fischer–Tropsch process converts **syngas (CO + H₂)** into hydrocarbons through catalytic reactions. These reactors are commonly used in **gas-to-liquids (GTL)** and **synthetic fuel production systems**. ([sciencedirect.com][1])

This project provides a computational framework to:

* simulate FT reactor behavior
* generate large datasets from parametric studies
* train surrogate models
* perform optimization of reactor performance

The code is designed for **modularity, reproducibility, and integration with machine learning workflows**.

---

# Repository Structure

```
FT-Reactor/
│
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── configs/
│   ├── training.yaml
│   └── user_job.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
│
├── notebooks/
│   └── ft_reactor_exploration.ipynb
│
├── src/
│   └── ft_reactor/
│       ├── __init__.py
│       ├── constants.py
│       ├── asf.py
│       ├── mass_balance.py
│       ├── energy.py
│       ├── model.py
│       ├── batch_runner.py
│       ├── surrogate.py
│       ├── optimizer.py
│       └── io_utils.py
│
├── tests/
│   ├── test_asf.py
│   ├── test_mass_balance.py
│   └── test_model.py
│
└── main.py
```

---

# Key Features

### 1️⃣ Reactor Model

The core simulation includes:

* mass balances
* energy balances
* ASF distribution for hydrocarbon products
* thermodynamic calculations
* configurable reactor conditions

Main modules:

```
mass_balance.py
energy.py
asf.py
model.py
```

---

### 2️⃣ Batch Simulation Engine

The framework allows automated generation of datasets using multiple reactor configurations.

Implemented in:

```
batch_runner.py
configs/training.yaml
```

Typical workflow:

1. Define parameter ranges in YAML
2. Run multiple simulations
3. Save results for ML training

---

### 3️⃣ Machine Learning Surrogate Model

A surrogate model approximates the reactor simulation to enable faster evaluation during optimization.

Implemented in:

```
surrogate.py
```

Capabilities include:

* training surrogate models
* saving trained models
* prediction from trained models

---

### 4️⃣ Optimization Module

The optimizer searches for reactor operating conditions that improve selected objectives such as:

* hydrocarbon yield
* selectivity
* energy efficiency

Implemented in:

```
optimizer.py
```

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

Run the main simulation script:

```bash
python main.py
```

Configuration can be modified in:

```
configs/user_job.yaml
```

---

# Running Batch Simulations

To generate datasets for training surrogate models:

```bash
python -m src.ft_reactor.batch_runner
```

This will:

1. Sample reactor operating conditions
2. Run simulations
3. Store outputs in `data/processed/`

---

# Training the Surrogate Model

After generating simulation data:

```bash
python -m src.ft_reactor.surrogate
```

The trained model will be stored in:

```
data/models/
```

---

# Testing

Unit tests ensure correctness of key model components.

Run tests with:

```bash
pytest
```

---

# Notebook Exploration

Interactive analysis is available in:

```
notebooks/ft_reactor_exploration.ipynb
```

This notebook demonstrates:

* running simulations
* exploring output trends
* validating model behavior

---

# Future Improvements

Possible extensions include:

* detailed reactor hydrodynamics
* catalyst deactivation modeling
* process integration with recycle loops
* advanced optimization algorithms
* uncertainty quantification

---

# Author

Dr. Ali's Team
Chemical and Process Engineering
Sultan Qaboos University

---

# License

This project is for academic and research purposes.

[1]: https://www.sciencedirect.com/topics/engineering/fischer-tropsch-reactor?utm_source=chatgpt.com "Fischer-Tropsch Reactor - an overview | ScienceDirect Topics"

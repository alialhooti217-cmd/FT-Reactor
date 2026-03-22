# IR4 Project Report — FT Reactor Simulation & Optimization Platform

**Chemical and Process Engineering | Sultan Qaboos University**

---

## Overview

Designing a Fischer–Tropsch (FT) reactor requires solving tightly coupled equations across
temperature, pressure, kinetics, geometry, and thermodynamics simultaneously. Traditional
approaches are slow, narrow in scope, and highly expert-dependent — making large-scale design
exploration impractical.

This project delivers a fully integrated computational platform that combines physics-based
simulation, machine learning, and interactive visualization into a single end-to-end framework.

---

## Simulation & Dataset

A modular Python simulator solves the complete FT reactor problem from first principles —
including feed normalization, Langmuir–Hinshelwood kinetics, Anderson–Schulz–Flory product
distribution, Ergun pressure drop, and recycle loop convergence. This simulator was executed
**21,000 times** across the full operating parameter space, producing a structured dataset of
which **10,527 cases (50.1%)** met all engineering feasibility constraints.

---

## Machine Learning Surrogate

A machine learning surrogate model (ExtraTreesRegressor) was trained on this dataset to predict
six reactor KPIs — product rate, selectivity, specific energy, compressor power, cooling duty,
and pressure drop — from nine operating inputs, with near-instant response time replacing minutes
of computation.

---

## Interactive Interfaces

Two interactive interfaces make the platform accessible to non-experts:

- **Streamlit web dashboard** — live sliders, ML/Physics mode toggle, and a side-by-side
  comparison tab.
- **Rich terminal UI** — color-coded KPI charts and feasibility indicators.

---

## Key Contributions

What makes this work unique is the seamless integration of:

| Feature        | Detail                                      |
|----------------|---------------------------------------------|
| Scale          | 21,000 simulated cases                      |
| Speed          | Millisecond ML predictions                  |
| Transparency   | Live physics vs. ML comparison              |
| Accessibility  | No coding required                          |

Together, these components transform reactor design into a fast, data-driven, and visual workflow.

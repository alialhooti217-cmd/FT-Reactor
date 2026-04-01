# IR4 Fischer–Tropsch Packed-Bed Reactor Modeling Studio

A complete web-based engineering project for **IR4 FT reactor modeling**, built from the original scientific reactor core.

This repository now includes:
- a **Flask backend API** for full-physics packed-bed reactor simulation,
- a **professional frontend dashboard** (HTML/CSS/JS),
- the original **modular FT reactor logic** in `src/` (reused, not replaced),
- optional ML artifacts retained for offline research workflows.

---

## What this project does

The app simulates a multitubular packed-bed Fischer–Tropsch reactor with recycle and purge loop, computing engineering KPIs such as:
- C8–C16 production rate,
- target product fraction,
- specific energy,
- compressor power,
- cooling duty,
- pressure drop,
- reactor geometry feasibility.

The web UI is intentionally focused on **reactor simulation experience** and excludes ML diagnostic visuals.

---

## New architecture

```text
FT-Reactor/
├── run_web.py                       # Start Flask web server
├── webapp/
│   ├── backend/
│   │   ├── __init__.py              # Flask app factory
│   │   ├── routes.py                # GET /, POST /simulate, GET /health
│   │   └── services/
│   │       └── simulation_service.py# Validation + core model integration
│   └── frontend/
│       ├── templates/
│       │   └── index.html           # Main dashboard page
│       └── static/
│           ├── css/styles.css       # Engineering visual style
│           ├── js/app.js            # Client-side form/API logic
│           └── img/pbr-schematic.svg# Reactor schematic visual
├── src/                             # Core FT reactor science modules (reused)
├── ml/                              # ML utilities kept for offline use
├── data/, models/, plots/           # Existing assets
└── ...
```

---

## API routes

- `GET /` → web dashboard
- `POST /simulate` → run full-physics simulation from user inputs
- `GET /health` → health check JSON

---

## Installation

## 1) Clone
```bash
git clone <your-repo-url>
cd FT-Reactor
```

## 2) Create and activate virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

(Windows PowerShell)
```powershell
.venv\Scripts\Activate.ps1
```

## 3) Install dependencies
```bash
pip install -r requirements.txt
```

---

## Run the website

## Start web app
```bash
python run_web.py
```

Then open:
- `http://127.0.0.1:5000`

## Health check
```bash
curl http://127.0.0.1:5000/health
```

---

## How to use the dashboard

1. Open the landing page.
2. Enter reactor operating conditions and geometry.
3. Click **Run Full Physics Simulation**.
4. Review feasibility + KPI cards and reactor visual panel.

---

## Engineering and implementation notes

- Core reactor logic is reused from `src/reactor.py` through a service layer.
- Input values are validated with engineering bounds before simulation.
- Invalid inputs return clear API errors.
- UI is responsive and designed for technical review/demo settings.

---

## Legacy utilities (still available)

- `python main.py` → single CLI simulation
- `streamlit run app.py` → previous dashboard (legacy)
- `python batch/run_batch.py` → dataset generation and surrogate workflow

---

## Assumptions

- Python 3.10+
- Existing FT model equations in `src/` are the source of truth
- Web endpoint `/simulate` runs the full-physics model (not surrogate)


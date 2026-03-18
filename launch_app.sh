#!/usr/bin/env bash
# FT-Reactor Streamlit Launcher
# Activates the environment (if any), then starts the Streamlit dashboard.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Optional: activate a virtual environment if one exists ---
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
elif [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# --- Launch Streamlit and open the browser automatically ---
streamlit run "$SCRIPT_DIR/app.py" \
    --server.headless false \
    --browser.gatherUsageStats false \
    --server.port 8501

#!/usr/bin/env bash
# Start the complaint API and the Streamlit chat UI in one process group.
set -euo pipefail
cd "$(dirname "$0")/.."

uvicorn api:app --host 0.0.0.0 --port 8000 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT

exec streamlit run app.py --server.port "${PORT:-8501}" --server.address 0.0.0.0

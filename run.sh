#!/usr/bin/env bash
# Start TherapyTrace locally without Docker.
# Backend on :8000, frontend dev server on :5173 with an /api proxy.
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d backend/.venv ]; then
  echo "Creating the backend virtualenv..."
  python3 -m venv backend/.venv
  backend/.venv/bin/pip install -q -r backend/requirements.txt
fi

if [ ! -d frontend/node_modules ]; then
  echo "Installing frontend packages..."
  (cd frontend && npm install)
fi

if ! backend/.venv/bin/python -c "from app.main import app" 2>/tmp/tt_import.log; then
  echo "The backend could not start:"; cat /tmp/tt_import.log
  echo; echo "Try: backend/.venv/bin/pip install -r backend/requirements.txt"
  exit 1
fi

trap 'kill 0' EXIT
(cd backend && .venv/bin/uvicorn app.main:app --reload --port 8000) &
(cd frontend && npm run dev) &

echo
echo "  API   http://localhost:8000/docs"
echo "  App   http://localhost:5173"
echo
wait

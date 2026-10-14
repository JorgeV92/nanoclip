#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Activate .venv first, or supply PYTHON=/absolute/path/to/python.
PYTHON="${PYTHON:-python}"
"$PYTHON" -m ruff check nanoclip scripts benchmarks tests
"$PYTHON" -m ruff format --check nanoclip scripts benchmarks tests
"$PYTHON" -m pytest -q

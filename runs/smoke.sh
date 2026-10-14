#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run --no-sync python -m scripts.train \
  --size tiny --device cpu --batch-size 8 --samples 64 --val-samples 16 \
  --steps 10 --log-every 1 --eval-every 5 --save-every 5 \
  --output-dir checkpoints/smoke "$@"

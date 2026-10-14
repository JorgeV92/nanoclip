#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# DATA_ROOT and RUN_DIR should point to a persistent mount, not ephemeral container storage.
: "${RUN_DIR:?Set RUN_DIR to a directory on persistent storage}"
: "${DATA_ROOT:?Set DATA_ROOT to the directory containing train.jsonl and val.jsonl}"
uv run --no-sync python -m scripts.train \
  --device cuda --size "${SIZE:-tiny}" --precision "${PRECISION:-bf16}" \
  --norm "${NORM:-torch}" --batch-size "${BATCH_SIZE:-64}" \
  --grad-accum "${GRAD_ACCUM:-1}" --steps "${STEPS:-2000}" \
  --workers "${WORKERS:-4}" --data "$DATA_ROOT/train.jsonl" --val-data "$DATA_ROOT/val.jsonl" \
  --output-dir "$RUN_DIR" --log-every 10 --eval-every 100 --save-every 100 "$@"

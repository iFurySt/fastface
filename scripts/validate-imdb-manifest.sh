#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data}"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${ENV_NAME:-faceattr}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/manifests/imdb_clean.jsonl}"
MIN_MANIFEST_ROWS="${MIN_MANIFEST_ROWS:-250000}"
OUTPUT="${OUTPUT:-${DATA_ROOT}/manifests/imdb_clean.validation.json}"
CHECK_READABLE="${CHECK_READABLE:-0}"
MAX_READABLE="${MAX_READABLE:-0}"

cd "${PROJECT_DIR}"

ARGS=(
  --manifest "${MANIFEST}"
  --require-min-rows "${MIN_MANIFEST_ROWS}"
  --require-split train
  --require-split val
  --require-dataset imdb-clean
  --check-images
  --output "${OUTPUT}"
)

if [[ "${CHECK_READABLE}" == "1" ]]; then
  ARGS+=(--check-readable --max-readable "${MAX_READABLE}")
fi

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m fastface.data.validate_manifest "${ARGS[@]}"

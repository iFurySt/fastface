#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data}"
COMPARISON_DIR="${COMPARISON_DIR:-${PROJECT_DIR}/outputs/analysis/gender-comparison-current}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/analysis/manual-public-gender-review-current}"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${ENV_NAME:-faceattr}"

cd "${PROJECT_DIR}"

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m fastface.evaluation.build_manual_gender_review \
  --comparison-dir "${COMPARISON_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --manifest "${DATA_ROOT}/manifests/fairface.jsonl" \
  --manifest "${DATA_ROOT}/manifests/utkface.jsonl" \
  --manifest "${DATA_ROOT}/manifests/imdb_clean.jsonl"

echo "manual_review_output=${OUTPUT_DIR}"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data}"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${ENV_NAME:-faceattr}"

cd "${PROJECT_DIR}"

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m fastface.data.report_dataset_status \
  --data-root "${DATA_ROOT}" \
  --output "${DATA_ROOT}/manifests/dataset-status.json"

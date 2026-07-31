#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${ENV_NAME:-faceattr}"
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/runs/mobilenetv3_real_fairface_utkface}"
INPUT_SIZE="${INPUT_SIZE:-128}"
MANIFEST_ARGS=()

if [[ -n "${MANIFESTS:-}" ]]; then
  for manifest in ${MANIFESTS}; do
    MANIFEST_ARGS+=(--manifest "${manifest}")
  done
fi

cd "${PROJECT_DIR}"

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m fastface.evaluation.evaluate_checkpoint \
  --checkpoint "${RUN_DIR}/best.pt" \
  "${MANIFEST_ARGS[@]}" \
  --input-size "${INPUT_SIZE}" \
  --output "${RUN_DIR}/evaluation_val.json"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${ENV_NAME:-faceattr}"
CONFIG="${CONFIG:?CONFIG is required}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
INPUT_SIZE="${INPUT_SIZE:-}"
FINALIZE_AFTER_TRAIN="${FINALIZE_AFTER_TRAIN:-1}"

cd "${PROJECT_DIR}"
mkdir -p "${RUN_DIR}"

if [[ -z "${INPUT_SIZE}" ]]; then
  INPUT_SIZE="$("${CONDA_BIN}" run -n "${ENV_NAME}" python -c 'import sys, yaml; config = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8")); print(config.get("data", {}).get("input_size", 128))' "${CONFIG}")"
fi

"${CONDA_BIN}" run -n "${ENV_NAME}" torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  -m fastface.training.train_age_gender \
  --config "${CONFIG}" \
  --output-dir "${RUN_DIR}"

if [[ "${FINALIZE_AFTER_TRAIN}" == "1" ]]; then
  RUN_DIR="${RUN_DIR}" INPUT_SIZE="${INPUT_SIZE}" bash scripts/finalize-model-run.sh
fi

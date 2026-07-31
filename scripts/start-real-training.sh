#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${ENV_NAME:-faceattr}"
CONFIG="${CONFIG:-${PROJECT_DIR}/configs/train/mobilenetv3_real.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/runs/mobilenetv3_real_fairface_utkface}"

cd "${PROJECT_DIR}"
mkdir -p "${RUN_DIR}"

exec "${CONDA_BIN}" run -n "${ENV_NAME}" torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  -m fastface.training.train_age_gender \
  --config "${CONFIG}" \
  --output-dir "${RUN_DIR}"

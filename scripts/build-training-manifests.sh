#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data}"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${ENV_NAME:-faceattr}"

cd "${PROJECT_DIR}"

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m fastface.data.build_manifest \
  --dataset fairface \
  --raw-dir "${DATA_ROOT}/raw/fairface" \
  --out "${DATA_ROOT}/manifests/fairface.jsonl"

if [[ -d "${DATA_ROOT}/raw/utkface" ]]; then
  "${CONDA_BIN}" run -n "${ENV_NAME}" python -m fastface.data.build_manifest \
    --dataset utkface \
    --raw-dir "${DATA_ROOT}/raw/utkface" \
    --out "${DATA_ROOT}/manifests/utkface.jsonl"
fi

if [[ -s "${DATA_ROOT}/raw/lagenda/lagenda_annotations.zip" && -d "${DATA_ROOT}/raw/lagenda/images" ]]; then
  "${CONDA_BIN}" run -n "${ENV_NAME}" python -m fastface.data.build_manifest \
    --dataset lagenda \
    --annotation "${DATA_ROOT}/raw/lagenda/lagenda_annotations.zip" \
    --images-dir "${DATA_ROOT}/raw/lagenda/images" \
    --out "${DATA_ROOT}/manifests/lagenda.jsonl"
fi

if [[ -s "${DATA_ROOT}/raw/lagenda_hf_uaebn/lagenda_annotation.csv" && -d "${DATA_ROOT}/raw/lagenda_hf_uaebn/lag_benchmark" ]]; then
  "${CONDA_BIN}" run -n "${ENV_NAME}" python -m fastface.data.build_manifest \
    --dataset lagenda \
    --annotation "${DATA_ROOT}/raw/lagenda_hf_uaebn/lagenda_annotation.csv" \
    --images-dir "${DATA_ROOT}/raw/lagenda_hf_uaebn" \
    --out "${DATA_ROOT}/manifests/lagenda_hf_uaebn.jsonl"
fi

if [[ -s "${DATA_ROOT}/raw/imdb_clean/mivolo_imdb_annotations.zip" && -d "${DATA_ROOT}/raw/imdb_clean/images" ]]; then
  "${CONDA_BIN}" run -n "${ENV_NAME}" python -m fastface.data.build_manifest \
    --dataset imdb-clean \
    --annotation "${DATA_ROOT}/raw/imdb_clean/mivolo_imdb_annotations.zip" \
    --images-dir "${DATA_ROOT}/raw/imdb_clean/images" \
    --out "${DATA_ROOT}/manifests/imdb_clean.jsonl"
fi

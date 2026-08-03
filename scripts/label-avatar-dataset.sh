#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATASET_ROOT="${DATASET_ROOT:-${HOME}/datasets/avatar_merged_20260730}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/avatar-labeling}"
PYTHON_BIN="${PYTHON_BIN:-python}"

FASTFACE_MODEL="${FASTFACE_MODEL:-${PROJECT_DIR}/outputs/hf/fastface-v0.1.0-detector-stage/models/fastface-large-128/model_fp32.onnx}"
PUBLIC_FAIRFACE_ONNX="${PUBLIC_FAIRFACE_ONNX:-${PROJECT_DIR}/outputs/models/public/fairface.onnx}"
DETECTOR_ONNX="${DETECTOR_ONNX:-${PROJECT_DIR}/outputs/hf/fastface-v0.1.0-detector-stage/models/fastfacedetector-retinaface-mnetv1-960/fastfacedetector_retinaface_mobilenetv1_050_whole960_epoch34.onnx}"

cd "${PROJECT_DIR}"

PYTHONPATH="${PROJECT_DIR}/packages${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m fastface.evaluation.label_avatar_dataset \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --fastface-model "${FASTFACE_MODEL}" \
    --public-fairface-onnx "${PUBLIC_FAIRFACE_ONNX}" \
    --detector-onnx "${DETECTOR_ONNX}" \
    "$@"

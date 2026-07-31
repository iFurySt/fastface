#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${ENV_NAME:-faceattr}"
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/runs/mobilenetv3_real_fairface_utkface}"
INPUT_SIZE="${INPUT_SIZE:-}"
MANIFEST_ARGS=()
FACE_CROP_MARGIN="${FACE_CROP_MARGIN:-}"

cd "${PROJECT_DIR}"

if [[ -z "${INPUT_SIZE}" ]]; then
  INPUT_SIZE="$("${CONDA_BIN}" run -n "${ENV_NAME}" python -c 'import sys, torch; checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False); print(checkpoint.get("config", {}).get("data", {}).get("input_size", 128))' "${RUN_DIR}/best.pt")"
fi

if [[ -z "${FACE_CROP_MARGIN}" ]]; then
  FACE_CROP_MARGIN="$("${CONDA_BIN}" run -n "${ENV_NAME}" python -c 'import sys, torch; checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False); print(checkpoint.get("config", {}).get("data", {}).get("face_crop_margin", 0.0))' "${RUN_DIR}/best.pt")"
  FACE_CROP_MARGIN="${FACE_CROP_MARGIN:-0.0}"
fi

if [[ -n "${MANIFESTS:-}" ]]; then
  for manifest in ${MANIFESTS}; do
    MANIFEST_ARGS+=(--manifest "${manifest}")
  done
else
  while IFS= read -r manifest; do
    [[ -z "${manifest}" ]] && continue
    MANIFEST_ARGS+=(--manifest "${manifest}")
  done < <("${CONDA_BIN}" run -n "${ENV_NAME}" python -c 'import sys, torch; checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False); data = checkpoint.get("config", {}).get("data", {}); manifests = data.get("calibration_manifests") or data.get("eval_manifests") or data.get("manifests", []); [print(manifest) for manifest in manifests]' "${RUN_DIR}/best.pt")
fi

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m fastface.export.export_onnx \
  --checkpoint "${RUN_DIR}/best.pt" \
  --output "${RUN_DIR}/model_fp32.onnx" \
  --input-size "${INPUT_SIZE}"

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m fastface.export.quantize_static_onnx \
  --model "${RUN_DIR}/model_fp32.onnx" \
  --output "${RUN_DIR}/model_int8_static.onnx" \
  "${MANIFEST_ARGS[@]}" \
  --input-size "${INPUT_SIZE}" \
  --face-crop-margin "${FACE_CROP_MARGIN}"

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m fastface.export.benchmark_onnx \
  --model "${RUN_DIR}/model_fp32.onnx" \
  --output "${RUN_DIR}/benchmark_fp32_cpu.json" \
  --input-size "${INPUT_SIZE}"

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m fastface.export.benchmark_onnx \
  --model "${RUN_DIR}/model_int8_static.onnx" \
  --output "${RUN_DIR}/benchmark_int8_static_cpu.json" \
  --input-size "${INPUT_SIZE}"

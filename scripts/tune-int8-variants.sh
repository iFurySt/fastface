#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${ENV_NAME:-faceattr}"
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/runs/mobilenetv3_small112_real_fairface_utkface}"
INPUT_SIZE="${INPUT_SIZE:-112}"
THREADS="${THREADS:-1 2 4 8 16 28 56 112}"
BATCH_SIZES="${BATCH_SIZES:-1 8 32 128}"
WARMUP="${WARMUP:-10}"
ITERATIONS="${ITERATIONS:-100}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-1024}"
CALIBRATION_BATCH_SIZE="${CALIBRATION_BATCH_SIZE:-32}"
PREPROCESS_MODEL="${PREPROCESS_MODEL:-0}"
TUNING_DIR_NAME="${TUNING_DIR_NAME:-int8-tuning}"

cd "${PROJECT_DIR}"

OUT_DIR="${RUN_DIR}/${TUNING_DIR_NAME}"
mkdir -p "${OUT_DIR}/cpu-thread-sweep" "${OUT_DIR}/logs"

FACE_CROP_MARGIN="$("${CONDA_BIN}" run -n "${ENV_NAME}" python -c 'import sys, torch; checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False); print(checkpoint.get("config", {}).get("data", {}).get("face_crop_margin", 0.0))' "${RUN_DIR}/best.pt")"
FACE_CROP_MARGIN="${FACE_CROP_MARGIN:-0.0}"

MANIFEST_ARGS=()
while IFS= read -r manifest; do
  [[ -z "${manifest}" ]] && continue
  MANIFEST_ARGS+=(--manifest "${manifest}")
done < <("${CONDA_BIN}" run -n "${ENV_NAME}" python -c 'import sys, torch; checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False); [print(manifest) for manifest in checkpoint.get("config", {}).get("data", {}).get("manifests", [])]' "${RUN_DIR}/best.pt")

SOURCE_MODEL="${RUN_DIR}/model_fp32.onnx"
if [[ "${PREPROCESS_MODEL}" == "1" ]]; then
  SOURCE_MODEL="${OUT_DIR}/model_fp32.preprocessed.onnx"
  "${CONDA_BIN}" run -n "${ENV_NAME}" python -m fastface.export.preprocess_quant_onnx \
    --model "${RUN_DIR}/model_fp32.onnx" \
    --output "${SOURCE_MODEL}"
fi

run_variant() {
  local name="$1"
  shift
  local model_path="${OUT_DIR}/${name}.onnx"
  local quant_log="${OUT_DIR}/logs/${name}.quantize.log"
  echo "quantize ${name}"
  if ! "${CONDA_BIN}" run -n "${ENV_NAME}" python -m fastface.export.quantize_static_onnx \
    --model "${SOURCE_MODEL}" \
    --output "${model_path}" \
    "${MANIFEST_ARGS[@]}" \
    --input-size "${INPUT_SIZE}" \
    --samples "${CALIBRATION_SAMPLES}" \
    --batch-size "${CALIBRATION_BATCH_SIZE}" \
    --face-crop-margin "${FACE_CROP_MARGIN}" \
    "$@" >"${quant_log}" 2>&1; then
    echo "quantize failed: ${name}; see ${quant_log}" >&2
    return 0
  fi

  for threads in ${THREADS}; do
    echo "benchmark ${name} threads=${threads}"
    "${CONDA_BIN}" run -n "${ENV_NAME}" python -m fastface.export.benchmark_onnx \
      --model "${model_path}" \
      --output "${OUT_DIR}/cpu-thread-sweep/${name}_threads${threads}.json" \
      --input-size "${INPUT_SIZE}" \
      --batch-sizes ${BATCH_SIZES} \
      --warmup "${WARMUP}" \
      --iterations "${ITERATIONS}" \
      --intra-op-num-threads "${threads}" \
      --inter-op-num-threads 1 \
      --execution-mode sequential >/dev/null
  done
}

run_variant qdq_u8s8_pc --quant-format QDQ --activation-type QUInt8 --weight-type QInt8 --per-channel
run_variant qdq_u8s8_tensor --quant-format QDQ --activation-type QUInt8 --weight-type QInt8 --no-per-channel
run_variant qoperator_u8s8_pc --quant-format QOperator --activation-type QUInt8 --weight-type QInt8 --per-channel
run_variant qoperator_u8s8_tensor --quant-format QOperator --activation-type QUInt8 --weight-type QInt8 --no-per-channel
run_variant qdq_s8s8_pc --quant-format QDQ --activation-type QInt8 --weight-type QInt8 --per-channel
run_variant qoperator_s8s8_pc --quant-format QOperator --activation-type QInt8 --weight-type QInt8 --per-channel

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m fastface.export.summarize_thread_sweep \
  --run-dir "${OUT_DIR}"

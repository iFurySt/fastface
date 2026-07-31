#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${ENV_NAME:-faceattr}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
INPUT_SIZE="${INPUT_SIZE:-}"
THREADS="${THREADS:-1 2 4 8 16 28 56 112}"
BATCH_SIZES="${BATCH_SIZES:-1 8 32 128}"
WARMUP="${WARMUP:-10}"
ITERATIONS="${ITERATIONS:-100}"
FINALIZE_THREAD_SWEEP="${FINALIZE_THREAD_SWEEP:-1}"

cd "${PROJECT_DIR}"

if [[ ! -s "${RUN_DIR}/best.pt" ]]; then
  echo "missing checkpoint: ${RUN_DIR}/best.pt" >&2
  exit 1
fi

if [[ -z "${INPUT_SIZE}" ]]; then
  INPUT_SIZE="$("${CONDA_BIN}" run -n "${ENV_NAME}" python -c 'import sys, torch; checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False); print(checkpoint.get("config", {}).get("data", {}).get("input_size", 128))' "${RUN_DIR}/best.pt")"
fi

RUN_DIR="${RUN_DIR}" INPUT_SIZE="${INPUT_SIZE}" bash scripts/evaluate-checkpoint.sh
RUN_DIR="${RUN_DIR}" INPUT_SIZE="${INPUT_SIZE}" bash scripts/export-and-benchmark.sh

if [[ "${FINALIZE_THREAD_SWEEP}" != "1" ]]; then
  echo "skip thread sweep because FINALIZE_THREAD_SWEEP=${FINALIZE_THREAD_SWEEP}"
  bash scripts/generate-model-card.sh "${RUN_DIR}"
  exit 0
fi

OUT_DIR="${RUN_DIR}/cpu-thread-sweep"
mkdir -p "${OUT_DIR}"
for model_name in model_fp32 model_int8_static; do
  model_path="${RUN_DIR}/${model_name}.onnx"
  if [[ ! -s "${model_path}" ]]; then
    echo "skip missing model: ${model_path}" >&2
    continue
  fi
  for threads in ${THREADS}; do
    output_path="${OUT_DIR}/${model_name}_threads${threads}.json"
    if [[ -s "${output_path}" ]]; then
      echo "skip existing benchmark ${model_name} threads=${threads}"
      continue
    fi
    echo "benchmark ${model_name} threads=${threads}"
    "${CONDA_BIN}" run -n "${ENV_NAME}" python -m fastface.export.benchmark_onnx \
      --model "${model_path}" \
      --output "${output_path}" \
      --input-size "${INPUT_SIZE}" \
      --batch-sizes ${BATCH_SIZES} \
      --warmup "${WARMUP}" \
      --iterations "${ITERATIONS}" \
      --intra-op-num-threads "${threads}" \
      --inter-op-num-threads 1 \
      --execution-mode sequential >/dev/null
  done
done

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m fastface.export.summarize_thread_sweep \
  --run-dir "${RUN_DIR}"
bash scripts/generate-model-card.sh "${RUN_DIR}"

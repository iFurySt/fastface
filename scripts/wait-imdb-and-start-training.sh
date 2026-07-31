#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${ENV_NAME:-faceattr}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data}"
CONFIG="${CONFIG:-${PROJECT_DIR}/configs/train/mobilenetv3_small_112_imdb_real.yaml}"
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/runs/mobilenetv3_small112_imdb_facecrop_real_fairface_utkface}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/manifests/imdb_clean.jsonl}"
MIN_MANIFEST_ROWS="${MIN_MANIFEST_ROWS:-250000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
POLL_SECONDS="${POLL_SECONDS:-300}"
FINALIZE_AFTER_TRAIN="${FINALIZE_AFTER_TRAIN:-1}"
INPUT_SIZE="${INPUT_SIZE:-112}"
WAIT_FOR_IDLE_TRAINING="${WAIT_FOR_IDLE_TRAINING:-1}"
TRAINING_IDLE_POLL_SECONDS="${TRAINING_IDLE_POLL_SECONDS:-300}"
IDLE_PROCESS_PATTERN="${IDLE_PROCESS_PATTERN:-torchrun|fastface.training.train_age_gender|finalize-model-run|evaluate_checkpoint|export_onnx|quantize_static|benchmark_onnx|summarize_thread_sweep|generate_model_card}"

cd "${PROJECT_DIR}"
mkdir -p "${RUN_DIR}"

manifest_rows() {
  if [[ ! -s "${MANIFEST}" ]]; then
    echo 0
    return 0
  fi
  wc -l <"${MANIFEST}"
}

blocking_processes() {
  { pgrep -af "${IDLE_PROCESS_PATTERN}" 2>/dev/null || true; } |
    awk -v run_dir="${RUN_DIR}" '
      !/grep|pgrep|wait-imdb-and-start-training/ && index($0, run_dir) == 0 {print}
    '
}

while true; do
  rows="$(manifest_rows)"
  if [[ "${rows}" -ge "${MIN_MANIFEST_ROWS}" ]]; then
    echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') imdb manifest ready: ${rows} rows"
    echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') validating imdb manifest"
    MANIFEST="${MANIFEST}" MIN_MANIFEST_ROWS="${MIN_MANIFEST_ROWS}" bash scripts/validate-imdb-manifest.sh
    break
  fi
  if [[ -f "${RUN_DIR}/best.pt" ]]; then
    echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') run already has best.pt: ${RUN_DIR}"
    exit 0
  fi
  if [[ "${rows}" -gt 0 ]]; then
    echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') waiting for ${MANIFEST}: ${rows}/${MIN_MANIFEST_ROWS} rows"
  else
    echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') waiting for ${MANIFEST}"
  fi
  sleep "${POLL_SECONDS}"
done

if [[ "${WAIT_FOR_IDLE_TRAINING}" == "1" ]]; then
  while true; do
    active_processes="$(blocking_processes)"
    if [[ -z "${active_processes}" ]]; then
      break
    fi
    echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') waiting for existing training/finalization processes before starting IMDB run"
    echo "${active_processes}"
    sleep "${TRAINING_IDLE_POLL_SECONDS}"
  done
fi

echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') starting training: ${RUN_DIR}"
CONFIG="${CONFIG}" RUN_DIR="${RUN_DIR}" NPROC_PER_NODE="${NPROC_PER_NODE}" bash scripts/start-real-training.sh

if [[ "${FINALIZE_AFTER_TRAIN}" == "1" ]]; then
  echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') finalizing run: ${RUN_DIR}"
  RUN_DIR="${RUN_DIR}" INPUT_SIZE="${INPUT_SIZE}" bash scripts/finalize-model-run.sh
fi

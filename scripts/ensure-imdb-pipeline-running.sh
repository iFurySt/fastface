#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data}"
PREPARE_RUN_DIR="${PREPARE_RUN_DIR:-${PROJECT_DIR}/runs/imdb_clean_prepare}"
TRAIN_RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/runs/mobilenetv3_small112_imdb_facecrop_real_fairface_utkface}"
CONFIG="${CONFIG:-${PROJECT_DIR}/configs/train/mobilenetv3_small_112_imdb_real.yaml}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/manifests/imdb_clean.jsonl}"
MIN_MANIFEST_ROWS="${MIN_MANIFEST_ROWS:-250000}"
DOWNLOAD_TOOL="${DOWNLOAD_TOOL:-aria2c}"
DOWNLOAD_JOBS="${DOWNLOAD_JOBS:-10}"
ARIA2_CONNECTIONS="${ARIA2_CONNECTIONS:-8}"
ARIA2_SPLIT="${ARIA2_SPLIT:-8}"
EXTRACT_JOBS="${EXTRACT_JOBS:-4}"
PIPELINE_EXTRACT="${PIPELINE_EXTRACT:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
POLL_SECONDS="${POLL_SECONDS:-300}"
WAIT_FOR_IDLE_TRAINING="${WAIT_FOR_IDLE_TRAINING:-1}"
TRAINING_IDLE_POLL_SECONDS="${TRAINING_IDLE_POLL_SECONDS:-300}"

cd "${PROJECT_DIR}"
mkdir -p "${PREPARE_RUN_DIR}" "${TRAIN_RUN_DIR}"

pid_is_running() {
  local pid_file="$1"
  local pattern="$2"
  local pid
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  [[ -n "${pid}" ]] && ps -p "${pid}" -o cmd= 2>/dev/null | grep -q "${pattern}"
}

manifest_rows() {
  if [[ ! -s "${MANIFEST}" ]]; then
    echo 0
    return 0
  fi
  wc -l <"${MANIFEST}"
}

rows="$(manifest_rows)"
if [[ "${rows}" -lt "${MIN_MANIFEST_ROWS}" ]]; then
  if pid_is_running "${PREPARE_RUN_DIR}/prepare.pid" "prepare-imdb-clean-images"; then
    echo "imdb preparation already running: $(cat "${PREPARE_RUN_DIR}/prepare.pid")"
  else
    echo "starting imdb preparation: ${rows}/${MIN_MANIFEST_ROWS} manifest rows"
    setsid env PROJECT_DIR="${PROJECT_DIR}" DATA_ROOT="${DATA_ROOT}" \
      DOWNLOAD_TOOL="${DOWNLOAD_TOOL}" DOWNLOAD_JOBS="${DOWNLOAD_JOBS}" \
      ARIA2_CONNECTIONS="${ARIA2_CONNECTIONS}" ARIA2_SPLIT="${ARIA2_SPLIT}" \
      EXTRACT_JOBS="${EXTRACT_JOBS}" PIPELINE_EXTRACT="${PIPELINE_EXTRACT}" \
      bash scripts/prepare-imdb-clean-images.sh \
      > "${PREPARE_RUN_DIR}/prepare.log" 2>&1 < /dev/null &
    echo $! > "${PREPARE_RUN_DIR}/prepare.pid"
  fi
else
  echo "imdb manifest ready: ${rows} rows"
fi

if [[ -s "${TRAIN_RUN_DIR}/model_card.md" ]]; then
  echo "imdb run finalized: ${TRAIN_RUN_DIR}"
elif pid_is_running "${TRAIN_RUN_DIR}/launcher.pid" "wait-imdb-and-start-training"; then
  echo "imdb watcher already running"
elif pgrep -af "${TRAIN_RUN_DIR}" | grep -v grep >/dev/null; then
  echo "imdb watcher/training/finalizer already running"
else
  echo "starting imdb watcher"
  setsid env PROJECT_DIR="${PROJECT_DIR}" DATA_ROOT="${DATA_ROOT}" \
    CONFIG="${CONFIG}" RUN_DIR="${TRAIN_RUN_DIR}" MANIFEST="${MANIFEST}" \
    MIN_MANIFEST_ROWS="${MIN_MANIFEST_ROWS}" \
    NPROC_PER_NODE="${NPROC_PER_NODE}" POLL_SECONDS="${POLL_SECONDS}" \
    WAIT_FOR_IDLE_TRAINING="${WAIT_FOR_IDLE_TRAINING}" \
    TRAINING_IDLE_POLL_SECONDS="${TRAINING_IDLE_POLL_SECONDS}" \
    FINALIZE_AFTER_TRAIN=1 INPUT_SIZE=112 \
    bash scripts/wait-imdb-and-start-training.sh \
    >> "${TRAIN_RUN_DIR}/train.log" 2>&1 < /dev/null &
  echo $! > "${TRAIN_RUN_DIR}/launcher.pid"
fi

echo "prepare_pid=$(cat "${PREPARE_RUN_DIR}/prepare.pid" 2>/dev/null || true)"
echo "launcher_pid=$(cat "${TRAIN_RUN_DIR}/launcher.pid" 2>/dev/null || true)"

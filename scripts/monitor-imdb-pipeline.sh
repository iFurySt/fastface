#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/runs/imdb_pipeline_monitor}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-${PROJECT_DIR}/runs/mobilenetv3_small112_imdb_facecrop_real_fairface_utkface}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
LOCK_FILE="${RUN_DIR}/monitor.lock"
PID_FILE="${RUN_DIR}/monitor.pid"
LOG_FILE="${RUN_DIR}/monitor.log"
LATEST_STATUS="${RUN_DIR}/status-latest.txt"
HISTORY_LOG="${RUN_DIR}/status-history.log"

mkdir -p "${RUN_DIR}"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "another IMDB pipeline monitor is already running: ${LOCK_FILE}" >&2
  exit 1
fi

echo "$$" > "${PID_FILE}"
cd "${PROJECT_DIR}"

while true; do
  timestamp="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  {
    echo "===== ${timestamp} ensure ====="
    bash scripts/ensure-imdb-pipeline-running.sh
    echo "===== ${timestamp} status ====="
    tmp_status="${LATEST_STATUS}.tmp"
    bash scripts/imdb-pipeline-status.sh > "${tmp_status}"
    mv "${tmp_status}" "${LATEST_STATUS}"
    cat "${LATEST_STATUS}"
  } >> "${LOG_FILE}" 2>&1

  {
    echo "===== ${timestamp} ====="
    cat "${LATEST_STATUS}" 2>/dev/null || true
  } >> "${HISTORY_LOG}" 2>&1

  if [[ -s "${TRAIN_RUN_DIR}/model_card.md" ]]; then
    echo "===== $(date -u +'%Y-%m-%dT%H:%M:%SZ') finalized: ${TRAIN_RUN_DIR} =====" >> "${LOG_FILE}"
    exit 0
  fi

  sleep "${INTERVAL_SECONDS}"
done

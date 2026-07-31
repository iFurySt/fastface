#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data}"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${ENV_NAME:-faceattr}"
PREPARE_RUN_DIR="${PREPARE_RUN_DIR:-${PROJECT_DIR}/runs/imdb_clean_prepare}"
TRAIN_RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/runs/mobilenetv3_small112_imdb_facecrop_real_fairface_utkface}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/manifests/imdb_clean.jsonl}"
MIN_MANIFEST_ROWS="${MIN_MANIFEST_ROWS:-250000}"

manifest_rows() {
  if [[ ! -s "${MANIFEST}" ]]; then
    echo 0
    return 0
  fi
  wc -l <"${MANIFEST}"
}

pid_status() {
  local label="$1"
  local pid_file="$2"
  local pid
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  if [[ -z "${pid}" ]]; then
    echo "${label}_pid="
    echo "${label}_running=0"
    return 0
  fi
  echo "${label}_pid=${pid}"
  if ps -p "${pid}" >/dev/null 2>&1; then
    echo "${label}_running=1"
  else
    echo "${label}_running=0"
  fi
}

process_count() {
  local pattern="$1"
  { pgrep -af "${pattern}" 2>/dev/null || true; } | awk '!/grep|pgrep/ {count++} END {print count + 0}'
}

process_count_for_run() {
  local pattern="$1"
  local run_dir="$2"
  { pgrep -af "${pattern}" 2>/dev/null || true; } | awk -v run_dir="${run_dir}" '!/grep|pgrep/ && index($0, run_dir) > 0 {count++} END {print count + 0}'
}

cd "${PROJECT_DIR}"

echo "manifest=${MANIFEST}"
echo "manifest_rows=$(manifest_rows)"
echo "min_manifest_rows=${MIN_MANIFEST_ROWS}"
pid_status "prepare" "${PREPARE_RUN_DIR}/prepare.pid"
pid_status "launcher" "${TRAIN_RUN_DIR}/launcher.pid"
echo "aria2_imdb_processes=$(process_count 'aria2c.*imdb_')"
echo "tar_extract_processes=$(process_count 'tar -xf .*/imdb_[0-9]+.tar')"
echo "imdb_training_processes=$(process_count_for_run 'torchrun|fastface.training.train_age_gender' "${TRAIN_RUN_DIR}")"
echo "all_training_processes=$(process_count 'torchrun|fastface.training.train_age_gender')"
echo "finalizer_processes=$(process_count 'finalize-model-run')"
echo "---disk---"
df -h "${DATA_ROOT}" "${PROJECT_DIR}" 2>/dev/null || true
echo "---dataset-status---"
bash scripts/report-dataset-status.sh
echo "---train-log-tail---"
tail -n 20 "${TRAIN_RUN_DIR}/train.log" 2>/dev/null || true

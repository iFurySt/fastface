#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data}"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${ENV_NAME:-faceattr}"
DOWNLOAD_JOBS="${DOWNLOAD_JOBS:-10}"
EXTRACT_JOBS="${EXTRACT_JOBS:-4}"
DOWNLOAD_TOOL="${DOWNLOAD_TOOL:-aria2c}"
ARIA2_CONNECTIONS="${ARIA2_CONNECTIONS:-8}"
ARIA2_SPLIT="${ARIA2_SPLIT:-8}"
PIPELINE_EXTRACT="${PIPELINE_EXTRACT:-1}"

RAW_DIR="${DATA_ROOT}/raw/imdb_clean"
TAR_DIR="${RAW_DIR}/tars"
IMAGES_DIR="${RAW_DIR}/images"
ANNOTATION="${RAW_DIR}/mivolo_imdb_annotations.zip"
MANIFEST="${DATA_ROOT}/manifests/imdb_clean.jsonl"
LOG_DIR="${PROJECT_DIR}/logs/downloads/imdb_clean"
LOCK_FILE="${RAW_DIR}/.prepare-imdb-clean.lock"

mkdir -p "${TAR_DIR}" "${IMAGES_DIR}" "${LOG_DIR}" "$(dirname "${MANIFEST}")"

if [[ ! -s "${ANNOTATION}" ]]; then
  echo "missing annotation archive: ${ANNOTATION}" >&2
  exit 1
fi

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "another IMDB-clean preparation is already running: ${LOCK_FILE}" >&2
  exit 1
fi

expected_tar_bytes() {
  case "$1" in
    0) echo 28708782080 ;;
    1) echo 27734599680 ;;
    2) echo 29475174400 ;;
    3) echo 30881392640 ;;
    4) echo 27863429120 ;;
    5) echo 30502092800 ;;
    6) echo 28542679040 ;;
    7) echo 28599234560 ;;
    8) echo 27647651840 ;;
    9) echo 25642557440 ;;
    *) return 1 ;;
  esac
}

download_part() {
  local part="$1"
  local url="https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_${part}.tar"
  local out="${TAR_DIR}/imdb_${part}.tar"
  local log="${LOG_DIR}/download_imdb_${part}.log"
  echo "download imdb_${part}.tar"
  if [[ "${DOWNLOAD_TOOL}" == "aria2c" ]] && command -v aria2c >/dev/null 2>&1; then
    aria2c \
      --continue=true \
      --max-connection-per-server="${ARIA2_CONNECTIONS}" \
      --split="${ARIA2_SPLIT}" \
      --min-split-size=4M \
      --max-tries=0 \
      --retry-wait=30 \
      --timeout=60 \
      --connect-timeout=60 \
      --file-allocation=none \
      --allow-overwrite=true \
      --auto-file-renaming=false \
      --dir="${TAR_DIR}" \
      --out="imdb_${part}.tar" \
      "${url}" >"${log}" 2>&1
  else
    wget -c --tries=0 --timeout=60 --read-timeout=60 --waitretry=30 \
      -O "${out}" "${url}" >"${log}" 2>&1
  fi
}

validate_tar_part() {
  local part="$1"
  local tar_path="${TAR_DIR}/imdb_${part}.tar"
  local control_path="${tar_path}.aria2"
  local expected
  local actual
  expected="$(expected_tar_bytes "${part}")"
  if [[ -s "${control_path}" ]]; then
    echo "download still incomplete: ${control_path}" >&2
    return 1
  fi
  if [[ ! -s "${tar_path}" ]]; then
    echo "missing tar: ${tar_path}" >&2
    return 1
  fi
  actual="$(stat -c %s "${tar_path}" 2>/dev/null || stat -f %z "${tar_path}")"
  if [[ "${actual}" -ne "${expected}" ]]; then
    echo "unexpected tar size for imdb_${part}.tar: actual=${actual} expected=${expected}" >&2
    return 1
  fi
}

extract_part() {
  local part="$1"
  local tar_path="${TAR_DIR}/imdb_${part}.tar"
  local marker="${IMAGES_DIR}/.imdb_${part}.extract.done"
  local log="${LOG_DIR}/extract_imdb_${part}.log"
  if [[ -s "${marker}" ]]; then
    echo "skip extracted imdb_${part}.tar"
    return 0
  fi
  validate_tar_part "${part}"
  echo "extract imdb_${part}.tar"
  tar -xf "${tar_path}" -C "${IMAGES_DIR}" >"${log}" 2>&1
  date -u +"%Y-%m-%dT%H:%M:%SZ" >"${marker}"
}

download_and_extract_part() {
  local part="$1"
  download_part "${part}"
  extract_part "${part}"
}

export -f expected_tar_bytes download_part validate_tar_part extract_part download_and_extract_part
export TAR_DIR IMAGES_DIR LOG_DIR DOWNLOAD_TOOL ARIA2_CONNECTIONS ARIA2_SPLIT

if [[ "${PIPELINE_EXTRACT}" == "1" ]]; then
  printf "%s\n" 0 1 2 3 4 5 6 7 8 9 | xargs -n 1 -P "${DOWNLOAD_JOBS}" bash -c 'download_and_extract_part "$0"'
else
  printf "%s\n" 0 1 2 3 4 5 6 7 8 9 | xargs -n 1 -P "${DOWNLOAD_JOBS}" bash -c 'download_part "$0"'
  printf "%s\n" 0 1 2 3 4 5 6 7 8 9 | xargs -n 1 -P "${EXTRACT_JOBS}" bash -c 'extract_part "$0"'
fi

cd "${PROJECT_DIR}"
TMP_MANIFEST="${MANIFEST}.tmp"
rm -f "${TMP_MANIFEST}"
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m fastface.data.build_manifest \
  --dataset imdb-clean \
  --annotation "${ANNOTATION}" \
  --images-dir "${IMAGES_DIR}" \
  --out "${TMP_MANIFEST}"
mv "${TMP_MANIFEST}" "${MANIFEST}"

echo "ready: ${MANIFEST}"

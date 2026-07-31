#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data}"
THIRD_PARTY_DIR="${THIRD_PARTY_DIR:-${PROJECT_DIR}/third_party}"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${ENV_NAME:-faceattr}"

RAW_DIR="${DATA_ROOT}/raw"
MANIFEST_DIR="${DATA_ROOT}/manifests"
LOG_DIR="${PROJECT_DIR}/logs/downloads"

mkdir -p "${RAW_DIR}" "${MANIFEST_DIR}" "${LOG_DIR}" "${THIRD_PARTY_DIR}"
DOWNLOAD_LOG="${MANIFEST_DIR}/downloads.jsonl"

record() {
  local dataset="$1"
  local status="$2"
  local route="$3"
  local note="$4"
  "${CONDA_BIN}" run -n "${ENV_NAME}" python - "$DOWNLOAD_LOG" "$dataset" "$status" "$route" "$note" <<'PY'
import json
import sys
import time

path, dataset, status, route, note = sys.argv[1:]
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps({
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": dataset,
        "status": status,
        "route": route,
        "note": note,
    }, ensure_ascii=True) + "\n")
PY
}

run_py() {
  "${CONDA_BIN}" run -n "${ENV_NAME}" python "$@"
}

run_hf_download() {
  local repo_id="$1"
  local out_dir="$2"
  mkdir -p "${out_dir}"
  HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  "${CONDA_BIN}" run -n "${ENV_NAME}" hf download "${repo_id}" \
    --repo-type dataset \
    --local-dir "${out_dir}"
}

download_gdrive() {
  local file_id="$1"
  local output="$2"
  mkdir -p "$(dirname "${output}")"
  run_py -m gdown "https://drive.google.com/uc?id=${file_id}" -O "${output}" --continue
}

download_fairface_file() {
  local file_id="$1"
  local output="$2"
  local label="$3"
  if [[ -s "${output}" ]]; then
    record "fairface" "present" "FairFace Google Drive" "${output}"
    return 0
  fi
  if download_gdrive "${file_id}" "${output}" 2>&1 | tee "${LOG_DIR}/fairface_${label}.log"; then
    record "fairface" "downloaded" "FairFace Google Drive" "${output}"
  else
    record "fairface" "failed" "FairFace Google Drive" "see ${LOG_DIR}/fairface_${label}.log"
  fi
}

echo "Writing download log to ${DOWNLOAD_LOG}"

if [[ "${DOWNLOAD_FAIRFACE:-1}" == "1" ]]; then
  dataset_dir="${RAW_DIR}/fairface"
  mkdir -p "${dataset_dir}"
  download_fairface_file "1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86" "${dataset_dir}/fairface-img-margin025-trainval.zip" "margin025"
  download_fairface_file "1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH" "${dataset_dir}/fairface_label_train.csv" "label_train"
  download_fairface_file "1wOdja-ezstMEp81tX1a-EYkFebev4h7D" "${dataset_dir}/fairface_label_val.csv" "label_val"
  if [[ "${DOWNLOAD_FAIRFACE_MARGIN125:-0}" == "1" ]]; then
    download_fairface_file "1g7qNOZz9wC7OfOhcPqH1EZ5bk1UFGmlL" "${dataset_dir}/fairface-img-margin125-trainval.zip" "margin125"
  fi
fi

if [[ "${DOWNLOAD_LAGENDA:-1}" == "1" ]]; then
  dataset_dir="${RAW_DIR}/lagenda"
  mkdir -p "${dataset_dir}"
  if [[ ! -s "${dataset_dir}/lagenda_images.zip" ]]; then
    if download_gdrive "1QXO0NlkABPZT6x1_0Uc2i6KAtdcrpTbG" "${dataset_dir}/lagenda_images.zip" 2>&1 | tee "${LOG_DIR}/lagenda_images.log"; then
      record "lagenda_images" "downloaded" "MiVOLO Google Drive" "${dataset_dir}/lagenda_images.zip"
    else
      record "lagenda_images" "failed" "MiVOLO Google Drive" "see ${LOG_DIR}/lagenda_images.log"
    fi
  else
    record "lagenda_images" "present" "MiVOLO Google Drive" "${dataset_dir}/lagenda_images.zip"
  fi
  if [[ ! -s "${dataset_dir}/lagenda_annotations.zip" ]]; then
    if download_gdrive "1mNYjYFb3MuKg-OL1UISoYsKObMUllbJx" "${dataset_dir}/lagenda_annotations.zip" 2>&1 | tee "${LOG_DIR}/lagenda_annotations.log"; then
      record "lagenda_annotations" "downloaded" "MiVOLO Google Drive" "${dataset_dir}/lagenda_annotations.zip"
    else
      record "lagenda_annotations" "failed" "MiVOLO Google Drive" "see ${LOG_DIR}/lagenda_annotations.log"
    fi
  else
    record "lagenda_annotations" "present" "MiVOLO Google Drive" "${dataset_dir}/lagenda_annotations.zip"
  fi
fi

if [[ "${DOWNLOAD_LAGENDA_HF:-0}" == "1" ]]; then
  dataset_dir="${RAW_DIR}/lagenda_hf_uaebn"
  if run_hf_download "uaebn/lagenda" "${dataset_dir}" 2>&1 | tee "${LOG_DIR}/lagenda_hf_uaebn.log"; then
    record "lagenda-hf-uaebn" "downloaded" "huggingface:uaebn/lagenda" "${dataset_dir}"
  else
    record "lagenda-hf-uaebn" "failed" "huggingface:uaebn/lagenda" "see ${LOG_DIR}/lagenda_hf_uaebn.log"
  fi
fi

if [[ "${DOWNLOAD_IMDB_CLEAN:-1}" == "1" ]]; then
  repo_dir="${THIRD_PARTY_DIR}/imdb-clean"
  if [[ -d "${repo_dir}/.git" ]]; then
    git -C "${repo_dir}" pull --ff-only || true
    record "imdb-clean-repo" "present" "https://github.com/yiminglin-ai/imdb-clean" "${repo_dir}"
  else
    if git clone https://github.com/yiminglin-ai/imdb-clean "${repo_dir}" 2>&1 | tee "${LOG_DIR}/imdb_clean_repo.log"; then
      record "imdb-clean-repo" "downloaded" "https://github.com/yiminglin-ai/imdb-clean" "${repo_dir}"
    else
      record "imdb-clean-repo" "failed" "https://github.com/yiminglin-ai/imdb-clean" "see ${LOG_DIR}/imdb_clean_repo.log"
    fi
  fi

  dataset_dir="${RAW_DIR}/imdb_clean"
  mkdir -p "${dataset_dir}"
  if [[ ! -s "${dataset_dir}/mivolo_imdb_annotations.zip" ]]; then
    if download_gdrive "17uEqyU3uQ5trWZ5vRJKzh41yeuDe5hyL" "${dataset_dir}/mivolo_imdb_annotations.zip" 2>&1 | tee "${LOG_DIR}/imdb_clean_annotations.log"; then
      record "imdb-clean-annotations" "downloaded" "MiVOLO Google Drive" "${dataset_dir}/mivolo_imdb_annotations.zip"
    else
      record "imdb-clean-annotations" "failed" "MiVOLO Google Drive" "see ${LOG_DIR}/imdb_clean_annotations.log"
    fi
  else
    record "imdb-clean-annotations" "present" "MiVOLO Google Drive" "${dataset_dir}/mivolo_imdb_annotations.zip"
  fi
fi

if [[ "${DOWNLOAD_UTKFACE:-1}" == "1" ]]; then
  if [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
    dataset_dir="${RAW_DIR}/utkface"
    mkdir -p "${dataset_dir}"
    if run_py -m kaggle datasets download -d jangedoo/utkface-new -p "${dataset_dir}" 2>&1 | tee "${LOG_DIR}/utkface_kaggle.log"; then
      record "utkface" "downloaded" "kaggle:jangedoo/utkface-new" "${dataset_dir}"
    else
      record "utkface" "failed" "kaggle:jangedoo/utkface-new" "see ${LOG_DIR}/utkface_kaggle.log"
    fi
  else
    record "utkface" "gated" "kaggle:jangedoo/utkface-new" "KAGGLE_USERNAME/KAGGLE_KEY not set"
  fi
fi

if [[ "${DOWNLOAD_CELEBA:-1}" == "1" ]]; then
  record "celeba" "gated" "official mirrors vary" "requires source selection and license/access confirmation"
fi

if [[ "${DOWNLOAD_ADIENCE:-1}" == "1" ]]; then
  record "adience" "pending" "official project links" "download route needs scripted confirmation before production use"
fi

echo "Done. Review ${DOWNLOAD_LOG}"

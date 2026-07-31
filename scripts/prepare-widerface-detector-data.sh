#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORK_ROOT="${FASTFACE_WORK_ROOT:-${PROJECT_DIR}}"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${ENV_NAME:-faceattr}"
WIDERFACE_DIR="${WIDERFACE_DIR:-${WORK_ROOT}/data/raw/widerface}"
DOWNLOAD_IMAGES="${DOWNLOAD_IMAGES:-1}"
RETINAFACE_GT_ZIP="${RETINAFACE_GT_ZIP:-${WIDERFACE_DIR}/retinaface_gt_v1.1.zip}"

mkdir -p "${WIDERFACE_DIR}" "${WORK_ROOT}/logs/downloads"

if [[ "${DOWNLOAD_IMAGES}" == "1" ]]; then
  "${CONDA_BIN}" run -n "${ENV_NAME}" python - <<'PY'
from pathlib import Path
import sys

try:
    from huggingface_hub import hf_hub_download
except Exception as exc:
    raise SystemExit(f"huggingface_hub is required to download WIDER FACE images: {exc}") from exc

out_dir = Path(sys.argv[1])
out_dir.mkdir(parents=True, exist_ok=True)
for filename in ["WIDER_train.zip", "WIDER_val.zip"]:
    path = hf_hub_download(
        repo_id="wider_face",
        repo_type="dataset",
        filename=filename,
        local_dir=out_dir,
        local_dir_use_symlinks=False,
    )
    print(path)
PY
fi

if [[ -s "${WIDERFACE_DIR}/WIDER_train.zip" && ! -d "${WIDERFACE_DIR}/train/images" ]]; then
  tmp_dir="${WIDERFACE_DIR}/_extract_train"
  rm -rf "${tmp_dir}"
  mkdir -p "${tmp_dir}"
  unzip -q "${WIDERFACE_DIR}/WIDER_train.zip" -d "${tmp_dir}"
  mkdir -p "${WIDERFACE_DIR}/train"
  if [[ -d "${tmp_dir}/WIDER_train/images" ]]; then
    mv "${tmp_dir}/WIDER_train/images" "${WIDERFACE_DIR}/train/images"
  elif [[ -d "${tmp_dir}/images" ]]; then
    mv "${tmp_dir}/images" "${WIDERFACE_DIR}/train/images"
  else
    echo "Could not find WIDER train images after extraction" >&2
    exit 2
  fi
  rm -rf "${tmp_dir}"
fi

if [[ -s "${RETINAFACE_GT_ZIP}" && ! -s "${WIDERFACE_DIR}/train/label.txt" ]]; then
  tmp_dir="${WIDERFACE_DIR}/_extract_retinaface_gt"
  rm -rf "${tmp_dir}"
  mkdir -p "${tmp_dir}"
  unzip -q "${RETINAFACE_GT_ZIP}" -d "${tmp_dir}"
  label_path="$(find "${tmp_dir}" -type f -name label.txt | head -1)"
  if [[ -z "${label_path}" ]]; then
    echo "Could not find label.txt inside ${RETINAFACE_GT_ZIP}" >&2
    exit 3
  fi
  mkdir -p "${WIDERFACE_DIR}/train"
  cp "${label_path}" "${WIDERFACE_DIR}/train/label.txt"
  rm -rf "${tmp_dir}"
fi

if [[ ! -s "${WIDERFACE_DIR}/train/label.txt" ]]; then
  cat >&2 <<EOF
WIDER FACE images may be staged, but RetinaFace 5-point landmark labels are missing.

Expected:
  ${WIDERFACE_DIR}/train/label.txt

Place the RetinaFace landmark archive at:
  ${RETINAFACE_GT_ZIP}

Then rerun this script. The archive is commonly named retinaface_gt_v1.1.zip
and contains the training label.txt used by RetinaFace implementations.
EOF
  exit 4
fi

if [[ ! -d "${WIDERFACE_DIR}/train/images" ]]; then
  echo "Missing WIDER FACE train images under ${WIDERFACE_DIR}/train/images" >&2
  exit 5
fi

echo "WIDER FACE detector data is ready:"
echo "  ${WIDERFACE_DIR}/train/images"
echo "  ${WIDERFACE_DIR}/train/label.txt"

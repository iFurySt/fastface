#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORK_ROOT="${FASTFACE_WORK_ROOT:-${PROJECT_DIR}}"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${ENV_NAME:-faceattr}"

DETECTOR_REPO_URL="${DETECTOR_REPO_URL:-https://github.com/yakhyo/retinaface-pytorch.git}"
DETECTOR_REPO_DIR="${DETECTOR_REPO_DIR:-${WORK_ROOT}/third_party/retinaface-pytorch}"
WIDERFACE_DIR="${WIDERFACE_DIR:-${WORK_ROOT}/data/raw/widerface}"
RUN_NAME="${RUN_NAME:-fastfacedetector_retinaface_mobilenetv2_widerface}"
RUN_DIR="${RUN_DIR:-${WORK_ROOT}/runs/${RUN_NAME}}"
NETWORK="${NETWORK:-mobilenetv2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
EXPORT_ONNX="${EXPORT_ONNX:-1}"
RESUME="${RESUME:-0}"

if [[ ! -d "${DETECTOR_REPO_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${DETECTOR_REPO_DIR}")"
  git clone "${DETECTOR_REPO_URL}" "${DETECTOR_REPO_DIR}"
else
  git -C "${DETECTOR_REPO_DIR}" pull --ff-only
fi

if [[ ! -s "${WIDERFACE_DIR}/train/label.txt" || ! -d "${WIDERFACE_DIR}/train/images" ]]; then
  cat >&2 <<EOF
Missing WIDER FACE RetinaFace training data.

Expected:
  ${WIDERFACE_DIR}/train/label.txt
  ${WIDERFACE_DIR}/train/images/

Prepare the organized RetinaFace WIDER FACE dataset before launching training.
EOF
  exit 2
fi

mkdir -p "${RUN_DIR}/weights" "${RUN_DIR}/logs" "${RUN_DIR}/exports"

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install -r "${DETECTOR_REPO_DIR}/requirements.txt"

{
  echo "run_name=${RUN_NAME}"
  echo "detector_repo_url=${DETECTOR_REPO_URL}"
  git -C "${DETECTOR_REPO_DIR}" rev-parse HEAD | sed 's/^/detector_repo_commit=/'
  echo "network=${NETWORK}"
  echo "widerface_dir=${WIDERFACE_DIR}"
  echo "run_dir=${RUN_DIR}"
  echo "batch_size=${BATCH_SIZE}"
  echo "num_workers=${NUM_WORKERS}"
  echo "learning_rate=${LEARNING_RATE}"
  date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ'
} > "${RUN_DIR}/run.env"

cd "${DETECTOR_REPO_DIR}"

train_args=(
  train.py
  --train-data "${WIDERFACE_DIR}/train"
  --network "${NETWORK}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --learning-rate "${LEARNING_RATE}"
  --save-dir "${RUN_DIR}/weights"
)

if [[ "${RESUME}" == "1" ]]; then
  train_args+=(--resume)
fi

"${CONDA_BIN}" run -n "${ENV_NAME}" python "${train_args[@]}" 2>&1 | tee "${RUN_DIR}/logs/train.log"

if [[ "${EXPORT_ONNX}" == "1" ]]; then
  final_weights="${RUN_DIR}/weights/${NETWORK}_final.pth"
  if [[ ! -s "${final_weights}" ]]; then
    echo "Expected final weights not found: ${final_weights}" >&2
    exit 3
  fi
  "${CONDA_BIN}" run -n "${ENV_NAME}" python -m scripts.onnx_export \
    --weights "${final_weights}" \
    --network "${NETWORK}" \
    --dynamic \
    2>&1 | tee "${RUN_DIR}/logs/export_onnx.log"
  exported_name="$(basename "${final_weights}" .pth).onnx"
  if [[ -s "${DETECTOR_REPO_DIR}/${exported_name}" ]]; then
    mv "${DETECTOR_REPO_DIR}/${exported_name}" "${RUN_DIR}/exports/fastfacedetector_${NETWORK}.onnx"
  fi
fi

date -u '+finished_utc=%Y-%m-%dT%H:%M:%SZ' >> "${RUN_DIR}/run.env"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${FASTFACE_WORK_ROOT:-${PROJECT_DIR}}}"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${ENV_NAME:-faceattr}"
CLONE_ENV="${CLONE_ENV:-vlm}"

mkdir -p "${PROJECT_DIR}" "${DATA_ROOT}"/{data/{raw,interim,processed,manifests},models,runs,third_party,logs/downloads}

ln -sfn "${DATA_ROOT}/data" "${PROJECT_DIR}/data"
ln -sfn "${DATA_ROOT}/models" "${PROJECT_DIR}/models"
ln -sfn "${DATA_ROOT}/runs" "${PROJECT_DIR}/runs"
ln -sfn "${DATA_ROOT}/third_party" "${PROJECT_DIR}/third_party"
ln -sfn "${DATA_ROOT}/logs" "${PROJECT_DIR}/logs"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  if "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${CLONE_ENV}"; then
    "${CONDA_BIN}" create -n "${ENV_NAME}" --clone "${CLONE_ENV}" -y
  else
    "${CONDA_BIN}" create -n "${ENV_NAME}" python=3.11 -y
  fi
fi

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install --upgrade pip

if ! "${CONDA_BIN}" run -n "${ENV_NAME}" python - <<'PY'
import torch
import torchvision
print(torch.__version__, torchvision.__version__)
PY
then
  "${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install \
  timm==0.8.13.dev0 opencv-python pandas numpy scikit-learn tqdm albumentations \
  datasets huggingface_hub gdown kaggle onnx onnxruntime onnxscript pillow pyyaml

"${CONDA_BIN}" run -n "${ENV_NAME}" python - <<'PY'
import torch
import torchvision
import cv2
import timm
import pandas
import numpy
import onnx
import onnxruntime

print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("torchvision", torchvision.__version__)
print("cv2", cv2.__version__)
print("timm", timm.__version__)
print("pandas", pandas.__version__)
print("numpy", numpy.__version__)
print("onnx", onnx.__version__)
print("onnxruntime", onnxruntime.__version__)
PY

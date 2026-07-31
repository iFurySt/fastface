#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${ENV_NAME:-faceattr}"

cd "${PROJECT_DIR}"

if [ "$#" -gt 0 ]; then
  RUN_DIRS=("$@")
else
  RUN_DIRS=(
    runs/efficientnet_v2_s_128_real_fairface_utkface
    runs/efficientnet_v2_s_128_gender_priority_real_fairface_utkface
    runs/efficientnet_b0_128_real_fairface_utkface
    runs/efficientnet_b0_128_gender_priority_real_fairface_utkface
    runs/resnet18_128_real_fairface_utkface
    runs/convnext_tiny_128_real_fairface_utkface
    runs/swin_t_128_real_fairface_utkface
    runs/mobilenetv3_real_fairface_utkface
    runs/mobilenetv3_large128_distill_efficientnet_v2_s_fairface_utkface
    runs/mobilenetv3_large128_distill_gender_efficientnet_v2_s_fairface_utkface
    runs/mobilenetv3_large128_distill_gender_priority_efficientnet_v2_s_fairface_utkface
    runs/mobilenetv3_large128_distill_light_efficientnet_v2_s_fairface_utkface
    runs/mobilenetv3_large112_real_fairface_utkface
    runs/mobilenetv3_large112_distill_efficientnet_v2_s_fairface_utkface
    runs/mobilenetv3_large112_distill_gender_efficientnet_v2_s_fairface_utkface
    runs/mobilenetv3_small112_real_fairface_utkface
    runs/mobilenetv3_small112_distill_efficientnet_b0_fairface_utkface
    runs/mobilenetv3_small112_distill_efficientnet_v2_s_fairface_utkface
    runs/mobilenetv3_small112_distill_gender_efficientnet_v2_s_fairface_utkface
    runs/mobilenetv3_small128_real_fairface_utkface
    runs/mobilenetv3_small112_distill_large112_fairface_utkface
    runs/mobilenetv3_small112_distill_light_large112_fairface_utkface
    runs/mobilenetv3_small112_lagenda_real_fairface_utkface
    runs/mobilenetv3_small112_lagenda_facecrop_real_fairface_utkface
  )
fi

for run_dir in "${RUN_DIRS[@]}"; do
  if [ ! -d "${run_dir}" ]; then
    echo "skip missing run: ${run_dir}" >&2
    continue
  fi
  "${CONDA_BIN}" run -n "${ENV_NAME}" python -m fastface.export.generate_model_card \
    --run-dir "${run_dir}"
done

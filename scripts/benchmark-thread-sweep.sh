#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${ENV_NAME:-faceattr}"
THREADS="${THREADS:-1 2 4 8 16 28 56 112}"
BATCH_SIZES="${BATCH_SIZES:-1 8 32 128}"
WARMUP="${WARMUP:-10}"
ITERATIONS="${ITERATIONS:-100}"

cd "${PROJECT_DIR}"

run_case() {
  local run_dir="$1"
  local model_name="$2"
  local input_size="$3"
  local model_path="${run_dir}/${model_name}.onnx"
  local out_dir="${run_dir}/cpu-thread-sweep"
  if [ ! -f "${model_path}" ]; then
    echo "skip missing model: ${model_path}" >&2
    return 0
  fi
  mkdir -p "${out_dir}"
  for threads in ${THREADS}; do
    local output_path="${out_dir}/${model_name}_threads${threads}.json"
    if [[ -s "${output_path}" ]]; then
      echo "skip existing benchmark: ${output_path}" >&2
      continue
    fi
    "${CONDA_BIN}" run -n "${ENV_NAME}" python -m fastface.export.benchmark_onnx \
      --model "${model_path}" \
      --output "${output_path}" \
      --input-size "${input_size}" \
      --batch-sizes ${BATCH_SIZES} \
      --warmup "${WARMUP}" \
      --iterations "${ITERATIONS}" \
      --intra-op-num-threads "${threads}" \
      --inter-op-num-threads 1 \
      --execution-mode sequential >/dev/null
  done
}

run_case runs/mobilenetv3_real_fairface_utkface model_fp32 128
run_case runs/mobilenetv3_real_fairface_utkface model_int8_static 128
run_case runs/efficientnet_b0_128_gender_priority_real_fairface_utkface model_fp32 128
run_case runs/efficientnet_b0_128_gender_priority_real_fairface_utkface model_int8_static 128
run_case runs/mobilenetv3_large128_distill_efficientnet_v2_s_fairface_utkface model_fp32 128
run_case runs/mobilenetv3_large128_distill_efficientnet_v2_s_fairface_utkface model_int8_static 128
run_case runs/mobilenetv3_large128_distill_gender_efficientnet_v2_s_fairface_utkface model_fp32 128
run_case runs/mobilenetv3_large128_distill_gender_efficientnet_v2_s_fairface_utkface model_int8_static 128
run_case runs/mobilenetv3_large128_distill_gender_priority_efficientnet_v2_s_fairface_utkface model_fp32 128
run_case runs/mobilenetv3_large128_distill_gender_priority_efficientnet_v2_s_fairface_utkface model_int8_static 128
run_case runs/mobilenetv3_large128_distill_light_efficientnet_v2_s_fairface_utkface model_fp32 128
run_case runs/mobilenetv3_large128_distill_light_efficientnet_v2_s_fairface_utkface model_int8_static 128
run_case runs/mobilenetv3_large112_real_fairface_utkface model_fp32 112
run_case runs/mobilenetv3_large112_real_fairface_utkface model_int8_static 112
run_case runs/mobilenetv3_large112_distill_efficientnet_v2_s_fairface_utkface model_fp32 112
run_case runs/mobilenetv3_large112_distill_efficientnet_v2_s_fairface_utkface model_int8_static 112
run_case runs/mobilenetv3_large112_distill_gender_efficientnet_v2_s_fairface_utkface model_fp32 112
run_case runs/mobilenetv3_large112_distill_gender_efficientnet_v2_s_fairface_utkface model_int8_static 112
run_case runs/mobilenetv3_small112_real_fairface_utkface model_fp32 112
run_case runs/mobilenetv3_small112_real_fairface_utkface model_int8_static 112
run_case runs/mobilenetv3_small112_distill_efficientnet_b0_fairface_utkface model_fp32 112
run_case runs/mobilenetv3_small112_distill_efficientnet_b0_fairface_utkface model_int8_static 112
run_case runs/mobilenetv3_small112_distill_efficientnet_v2_s_fairface_utkface model_fp32 112
run_case runs/mobilenetv3_small112_distill_efficientnet_v2_s_fairface_utkface model_int8_static 112
run_case runs/mobilenetv3_small112_distill_gender_efficientnet_v2_s_fairface_utkface model_fp32 112
run_case runs/mobilenetv3_small112_distill_gender_efficientnet_v2_s_fairface_utkface model_int8_static 112
run_case runs/mobilenetv3_small112_distill_large112_fairface_utkface model_fp32 112
run_case runs/mobilenetv3_small112_distill_large112_fairface_utkface model_int8_static 112
run_case runs/mobilenetv3_small112_distill_light_large112_fairface_utkface model_fp32 112
run_case runs/mobilenetv3_small112_distill_light_large112_fairface_utkface model_int8_static 112
run_case runs/mobilenetv3_small112_lagenda_real_fairface_utkface model_fp32 112
run_case runs/mobilenetv3_small112_lagenda_real_fairface_utkface model_int8_static 112
run_case runs/mobilenetv3_small112_lagenda_facecrop_real_fairface_utkface model_fp32 112
run_case runs/mobilenetv3_small112_lagenda_facecrop_real_fairface_utkface model_int8_static 112

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m fastface.export.summarize_thread_sweep \
  --run-dir runs/mobilenetv3_real_fairface_utkface \
  --run-dir runs/efficientnet_b0_128_gender_priority_real_fairface_utkface \
  --run-dir runs/mobilenetv3_large128_distill_efficientnet_v2_s_fairface_utkface \
  --run-dir runs/mobilenetv3_large128_distill_gender_efficientnet_v2_s_fairface_utkface \
  --run-dir runs/mobilenetv3_large128_distill_gender_priority_efficientnet_v2_s_fairface_utkface \
  --run-dir runs/mobilenetv3_large128_distill_light_efficientnet_v2_s_fairface_utkface \
  --run-dir runs/mobilenetv3_large112_real_fairface_utkface \
  --run-dir runs/mobilenetv3_large112_distill_efficientnet_v2_s_fairface_utkface \
  --run-dir runs/mobilenetv3_large112_distill_gender_efficientnet_v2_s_fairface_utkface \
  --run-dir runs/mobilenetv3_small112_real_fairface_utkface \
  --run-dir runs/mobilenetv3_small112_distill_efficientnet_b0_fairface_utkface \
  --run-dir runs/mobilenetv3_small112_distill_efficientnet_v2_s_fairface_utkface \
  --run-dir runs/mobilenetv3_small112_distill_gender_efficientnet_v2_s_fairface_utkface \
  --run-dir runs/mobilenetv3_small112_distill_large112_fairface_utkface \
  --run-dir runs/mobilenetv3_small112_distill_light_large112_fairface_utkface \
  --run-dir runs/mobilenetv3_small112_lagenda_real_fairface_utkface \
  --run-dir runs/mobilenetv3_small112_lagenda_facecrop_real_fairface_utkface

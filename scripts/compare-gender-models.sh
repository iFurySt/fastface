#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/analysis/gender-comparison-$(date +%Y%m%d-%H%M%S)}"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${ENV_NAME:-faceattr}"
MIVOLO_REPO="${MIVOLO_REPO:-${PROJECT_DIR}/third_party/MiVOLO}"
MIVOLO_CHECKPOINT="${MIVOLO_CHECKPOINT:-${PROJECT_DIR}/third_party/mivolo/weights/model_imdb_face_4.22_99.38.pth.tar}"
INCLUDE_MIVOLO="${INCLUDE_MIVOLO:-1}"

cd "${PROJECT_DIR}"

MODEL_ARGS=(
  --model "our_large128_imdb_distill:fastface:${RUN_ROOT}/mobilenetv3_large128_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface/best.pt:128:0.2"
  --model "our_small112_imdb_distill:fastface:${RUN_ROOT}/mobilenetv3_small112_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface/best.pt:112:0.2"
  --model "teacher_v2s_imdb:fastface:${RUN_ROOT}/efficientnet_v2_s_128_imdb_source_balanced_gender_priority_real_fairface_utkface/best.pt:128:0.2"
  --model "public_fairface_onnx:fairface_onnx:third_party/fairface-onnx/weights/fairface.onnx:224:0.25"
)

if [[ "${INCLUDE_MIVOLO}" == "1" ]]; then
  MODEL_ARGS+=(--model "mivolo_imdb_face:mivolo_face:${MIVOLO_CHECKPOINT}:224:0.2")
fi

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m fastface.evaluation.compare_gender_models \
  --manifest "${DATA_ROOT}/manifests/fairface.jsonl" \
  --manifest "${DATA_ROOT}/manifests/utkface.jsonl" \
  --manifest "${DATA_ROOT}/manifests/imdb_clean.jsonl" \
  --sample-limit fairface=10954 \
  --sample-limit utkface=2425 \
  --sample-limit imdb-clean=10954 \
  "${MODEL_ARGS[@]}" \
  --batch-size "${BATCH_SIZE:-512}" \
  --public-batch-size "${PUBLIC_BATCH_SIZE:-256}" \
  --mivolo-batch-size "${MIVOLO_BATCH_SIZE:-128}" \
  --mivolo-repo "${MIVOLO_REPO}" \
  --num-workers "${NUM_WORKERS:-12}" \
  --output-dir "${OUTPUT_DIR}"

echo "comparison_output=${OUTPUT_DIR}"

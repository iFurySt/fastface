#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

PYTHONPATH="${PROJECT_DIR}/packages${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m fastface.pipeline.predict_image "$@"

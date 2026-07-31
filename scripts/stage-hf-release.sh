#!/usr/bin/env bash
set -euo pipefail

RELEASE_ID="${RELEASE_ID:-fastface-v0.1.0}"
DEST_DIR="${DEST_DIR:-outputs/hf/${RELEASE_ID}}"

LARGE_RUN="${LARGE_RUN:?LARGE_RUN is required}"
SMALL_RUN="${SMALL_RUN:?SMALL_RUN is required}"
TEACHER_RUN="${TEACHER_RUN:?TEACHER_RUN is required}"
COMPARISON_DIR="${COMPARISON_DIR:-}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

mkdir -p "${DEST_DIR}/models" "${DEST_DIR}/reports"

copy_file() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "${dst}")"
  rsync -a "${src}" "${dst}"
}

copy_optional_file() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "${dst}")"
  if rsync -a "${src}" "${dst}" 2>/dev/null; then
    return 0
  fi
  echo "skip optional missing artifact: ${src}" >&2
}

stage_variant() {
  local variant="$1"
  local run_dir="$2"
  local variant_dir="${DEST_DIR}/models/${variant}"

  mkdir -p "${variant_dir}"
  for file in \
    best.pt \
    model_fp32.onnx \
    model_int8_static.onnx \
    config.resolved.yaml \
    evaluation_val.json \
    benchmark_fp32_cpu.json \
    benchmark_int8_static_cpu.json \
    model_card.md; do
    copy_file "${run_dir%/}/${file}" "${variant_dir}/${file}"
  done

  copy_optional_file "${run_dir%/}/cpu-thread-sweep-summary.json" "${variant_dir}/cpu-thread-sweep-summary.json"
  copy_optional_file "${run_dir%/}/metrics.jsonl" "${variant_dir}/metrics.jsonl"
}

stage_variant "fastface-large-128" "${LARGE_RUN}"
stage_variant "fastface-small-112" "${SMALL_RUN}"
stage_variant "fastface-teacher-v2s-128" "${TEACHER_RUN}"

copy_file "docs/cards/fastface-hf-model-card.md" "${DEST_DIR}/README.md"
copy_file "docs/TECHNICAL_REPORT.md" "${DEST_DIR}/technical_report.md"
copy_file "docs/DATA_PROVENANCE.md" "${DEST_DIR}/data_provenance.md"
copy_file "docs/model-runs.md" "${DEST_DIR}/model_runs.md"
if [[ -n "${COMPARISON_DIR}" ]]; then
  copy_optional_file "${COMPARISON_DIR%/}/summary.json" "${DEST_DIR}/reports/gender-comparison-summary.json"
fi

cat > "${DEST_DIR}/reports/manual-public-gender-review-schema.md" <<'MARKDOWN'
# Manual Public Gender Review Schema

The manual review workbook is intentionally not uploaded because it embeds source images.

Current source artifact:

```text
outputs/analysis/manual-public-gender-review-current
```

The workbook contains public FairFace-ONNX vs FastFace Large disagreement rows only.

Important columns:

- `review_buckets`
- `image`
- `manual_gender`
- `sample_id`
- `dataset`
- `label_gender_name`
- `label_age`
- `our_large_gender`
- `public_fairface_gender`
- `our_large_male_prob`
- `public_fairface_male_prob`
- `teacher_gender`
- `small112_gender`
- `correct_models`
- `wrong_models`
- `confidence_gap`
MARKDOWN

python - "${DEST_DIR}" "${RELEASE_ID}" > "${DEST_DIR}/release_manifest.json" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
release_id = sys.argv[2]
files = []

for path in sorted(root.rglob("*")):
    if not path.is_file() or path.name == "release_manifest.json":
        continue
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    files.append({
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    })

print(json.dumps({
    "release_id": release_id,
    "file_count": len(files),
    "files": files,
}, indent=2, sort_keys=True))
PY

echo "staged ${RELEASE_ID} at ${DEST_DIR}"

## [2026-07-31 11:34] | Task: Add MiVOLO baseline comparison

### Execution Context

- Agent ID: `TRAE CLI`
- Base Model: `GPT-5`
- Runtime: `TraeCode`

### User Query

> Add MiVOLO to the fixed gender baseline comparison, which previously compared
> only against public FairFace-ONNX. Use `<remote-gpu-host>` if needed.

### Changes Overview

- Area: evaluation baseline comparison and GPU reproducibility docs.
- Key actions:
  - Added `mivolo_face` support to `compare_gender_models.py`.
  - Wired MiVOLO into `scripts/compare-gender-models.sh` by default.
  - Restored generic pairwise focused disagreement CSV generation.
  - Staged the official MiVOLO face-only IMDB-clean age/gender checkpoint on the
    GPU host.
  - Ran the full fixed comparison and recorded MiVOLO results in docs.

### Design Intent

The comparison uses MiVOLO's face-only IMDB-clean checkpoint with manifest
`bbox_face` crops when available and aligned face images otherwise. This avoids
adding YOLO detector quality as a confounder in a model-vs-model gender
comparison. The adapter imports MiVOLO model/preprocessing code directly and
does not use MiVOLO's demo detector wrapper, which segfaulted in the GPU
environment through the `mivolo.structures` import path.

The GPU environment now pins `timm==0.8.13.dev0` because newer `timm 1.x` builds
are not compatible with MiVOLO's VOLO constructor/import expectations.

### Result

- Output: `outputs/analysis/gender-comparison-mivolo-current`.
- Selected rows: 24,333.
- MiVOLO aggregate gender balanced accuracy: `0.94461`.
- Public FairFace-ONNX aggregate gender balanced accuracy: `0.94658`.
- FastFace Large aggregate gender balanced accuracy: `0.95618`.
- Teacher aggregate gender balanced accuracy: `0.96638`.
- FastFace Large vs MiVOLO disagreements: 1,532 rows.

### Files Modified

- `packages/fastface/evaluation/compare_gender_models.py`
- `scripts/compare-gender-models.sh`
- `scripts/bootstrap-gpu-env.sh`
- `README.md`
- `docs/TECHNICAL_REPORT.md`
- `docs/model-runs.md`
- `docs/GPU_ENVIRONMENT.md`

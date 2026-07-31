## [2026-07-31 12:18] | Task: Detector Pipeline Baseline

### Execution Context

- Agent ID: `traecode`
- Base Model: `GPT-5`
- Runtime: `local macOS shell`

### User Query

> Build toward an end-to-end FastFace pipeline with a separate detector stage,
> using GPU training later for an owned detector.

### Changes Overview

- Area: full-image inference, serving contract, detector training plan.
- Key actions:
  - Added a pluggable full-image pipeline CLI that runs a detector backend,
    aligns detected faces, and sends crops into the FastFace ONNX model.
  - Added UniFace RetinaFace/SCRFD as optional baseline detector backends.
  - Added a serving contract for crop mode, full-image mode, JSON output schema,
    and future `fastfacedetector` training.
  - Added an active execution plan for detector and pipeline work.
  - Updated README, architecture, and quality docs to reflect the two-stage
    `fastfacedetector + fastface` direction.

### Design Intent

Keep the FastFace attribute model focused on aligned face crops while making
raw-image behavior explicit through a detector stage. UniFace is only the first
baseline; the long-term target is an owned detector with bbox, score, and
5-point landmarks so no-face images can be rejected before age/gender inference.

### Files Modified

- `packages/fastface/pipeline/detectors.py`
- `packages/fastface/pipeline/runtime.py`
- `packages/fastface/pipeline/predict_image.py`
- `scripts/predict-image.sh`
- `docs/SERVING_CONTRACT.md`
- `docs/exec-plans/active/fastface-detector-pipeline.md`
- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/QUALITY_SCORE.md`
- `pyproject.toml`

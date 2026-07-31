## [2026-07-31 12:45] | Task: Detector Training Entrypoints

### Execution Context

- Agent ID: `traecode`
- Base Model: `GPT-5`
- Runtime: `local macOS shell`

### User Query

> Commit and push the pipeline baseline, then start training an owned
> `fastfacedetector`.

### Changes Overview

- Area: detector training workflow.
- Key actions:
  - Added a WIDER FACE detector-data preparation script.
  - Added a RetinaFace-MobileNet training wrapper that records run metadata,
    writes artifacts outside Git, and exports ONNX after training.
  - Updated the active detector execution plan with the owned-detector training
    slice and data contract.

### Design Intent

Use a RetinaFace-MobileNetV2 training backend as the first owned detector slice
because it has a direct WIDER FACE plus 5-point-landmark training path and an
MIT-licensed reference implementation. SCRFD remains a later architecture target
after the data pipeline, artifact layout, and FastFace serving contract are
proven end to end.

### Files Modified

- `scripts/prepare-widerface-detector-data.sh`
- `scripts/start-detector-training.sh`
- `docs/exec-plans/active/fastface-detector-pipeline.md`

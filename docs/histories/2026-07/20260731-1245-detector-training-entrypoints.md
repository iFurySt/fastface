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
  - Added a WIDER FACE detector benchmark harness and used it to compare the
    owned bootstrap detector against UniFace detector baselines.

### Design Intent

Use a RetinaFace-MobileNetV2 training backend as the first owned detector slice
because it has a direct WIDER FACE plus 5-point-landmark training path and an
MIT-licensed reference implementation. SCRFD remains a later architecture target
after the data pipeline, artifact layout, and FastFace serving contract are
proven end to end.

The first bootstrap run used WIDER FACE boxes with generated pseudo landmarks
because the public RetinaFace landmark archive link was unavailable. This is a
bbox/no-face detector milestone, not the final alignment-quality detector. On
the full WIDER FACE validation split, the owned RetinaFace MobileNetV1 0.50
bootstrap detector at confidence `0.45` reached F1 `0.61629`, exceeding the
UniFace RetinaFace MNetV2 baseline F1 `0.59099` under the repository IoU/F1
harness. The exported ONNX artifact was checked with ONNX Runtime.

Follow-up detector experiments showed that beating the baseline on one metric is
not enough. Whole-image 960 training improved val100 F1 to `0.60827` at
confidence `0.45`, but precision remained `0.77907`. Raising the threshold to
confidence `0.75` improved precision to `0.91367` but dropped recall to
`0.40144` and F1 to `0.55780`. A clean teacher-only fine-tune from that
checkpoint improved precision further, reaching `0.87950` precision at
confidence `0.55`, but recall/F1 fell to `0.40421`/`0.55391`. This route is a
useful precision ceiling, not the final detector route.

The WIDER FACE benchmark harness now rejects unsupported non-square resize modes
for UniFace detector baselines instead of accepting a CLI argument that the
underlying baseline wrapper cannot honor.

### Files Modified

- `scripts/prepare-widerface-detector-data.sh`
- `scripts/start-detector-training.sh`
- `packages/fastface/data/build_widerface_detector_labels.py`
- `packages/fastface/evaluation/benchmark_widerface_detectors.py`
- `docs/exec-plans/active/fastface-detector-pipeline.md`

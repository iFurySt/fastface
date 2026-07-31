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

Profiling then showed that the optimized ONNX path was spending most latency in
image preprocessing and per-image prior generation rather than NMS. The harness
now uses `cv2.dnn.blobFromImage` for numerically equivalent BGR mean subtraction
and vectorized NumPy prior generation. On the full WIDER FACE validation split,
the owned ONNX detector with max-side 1280, confidence `0.55`, NMS `0.3`, and
pre-NMS topK `1000` reached precision `0.89052`, recall `0.45003`, F1 `0.59790`,
and `0.02038` seconds/image, exceeding UniFace RetinaFace MNetV2 on bbox
quality and latency under the repository benchmark harness. Landmark/alignment
quality remains a separate finalization gate.

The owned ONNX detector backend is now available in the full-image pipeline CLI.
A smoke test on a WIDER FACE sample returned three faces through
`crop_mode: "landmark_5pt"` and a generated plain image returned
`status: "no_face"`.

An alignment benchmark was added to compare candidate landmarks, aligned crops,
and FastFace predictions against UniFace RetinaFace MNetV2. The bbox/latency
bootstrap detector still has weak alignment evidence. Teacher-landmark
whole-image and clean fine-tune candidates improve alignment metrics, but one
misses precision and the other misses recall/F1, so detector finalization remains
open.

### Files Modified

- `scripts/prepare-widerface-detector-data.sh`
- `scripts/start-detector-training.sh`
- `packages/fastface/data/build_widerface_detector_labels.py`
- `packages/fastface/evaluation/benchmark_detector_alignment.py`
- `packages/fastface/evaluation/benchmark_widerface_detectors.py`
- `packages/fastface/pipeline/detectors.py`
- `packages/fastface/pipeline/predict_image.py`
- `docs/SERVING_CONTRACT.md`
- `docs/exec-plans/active/fastface-detector-pipeline.md`

# FastFace Detector And Full-Image Pipeline

## Goal

Provide an end-to-end FastFace image pipeline that accepts a raw image, rejects
images where no face is detected, and returns gender plus numeric age for each
accepted face. The long-term model shape is a two-stage pipeline:
`fastfacedetector` for face detection and alignment metadata, followed by
`fastface` for age/gender inference.

## Scope

- In scope:
  - Add a repository-native full-image prediction CLI.
  - Keep the current FastFace ONNX age/gender model as a crop-only attribute
    model.
  - Add a pluggable detector interface with UniFace detector backends as the
    first baseline.
  - Document the serving contract, detector output schema, and detector training
    direction.
- Out of scope:
  - Training `fastfacedetector` in this first slice.
  - Publishing detector weights in the phase-1 Hugging Face release.
  - Adding race prediction or FairFace attribute outputs.

## Context

- Relevant docs:
  - `docs/ARCHITECTURE.md`
  - `docs/TECHNICAL_REPORT.md`
  - `docs/MODEL_RELEASE.md`
  - `docs/GPU_ENVIRONMENT.md`
- Relevant code paths:
  - `packages/fastface/export/export_onnx.py`
  - `packages/fastface/data/manifest_dataset.py`
  - `packages/fastface/pipeline/`
  - `scripts/predict-image.sh`
- Constraints:
  - FastFace product output remains gender and numeric age only.
  - Raw detector data, checkpoints, and ONNX binaries must not be committed.
  - The detector dependency must remain optional until FastFace owns a detector
    model.

## Risks

- Risk: The current FastFace training path used bbox crops for some datasets,
  while production raw-image inference may use landmark alignment.
- Mitigation: Treat UniFace detector alignment as a baseline, then compare bbox
  crop versus landmark alignment on validation and manual-review samples before
  freezing the detector contract.

- Risk: Third-party detector code or weights could complicate release terms.
- Mitigation: Keep UniFace as an optional baseline backend and train/release
  `fastfacedetector` separately before making the full-image pipeline a fully
  self-contained release.

- Risk: No-face and multi-face cases can be ambiguous for product consumers.
- Mitigation: Return explicit JSON statuses and detected-face counts instead of
  silently resizing full scenes into the attribute model.

## Milestones

1. Implement a full-image baseline CLI with optional UniFace detector backends.
2. Document the serving contract and detector training plan.
3. Benchmark UniFace detector baselines against YuNet and current manifest bbox
   crops.
4. Train and export `fastfacedetector` on the GPU host.
5. Replace or complement the UniFace baseline with the owned detector backend.

## Owned Detector Training Slice

The first owned detector slice uses a RetinaFace-MobileNetV2 training backend
because it has a direct WIDER FACE + 5-point-landmark PyTorch training path and
MIT-licensed reference code. SCRFD remains the preferred architecture family for
later optimization, but its original training stack has higher setup cost.

Training data contract:

- `data/raw/widerface/train/images/`
- `data/raw/widerface/train/label.txt`

The `label.txt` file must include face boxes and five landmarks in the
RetinaFace format. Raw WIDER FACE detection boxes alone are insufficient for
the alignment contract.

If the RetinaFace landmark archive is unavailable, `scripts/prepare-widerface-detector-data.sh`
can generate bbox-derived pseudo landmarks with `ALLOW_PSEUDO_LANDMARKS=1`.
That mode is only a bootstrap path for face/no-face and bbox detection; it is
not a final alignment-quality detector.

Training entry points:

- `scripts/prepare-widerface-detector-data.sh`
- `scripts/start-detector-training.sh`

Default first run:

```sh
NETWORK=mobilenetv2 \
RUN_NAME=fastfacedetector_retinaface_mobilenetv2_widerface \
bash scripts/start-detector-training.sh
```

Expected non-Git artifacts:

- `runs/fastfacedetector_retinaface_mobilenetv2_widerface/weights/`
- `runs/fastfacedetector_retinaface_mobilenetv2_widerface/exports/`
- `runs/fastfacedetector_retinaface_mobilenetv2_widerface/logs/`

Stronger comparison target:

- Beat UniFace RetinaFace MNetV2 on full WIDER FACE validation F1.
- Beat or match UniFace RetinaFace MNetV2 on recall.
- Close the current precision and latency gaps, or record why the tradeoff is
  acceptable for the FastFace no-face/bbox gate.
- Add a separate landmark-quality benchmark before calling the detector final
  for alignment use.

Follow-up experiments:

- `fastfacedetector_retinaface_mobilenetv1_050_teacher_retinaface_mnetv2`:
  trains on WIDER FACE boxes with UniFace RetinaFace MNetV2 teacher landmarks
  where teacher boxes match GT boxes, falling back to bbox-derived landmarks.
- `fastfacedetector_retinaface_mobilenetv1_050_teacher_retinaface_mnetv2_fixed640`:
  uses the same teacher-label data with a remote training-worktree augmentation
  patch that preserves whole-image scale distribution for fixed-640 inference.
- `fastfacedetector_retinaface_mobilenetv2_widerface_bootstrap`: tests the
  stronger MobileNetV2 backbone offline with pretrained weights disabled.

## Validation

- Commands:
  - `make check`
  - `PYTHONPATH=packages python -m fastface.pipeline.predict_image --help`
  - `PYTHONPATH=packages python -m compileall -q packages`
- Manual checks:
  - Run the pipeline on one no-face image and one face image after model
    artifacts are available locally.
  - Compare output status, bbox, confidence, gender probabilities, and age.
- Observability checks:
  - JSON output must include detector backend, detector score, crop mode,
    FastFace model path, input size, and per-face predictions.

## Progress Log

- [x] Confirmed FastFace currently resizes crops and only consumes existing
  manifest `bbox_face`; it does not run automatic detection.
- [x] Confirmed `fairface-onnx` demo uses UniFace RetinaFace plus 5-point
  alignment, while the FairFace attribute model itself does not detect faces.
- [x] Reviewed UniFace detector options and licensing notes.
- [x] Implement the baseline full-image pipeline.
- [x] Document the detector training path and GPU workflow.
- [x] Run local validation.
- [x] Add detector data preparation and RetinaFace training wrapper scripts.
- [ ] Stage WIDER FACE images and RetinaFace label data on the GPU host.
- [ ] Start the first `fastfacedetector` RetinaFace-MobileNetV2 bootstrap
  training run.
- [x] Stage WIDER FACE train/val images and split annotations on the GPU host.
- [x] Train the first bootstrap owned detector using RetinaFace MobileNetV1
  0.50 and WIDER FACE bbox-derived pseudo landmarks.
- [x] Benchmark the owned detector against UniFace RetinaFace/SCRFD on WIDER
  FACE validation boxes.
- [x] Export the owned detector to ONNX and verify ONNX Runtime can load it.
- [x] Add UniFace teacher pseudo-landmark label generation.
- [x] Start teacher-label and fixed-640-friendly follow-up training runs.
- [ ] Decide whether teacher-label or fixed-640 training closes precision,
  latency, and landmark-quality gaps.

Validation notes:

- `PYTHONPATH=packages python -m fastface.pipeline.predict_image --help`
  passed.
- `bash scripts/predict-image.sh --help` passed.
- `PYTHONPATH=packages python -m compileall -q packages` passed.
- `bash -n scripts/*.sh` passed.
- `make check` passed.
- Created a local pipeline test environment with `python -m pip install
  ".[pipeline]"`.
- `retinaface_mnet_v2` no-face smoke test on a generated plain image returned
  `status: no_face` and `face_count: 0`.
- `retinaface_mnet_v2` full-image smoke test on a local Lagenda sample returned
  `status: ok`, `face_count: 3`, and `crop_mode: landmark_5pt` for every face.
- `scrfd_500m` full-image smoke test on the same sample with `--max-faces 1`
  returned `status: ok`, `face_count: 1`, and `crop_mode: landmark_5pt`.
- JSON assertions passed for status, face count, probability ranges, crop mode,
  and age bounds. Test outputs are under ignored `outputs/pipeline-test/`.

## Decision Log

- 2026-07-31: Keep FastFace as a crop-only attribute model and add a separate
  detector stage. Rationale: the product model should stay focused on age/gender
  and avoid silently treating full-scene images as aligned faces. Consequence:
  full-image inference is a pipeline concern.
- 2026-07-31: Use UniFace RetinaFace/SCRFD as optional baseline detector
  backends before training `fastfacedetector`. Rationale: this gives immediate
  end-to-end behavior and a comparison target. Consequence: UniFace remains an
  optional dependency, not part of the core model contract.
- 2026-07-31: Prefer future SCRFD-like or RetinaFace-MobileNet-like detector
  training over reproducing YuNet blindly. Rationale: the detector must provide
  robust bbox plus 5-point landmarks for alignment; YuNet is a lightweight
  baseline, not the accuracy ceiling. Consequence: YuNet remains a baseline in
  evaluation, not the default architecture choice.
- 2026-07-31: Use RetinaFace-MobileNetV2 as the first owned detector training
  slice. Rationale: it has a direct WIDER FACE + 5-point-landmark PyTorch
  training path with MIT reference code. Consequence: SCRFD remains a later
  architecture target after the detector data and pipeline contract are proven.
- 2026-07-31: Allow bbox-derived pseudo landmarks only for bootstrap training
  when RetinaFace landmark labels are unavailable. Rationale: WIDER FACE bbox
  annotations are easy to stage and let us train an owned no-face/bbox detector
  while the 5-point landmark archive is blocked. Consequence: bootstrap weights
  must not be treated as the final alignment detector.
- 2026-07-31: The first bootstrap detector exceeded the UniFace RetinaFace
  baseline on full WIDER FACE validation using the repository IoU/F1 harness.
  Evidence: owned RetinaFace MobileNetV1 0.50 at confidence `0.45` reached
  precision `0.81871`, recall `0.49412`, and F1 `0.61629` over 3,226 validation
  images; UniFace RetinaFace MNetV2 at confidence `0.5` reached precision
  `0.87753`, recall `0.44552`, and F1 `0.59099` on the same split. Consequence:
  the bootstrap detector satisfies the current bbox-detection target, while
  real landmark labels remain a future alignment-quality improvement.
- 2026-07-31: Dynamic/original-size ONNX inference can exceed UniFace RetinaFace
  MNetV2 on F1, precision, and recall after threshold tuning, but still lags
  latency. Fixed-640 inference gets closer to UniFace latency but loses too much
  recall, so the latency gap is a training-scale distribution issue rather than
  only a postprocessing issue.
- 2026-07-31: Added a max-side resize benchmark path. Max-side 1280 keeps
  owned-detector precision, recall, and F1 above UniFace RetinaFace MNetV2 but
  remains slower. Evidence on full WIDER FACE validation: owned max-side 1280
  reached precision `0.89052`, recall `0.45003`, F1 `0.59790`, and `0.04573`
  seconds/image; UniFace RetinaFace MNetV2 reached precision `0.87753`, recall
  `0.44552`, F1 `0.59099`, and `0.02699` seconds/image.
- 2026-07-31: Fixed-640-friendly training improves fixed-640 recall/F1 but has
  not reached the full target. Evidence on val100: epoch 40 reached F1 `0.59997`
  and recall `0.50138` at confidence `0.45`, but precision was only `0.74680`;
  higher confidence improves precision but lowers F1/recall. Clean teacher-only
  fixed-640 training started as a high-precision follow-up and reached val100 F1
  `0.52491` at epoch 13.

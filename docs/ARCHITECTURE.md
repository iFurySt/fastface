# Architecture

This file is the top-level map for the repository.

## Intended Repository Shape

- `apps/`: deployable applications or entry points.
- `packages/`: shared libraries for data manifests, model definitions, model factories, training, export, and inference.
- `infra/`: deployment, infrastructure, and environment definitions.
- `scripts/`: repository automation that agents can run directly, including dataset bootstrap and GPU setup scripts.
- `docs/`: the repository knowledge base and system of record.

## Boundary Rules

- Put business logic in reusable packages before spreading it across apps.
- Keep infrastructure and runtime orchestration explicit and versioned.
- Avoid hidden cross-package coupling; document allowed dependency directions once the stack is real.
- When the architecture changes, update this file in the same task.

## Product Architecture

FastFace is an age and gender model project optimized for CPU inference throughput.

The first production attribute-model target is a MobileNetV3 student with:

- Aligned face crop input.
- Gender classification head.
- Numeric age output derived from an age-distribution head.
- ONNX Runtime CPU deployment.
- INT8 benchmark path.

Raw-image inference is a two-stage pipeline:

- `fastfacedetector`: detects faces, rejects no-face images, and returns bbox
  plus 5-point alignment landmarks.
- `fastface`: predicts gender and numeric age from the aligned face crop.

The current repository pipeline uses optional UniFace detector backends as
baselines until an owned `fastfacedetector` is trained and released.

Race prediction is explicitly out of scope.

The model package also supports torchvision challengers such as EfficientNet-B0, EfficientNetV2-S, ResNet18, and ConvNeXt-Tiny through a shared age/gender factory. These are used for real accuracy/latency comparisons and teacher checkpoints while MobileNetV3 remains the CPU-first default.

## Training Architecture

Training uses a teacher/student approach:

- Teacher models optimize label quality and hard-sample discovery.
- Student models optimize CPU deployment speed and throughput.
- Dataset manifests are the boundary between raw data, processed crops, labels, and teacher outputs.

See `docs/design-docs/face-attribute-model.md` for model details.
See `docs/datasets.md` for dataset and label policy.
See `docs/GPU_ENVIRONMENT.md` for the GPU workspace.
See `docs/model-runs.md` for completed training, export, and CPU benchmark results.
See `docs/SERVING_CONTRACT.md` for crop-mode and full-image runtime contracts.

# Serving Contract

FastFace serves age and gender in two layers:

1. `fastfacedetector`: face detection and face-crop/alignment metadata.
2. `fastface`: age/gender inference on an aligned RGB face crop.

The phase-1 released FastFace attribute models do not detect faces. The
repository now provides a full-image pipeline CLI so raw-image behavior is
explicit and testable while the owned detector is trained separately.

## Input Modes

### Crop Mode

Use this mode when the caller already has an aligned face crop.

Input contract:

- RGB or BGR face crop, converted by the caller or runtime.
- `128x128` for `fastface-large-128`.
- `112x112` for `fastface-small-112`.
- ImageNet normalization:
  - mean: `[0.485, 0.456, 0.406]`
  - std: `[0.229, 0.224, 0.225]`

### Full-Image Mode

Use this mode when the caller has a raw image. The runtime must run a detector
before FastFace. It must never silently resize a full scene into the age/gender
model.

Detector output contract:

```json
{
  "bbox": [120.5, 44.0, 256.2, 210.8],
  "detector_score": 0.98,
  "landmarks": [
    [154.0, 91.0],
    [219.0, 90.5],
    [187.1, 124.0],
    [163.2, 159.8],
    [213.0, 160.1]
  ]
}
```

The preferred detector output includes 5 alignment landmarks: left eye, right
eye, nose, left mouth corner, and right mouth corner. If landmarks are missing,
the pipeline may fall back to bbox crop, but that is a weaker mode and should be
reported as `crop_mode: "bbox"`.

## Output Schema

The full-image CLI writes JSON:

```json
{
  "status": "ok",
  "image": "input.jpg",
  "detector": "uniface:retinaface:retinaface_mnet_v2",
  "fastface_model": "model_fp32.onnx",
  "fastface_input_size": 128,
  "face_count": 1,
  "faces": [
    {
      "index": 0,
      "bbox": [120.5, 44.0, 256.2, 210.8],
      "detector_score": 0.98,
      "landmarks": [[154.0, 91.0], [219.0, 90.5], [187.1, 124.0], [163.2, 159.8], [213.0, 160.1]],
      "crop_mode": "landmark_5pt",
      "gender": 0,
      "gender_name": "female",
      "female_prob": 0.92,
      "male_prob": 0.08,
      "gender_confidence": 0.92,
      "age": 31.7
    }
  ]
}
```

Statuses:

- `ok`: one or more faces were detected and scored.
- `no_face`: detector returned no face above threshold.

Future pipeline implementations may add statuses such as `low_confidence`,
`invalid_image`, or `detector_error`, but they must remain explicit.

## Current CLI

Run a raw-image prediction with the UniFace RetinaFace baseline:

```sh
bash scripts/predict-image.sh \
  --image input.jpg \
  --model models/fastface-large-128/model_fp32.onnx \
  --detector retinaface \
  --detector-model retinaface_mnet_v2 \
  --max-faces 0
```

Run with SCRFD:

```sh
bash scripts/predict-image.sh \
  --image input.jpg \
  --model models/fastface-large-128/model_fp32.onnx \
  --detector scrfd \
  --detector-model scrfd_10g
```

Run with the owned ONNX detector:

```sh
bash scripts/predict-image.sh \
  --image input.jpg \
  --model models/fastface-large-128/model_fp32.onnx \
  --detector owned-retinaface-onnx \
  --owned-detector-onnx models/fastfacedetector/fastfacedetector.onnx \
  --detector-input-size 1280 \
  --detector-resize-mode max-side \
  --detector-conf 0.55 \
  --detector-nms 0.3 \
  --detector-pre-nms-topk 1000
```

The UniFace backends are baseline dependencies, not the final owned detector.
Install the pipeline dependencies with:

```sh
python -m pip install ".[pipeline]"
```

## Detector Training Direction

`fastfacedetector` should be trained as a small ONNX face detector with:

- face/no-face rejection by confidence threshold,
- bbox output,
- 5-point landmark output,
- CPU-friendly latency,
- an Apache-2.0-compatible release story for code and weights.

Recommended architecture candidates:

- SCRFD-like detector as the primary training direction.
- RetinaFace-MobileNet-like detector as the mature baseline.
- CenterFace-like detector as a lightweight baseline if landmark quality is
  sufficient.

YuNet remains a comparison baseline. It is not required if `fastfacedetector`
covers the same role with stronger bbox and landmark quality.

## GPU Workflow

Detector training should run on the GPU host:

```sh
ssh <remote-gpu-host>
cd <repo-root>
```

Use the existing FastFace environment and artifact layout from
`docs/GPU_ENVIRONMENT.md`. Detector raw data, checkpoints, exports, and
benchmark outputs must stay outside GitHub under the training workspace.

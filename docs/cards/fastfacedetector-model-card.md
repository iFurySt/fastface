# FastFaceDetector RetinaFace MobileNetV1 0.50 Whole960

This detector is the current FastFace full-image pipeline release candidate.
It detects faces, returns bounding boxes and 5-point alignment landmarks, and
rejects no-face images by confidence threshold before FastFace age/gender
inference.

## Recommended Runtime

Use the owned detector backend with:

```sh
bash scripts/predict-image.sh \
  --image input.jpg \
  --model models/fastface-large-128/model_fp32.onnx \
  --detector owned-retinaface-onnx \
  --owned-detector-onnx models/fastfacedetector-retinaface-mnetv1-960/fastfacedetector_retinaface_mobilenetv1_050_whole960_epoch34.onnx \
  --detector-input-size 1280 \
  --detector-resize-mode max-side \
  --detector-conf 0.65 \
  --detector-nms 0.3 \
  --detector-pre-nms-topk 1000
```

The ONNX export uses external data. Keep
`mobilenetv1_0.50_last.onnx.data` in the same directory as the `.onnx` file.

## Architecture

- Detector family: RetinaFace.
- Backbone: MobileNetV1 0.50.
- Training image size: 960.
- Runtime input mode: max-side resize to 1280.
- Outputs: bbox, confidence, 5-point landmarks.

## Training Data

The detector was trained on WIDER FACE train images with UniFace RetinaFace
MNetV2 teacher landmarks where teacher boxes matched WIDER FACE ground-truth
boxes. This makes the detector suitable for bbox plus alignment metadata in the
FastFace pipeline, but WIDER FACE itself is still a public face-detection
benchmark with its own data-license terms.

## Benchmarks

Full WIDER FACE validation, same repository benchmark harness:

| Detector | Precision | Recall | F1 | Seconds/Image |
| --- | ---: | ---: | ---: | ---: |
| FastFaceDetector candidate | 0.87832 | 0.48082 | 0.62144 | 0.02088 |
| UniFace RetinaFace MNetV2 baseline | 0.87753 | 0.44552 | 0.59099 | 0.02699 |

Alignment comparison against UniFace RetinaFace MNetV2 on the first 50 WIDER
FACE validation images:

| Metric | Value |
| --- | ---: |
| Matched faces | 603 |
| Match rate vs baseline | 0.9000 |
| Mean normalized landmark error | 0.0908 |
| Mean aligned-crop MAE | 0.0896 |
| Mean FastFace age absolute difference | 4.2993 |
| FastFace gender agreement | 0.8740 |

## Limitations

- The alignment benchmark uses UniFace RetinaFace MNetV2 as the comparison
  baseline, not human landmark annotations.
- The model is optimized for CPU ONNX Runtime in the FastFace pipeline, not for
  GPU throughput.
- WIDER FACE benchmark quality does not guarantee performance on every product
  domain. Keep no-face and low-confidence handling explicit in callers.

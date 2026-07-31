---
license: apache-2.0
library_name: onnx
tags:
  - computer-vision
  - image-classification
  - face-analysis
  - age-estimation
  - gender-classification
  - onnx
  - cpu
pipeline_tag: image-classification
---

# FastFace

FastFace is a CPU-efficient face age and gender model family. Phase 1 freezes three variants:

| Variant | Purpose | Input | Primary artifact |
| --- | --- | --- | --- |
| `fastface-large-128` | Recommended accuracy/throughput student | aligned RGB face crop, 128x128 | `models/fastface-large-128/model_fp32.onnx` |
| `fastface-small-112` | Highest-throughput student | aligned RGB face crop, 112x112 | `models/fastface-small-112/model_fp32.onnx` |
| `fastface-teacher-v2s-128` | Audit/teacher model, not CPU default | aligned RGB face crop, 128x128 | `models/fastface-teacher-v2s-128/model_fp32.onnx` |

The released task is gender classification plus numeric age estimation. Race prediction is intentionally out of scope.

## Training Data

Training used aligned face crops and manifests built from public/research datasets available in the local training workspace:

- FairFace train/validation labels.
- UTKFace aligned face images.
- IMDB-clean derived from the IMDB-WIKI family after quality filtering.
- Lagenda-hosted face-age data was explored but is not part of the frozen phase-1 student release.

See `data_provenance.md` and `technical_report.md` in this repository for the exact source policy, manifest contract, and limitations.

## Metrics

Validation gender balanced accuracy on the mixed public validation set:

| Model | Mixed GBA | FairFace GBA | IMDB-clean GBA | UTKFace GBA |
| --- | ---: | ---: | ---: | ---: |
| `fastface-teacher-v2s-128` | 0.98605 | 0.94386 | 0.99138 | 0.95424 |
| `fastface-large-128` | 0.97929 | 0.92877 | 0.98548 | 0.95017 |
| `fastface-small-112` | 0.96800 | 0.90562 | 0.97542 | 0.94101 |

In a 24,333-sample comparison set, gender balanced accuracy was:

- `fastface-teacher-v2s-128`: 0.96638
- `fastface-large-128`: 0.95618
- public `fairface-onnx`: 0.94658
- `fastface-small-112`: 0.94059

Public FairFace-ONNX vs `fastface-large-128` disagreed on 1,301 samples. Against the available public labels, `fastface-large-128` was correct on 762 of those and public FairFace-ONNX was correct on 539.

## Intended Use

- High-throughput CPU inference on already-detected/aligned face crops.
- Gender and age signals for product analytics or moderation-assist workflows where uncertainty and bias are handled upstream/downstream.
- Teacher-assisted auditing and future distillation.

## Limitations

- The model does not detect faces; it expects aligned face crops.
- Age labels are noisy across public face-age datasets, so age should be treated as an estimate.
- Gender labels follow dataset annotations and can encode social and labeling bias.
- Race/ethnicity classification is not provided.
- Metrics reflect the assembled validation manifests, not universal real-world performance.

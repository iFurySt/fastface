# FastFace Phase-1 Technical Report

Status: phase-1 frozen candidate set.

Date: 2026-07-31.

## Scope

FastFace trains face-attribute models for:

- Gender classification.
- Numeric age prediction.

Race prediction is deliberately excluded. Race labels in public datasets are not
used as a product output because the category is culturally unstable, difficult
for humans to assign consistently, and creates avoidable product and compliance
risk.

The phase-1 production target is CPU inference throughput, not maximum GPU
accuracy. Heavy models are used as teachers and evaluators; deployment
candidates are MobileNetV3 students exported to ONNX.

## Hardware And Environment

Training was run on `<remote-gpu-host>`:

- Hostname: `<gpu-hostname>`.
- GPUs: 8 x NVIDIA GPU accelerator.
- Conda root: `<conda-root>`.
- Conda env: `faceattr`.
- Project path: `<repo-root>`.
- Data path: `${FASTFACE_WORK_ROOT}`.

The environment, package versions, and setup commands are recorded in
[`GPU_ENVIRONMENT.md`](GPU_ENVIRONMENT.md).

## Data

Phase-1 manifests on the GPU host:

| Dataset | Rows | Role | Label Notes |
| --- | ---: | --- | --- |
| FairFace | 97,698 | gender training, age-range auxiliary supervision, fairness/public validation | gender exact; age range weak supervision |
| UTKFace | 23,705 | exact-age and gender training/validation | gender exact; age exact from filename |
| IMDB-clean | 285,946 | exact-age/gender data expansion and teacher/student training | gender exact; age exact; MiVOLO face boxes |
| Lagenda-HF | 12,409 | staged but not promoted in phase 1 | exact labels, requires `bbox_face` crop |

See [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md) for source URLs, staging paths,
download caveats, and license/access cautions.

## Manifest Contract

Training consumes JSONL manifests. Required fields include:

- `sample_id`
- `dataset`
- `split`
- `image_path`
- `gender`, with `0=female`, `1=male`
- `age`, `age_min`, `age_max`
- `age_label_type`, either `exact` or `range`
- `age_loss_weight`

Optional fields include:

- `bbox_face`
- `bbox_person`
- upstream label fields such as `gender_original`

When `bbox_face` exists, training/evaluation can crop the face box with a margin
before resizing. Phase-1 IMDB and Lagenda experiments use this path.

## Model Architecture

All FastFace models use two heads:

- Gender head: binary logits.
- Age head: a `0..100` age distribution with numeric expectation output.

The age-distribution head keeps the product output simple while making training
more stable than direct scalar regression.

Implemented backbones:

- MobileNetV3 Small/Large.
- EfficientNet-B0.
- EfficientNetV2-S.
- ResNet18.
- ConvNeXt-Tiny.
- Swin-T.

The deployment candidates are MobileNetV3 students. EfficientNetV2-S is the
strongest phase-1 teacher/challenger.

## Training Objective

Primary selection target: gender balanced accuracy.

Secondary target: numeric age quality.

Representative gender-priority loss settings:

```yaml
loss:
  gender_weight: 4.0
  age_weight: 0.5
```

For teacher/student runs, distillation was used only where it improved the
student tradeoff. The final IMDB source-balanced student distillation uses:

```yaml
distillation:
  enabled: true
  temperature: 2.0
  gender_weight: 0.2
  age_weight: 0.0
```

This keeps teacher supervision low-weight and gender-focused.

## Source Balancing

Natural IMDB mixing caused the validation aggregate to be dominated by
IMDB-clean. Source-balanced follow-up runs cap IMDB-clean rows during training
and checkpoint-selection validation:

```yaml
data:
  train_sample_limits:
    imdb-clean: 90000
  val_sample_limits:
    imdb-clean: 10954
```

Final evaluation still reports full source-sliced metrics so aggregate gains are
not mistaken for robust public-domain gains.

## Main Runs

| Run | Backbone | Purpose | Mixed Gender Balanced Acc | FairFace | IMDB-clean | UTKFace | Tuned FP32 Batch-128 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_v2_s_128_imdb_source_balanced_gender_priority_real_fairface_utkface` | EfficientNetV2-S | IMDB-inclusive teacher/challenger | 0.98605 | 0.94386 | 0.99138 | 0.95424 | default FP32 1,044.1 img/s |
| `mobilenetv3_large128_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface` | MobileNetV3-Large | accuracy-oriented CPU candidate | 0.97929 | 0.92877 | 0.98548 | 0.95017 | 4,477.7 img/s |
| `mobilenetv3_small112_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface` | MobileNetV3-Small | throughput-oriented CPU candidate | 0.96800 | 0.90562 | 0.97542 | 0.94101 | 10,837.8 img/s |

The full historical run matrix, including negative challengers and CPU
benchmarks, is in [`model-runs.md`](model-runs.md).

## Public Baseline Comparison

A fixed comparison was run against public FairFace-ONNX and MiVOLO on:

- FairFace validation.
- UTKFace validation.
- A seed-stable IMDB-clean validation sample capped to FairFace size.

Selected rows: 24,333.

| Model | Gender Balanced Acc | Gender Acc |
| --- | ---: | ---: |
| `teacher_v2s_imdb` | 0.96638 | 0.96638 |
| `our_large128_imdb_distill` | 0.95618 | 0.95640 |
| `public_fairface_onnx` | 0.94658 | 0.94723 |
| `mivolo_imdb_face` | 0.94461 | 0.94530 |
| `our_small112_imdb_distill` | 0.94059 | 0.94103 |

For rows where `our_large128_imdb_distill` and public FairFace-ONNX disagree:

- Count: 1,301.
- FastFace Large correct / public wrong by public labels: 762.
- Public correct / FastFace Large wrong by public labels: 539.

This is not treated as final truth because public labels include ambiguous or
noisy samples. The current manual review workbook is built only from this
public-vs-FastFace disagreement set.

## Frozen Phase-1 Decision

Freeze these variants:

- `fastface-large-128`: default accuracy-oriented CPU candidate.
- `fastface-small-112`: throughput-oriented CPU candidate.
- `fastface-teacher-v2s-128`: teacher/evaluation artifact, not a CPU default.

Do not continue broad architecture search for phase 1. Next work should use the
manual disagreement review to decide whether data relabeling, filtering, or
targeted fine-tuning is justified.

## Known Limitations

- Public datasets contain noisy labels, especially in hard gender cases.
- FairFace age labels are ranges; FairFace age MAE is directional only.
- IMDB-clean is celebrity-heavy and can inflate mixed aggregate metrics.
- Lagenda raw images are full-scene images and require face-box cropping.
- INT8 did not beat FP32 for sustained MobileNetV3 high-batch throughput in the
  measured ONNX Runtime environment.
- MiVOLO is now included through the face-only IMDB-clean checkpoint. The GPU
  host still cannot reach Google Drive directly, so the checkpoint was
  downloaded locally and staged under `third_party/mivolo`.

## Reproducibility Entry Points

```sh
bash scripts/build-training-manifests.sh
bash scripts/train-and-finalize.sh
bash scripts/compare-gender-models.sh
bash scripts/build-manual-gender-review.sh
```

Every completed run should keep:

- `config.resolved.yaml`
- `metrics.jsonl`
- `best.pt`
- `evaluation_val.json`
- `model_fp32.onnx`
- `model_int8_static.onnx`
- CPU benchmark JSON files
- `model_card.md`

# Face Attribute Model

## Product Target

FastFace predicts only:

- `gender`: binary model output used as the primary accuracy target.
- `age`: a single numeric estimate, not an age range.

The project deliberately does not predict race. Race labels are hard for people to assign consistently, map poorly across cultures, and add avoidable policy and product risk. Datasets that contain race labels may be used only for bias analysis or stratified validation when license terms allow it.

## Deployment Target

The production model must prioritize CPU throughput:

- ONNX Runtime CPU is the default inference target.
- INT8 quantization is expected for production benchmarking.
- The attribute model consumes an aligned face crop from the upstream face detector/alignment pipeline.
- The production path should not depend on body crops, multi-image context, or heavyweight teacher models.

Default input contract:

- RGB aligned face crop.
- `128x128` for the balanced production model.
- `112x112` for the fastest model variant.
- Normalization must be recorded in the exported model metadata or companion config.
- Training manifests that point at full-scene images must provide a face box and use the shared face-crop path before resize. Full-scene resize is not a valid training input for the production student.

## Architecture Choice

Use a teacher/student strategy.

### Teacher

Teacher models optimize label quality, not production throughput. Candidate teachers:

- MiVOLO v2 for age and gender supervision, especially when body context helps disambiguate hard cases.
- ConvNeXt-Tiny or EfficientNetV2-S as image-only teachers for simpler crop-only distillation.
- An ensemble of public checkpoints and internally trained models when pseudo-label quality matters more than inference cost.

Teacher outputs may be used for:

- Pseudo-labeling unlabeled data.
- Hard-sample mining.
- Soft-label distillation.
- Disagreement triage for human review.

Teacher models are not production dependencies.

### Student

The first production student should be:

- Backbone: `MobileNetV3-Large`.
- Input: `128x128`.
- Heads:
  - `gender`: 2 logits.
  - `age`: 101 logits for ages `0..100`.

Age inference should return the expectation over the age distribution:

```text
age = sum(softmax(age_logits)[i] * i for i in 0..100)
```

This keeps the product output as a single number while avoiding the instability of direct scalar regression.

Additional student/challenger variants to benchmark:

- `MobileNetV3-Small`, `112x112`: fastest CPU candidate.
- `ResNet18`, `128x128`: sanity-check baseline, not expected to win throughput.
- `EfficientNet-B0`, `128x128`: accuracy/latency challenger using the shared torchvision model factory.
- `MobileNetV4` small variants: second-round challenger after the baseline pipeline is stable.
- `EfficientFormerV2-S0` or other lightweight transformer variants: optional research challenger, not the default production path.

Standard ViT/Swin-style models are not the production baseline because CPU throughput and quantized ONNX deployment are worse fits for this target. They remain useful as teachers or offline baselines.

## Training Objective

Primary loss:

```text
loss = gender_weight * CE(gender_logits, gender_label)
     + age_weight * CE(age_logits, soft_age_distribution)
     + distill_weight * teacher_distillation_loss
```

Default weights:

- `gender_weight = 2.0`
- `age_weight = 1.0`
- `distill_weight = 0.25` once teacher labels are available

Gender is the main business metric. Model selection should prefer a checkpoint with stronger gender balanced accuracy when age metrics are close.

Age labels should use soft distributions around the labeled age, for example a Gaussian-like target centered on the known age and clipped to `0..100`.

## Validation Metrics

Track all metrics on a fixed internal validation set before trusting public benchmark numbers.

Required:

- Gender accuracy.
- Gender balanced accuracy.
- Gender false-positive and false-negative rates by source dataset.
- Age MAE.
- Age median absolute error.
- Age CS@5, the share of predictions within 5 years.
- CPU latency and throughput for FP32 and INT8 ONNX.

When demographic labels are legally and ethically usable for evaluation, record fairness slices privately:

- Gender quality by apparent age bucket.
- Age MAE by source dataset.
- Error rate by image quality, pose, blur, occlusion, and lighting.

Do not ship race prediction.

## Export Requirements

Every production candidate must produce:

- PyTorch checkpoint.
- ONNX FP32 model.
- ONNX INT8 model.
- Model card with training data, metrics, known limits, and license notes.
- A reproducible CPU benchmark report.

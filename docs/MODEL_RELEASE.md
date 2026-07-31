# Model Release

Phase-1 model binaries are hosted on Hugging Face, not GitHub.

The source repository and frozen FastFace model artifacts use Apache-2.0.
Upstream dataset licenses and access terms remain separate and are documented in
`docs/DATA_PROVENANCE.md`.

Target model repository:

```text
https://huggingface.co/iFurySt/fastface
```

Release id:

```text
fastface-v0.1.0
```

Uploaded revision:

```text
https://huggingface.co/iFurySt/fastface/commit/a09750496274b7197fe55bf147b24f864b4f2abb
```

Latest Hugging Face license-sync revision:

```text
https://huggingface.co/iFurySt/fastface/commit/cbc06fb5aada3d286a6fad1785761a5b63450d1b
```

Source repository:

```text
https://github.com/iFurySt/fastface
```

## Why Hugging Face

GitHub is used for source, configs, docs, and reproducibility scripts. It is not
used for checkpoints or ONNX binaries because the frozen artifacts include
multiple checkpoint and ONNX files, including a 240MB+ EfficientNetV2-S teacher.

Hugging Face is the better fit for:

- model cards,
- variant directories,
- versioned model artifacts,
- downstream model download workflows.

## Frozen Variants

| Variant | Directory | Role | Include |
| --- | --- | --- | --- |
| `fastface-large-128` | `models/fastface-large-128/` | accuracy-oriented CPU candidate | PyTorch checkpoint, FP32 ONNX, static INT8 ONNX, config, metrics, eval, CPU benchmarks |
| `fastface-small-112` | `models/fastface-small-112/` | throughput-oriented CPU candidate | PyTorch checkpoint, FP32 ONNX, static INT8 ONNX, config, metrics, eval, CPU benchmarks |
| `fastface-teacher-v2s-128` | `models/fastface-teacher-v2s-128/` | teacher/evaluation artifact | PyTorch checkpoint, FP32 ONNX, static INT8 ONNX, config, metrics, eval, CPU benchmarks |

The public FairFace-ONNX baseline is not redistributed in the FastFace model
repo; it is referenced as an external baseline.

## Hugging Face Layout

```text
README.md
technical_report.md
data_provenance.md
model_runs.md
models/
  fastface-large-128/
    best.pt
    model_fp32.onnx
    model_int8_static.onnx
    config.resolved.yaml
    evaluation_val.json
    benchmark_fp32_cpu.json
    benchmark_int8_static_cpu.json
    cpu-thread-sweep-summary.json
    model_card.md
  fastface-small-112/
    ...
  fastface-teacher-v2s-128/
    ...
reports/
  gender-comparison-summary.json
  manual-public-gender-review-schema.md
```

## Artifact Source Directories

```text
runs/mobilenetv3_large128_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface
runs/mobilenetv3_small112_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface
runs/efficientnet_v2_s_128_imdb_source_balanced_gender_priority_real_fairface_utkface
```

## Upload Checklist

1. Regenerate model cards for frozen run directories.
2. Stage Hugging Face release files locally under `outputs/hf/fastface-v0.1.0`:

   ```sh
   make stage-hf-release
   ```

3. Upload with:

   ```sh
   hf upload iFurySt/fastface outputs/hf/fastface-v0.1.0 . --repo-type model
   ```

4. Verify the Hugging Face repository has all three variant directories.
5. Record the commit URL or revision in this document.
6. Keep GitHub free of raw datasets, checkpoints, ONNX files, and review
   workbooks.

## Current Manual Review Artifact

Manual public-vs-FastFace gender review output:

```text
outputs/analysis/manual-public-gender-review-current
```

The workbook is intentionally not part of the model release because it is a
human-labeling work product rather than a model binary.

# FastFace

FastFace is a transparent age/gender face-attribute training project optimized
for high-throughput CPU inference.

The product surface is intentionally narrow:

- Input: an aligned face crop, or a raw image routed through the explicit
  detector + FastFace pipeline.
- Output: gender and one numeric age value.
- Out of scope: race prediction.

Phase 1 is frozen around MobileNetV3 students trained on public age/gender data
with an EfficientNetV2-S teacher. Large model binaries are hosted outside
GitHub on Hugging Face: <https://huggingface.co/iFurySt/fastface>.

## Current Models

| Variant | Purpose | Input | Main Metric | CPU Throughput |
| --- | --- | ---: | ---: | ---: |
| `fastface-large-128` | Accuracy-oriented CPU candidate | 128 | gender balanced acc `0.95618` on the fixed public comparison set | tuned FP32 batch-128 `4,477.7 img/s` |
| `fastface-small-112` | Throughput-oriented CPU candidate | 112 | gender balanced acc `0.94059` on the fixed public comparison set | tuned FP32 batch-128 `10,837.8 img/s` |
| `fastface-teacher-v2s-128` | Training/evaluation teacher | 128 | gender balanced acc `0.96638` on the fixed public comparison set | not a CPU deployment default |

See [`docs/model-runs.md`](docs/model-runs.md) for the full run matrix,
source-sliced metrics, ONNX sizes, and CPU benchmarks.

## Full-Image Inference

The released FastFace attribute models do not detect faces. For raw images, use
the repository pipeline so no-face handling and face alignment are explicit:

```sh
bash scripts/predict-image.sh \
  --image input.jpg \
  --model models/fastface-large-128/model_fp32.onnx \
  --detector retinaface \
  --detector-model retinaface_mnet_v2
```

Current detector backends are optional UniFace baselines. The intended owned
runtime shape is `fastfacedetector + fastface`; see
[`docs/SERVING_CONTRACT.md`](docs/SERVING_CONTRACT.md).

## Repository Layout

```text
configs/train/                 Training configs used for phase-1 runs
packages/fastface/data/         Manifest builders and dataset loading
packages/fastface/models/       MobileNetV3 and torchvision age/gender models
packages/fastface/pipeline/     Full-image detector + FastFace runtime
packages/fastface/training/     DDP training entry point
packages/fastface/export/       ONNX export, quantization, benchmarks, model cards
packages/fastface/evaluation/   Evaluation, model comparison, manual review tooling
scripts/                        Reproducible GPU/data/release automation
docs/                           Technical report, data provenance, runs, model cards
```

## Documentation

Read these first:

- [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md): phase-1 technical report.
- [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md): datasets, access class, labels, and staging.
- [`docs/MODEL_RELEASE.md`](docs/MODEL_RELEASE.md): Hugging Face artifact layout and release checklist.
- [`docs/SERVING_CONTRACT.md`](docs/SERVING_CONTRACT.md): crop-mode and full-image inference contract.
- [`docs/GPU_ENVIRONMENT.md`](docs/GPU_ENVIRONMENT.md): GPU host setup and commands.
- [`docs/design-docs/face-attribute-model.md`](docs/design-docs/face-attribute-model.md): model design.

## Reproduce The Training Environment

On the GPU host:

```sh
ssh <remote-gpu-host>
cd <repo-root>
bash scripts/bootstrap-gpu-env.sh
```

Build or refresh manifests:

```sh
cd <repo-root>
bash scripts/build-training-manifests.sh
```

Start a real training run from a config:

```sh
cd <repo-root>
CONFIG=configs/train/mobilenetv3_large_128_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb.yaml \
RUN_DIR=runs/mobilenetv3_large128_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface \
NPROC_PER_NODE=8 \
bash scripts/train-and-finalize.sh
```

The finalizer evaluates the best checkpoint, exports ONNX FP32 and static INT8,
runs CPU benchmarks, runs thread sweeps when enabled, and writes a model card.

## Compare Against Public Baselines

Run the fixed public-baseline-vs-FastFace comparison:

```sh
cd <repo-root>
OUTPUT_DIR=outputs/analysis/gender-comparison-current \
bash scripts/compare-gender-models.sh
```

By default this compares the current FastFace Large/Small candidates, the
EfficientNetV2-S teacher, public FairFace-ONNX, and MiVOLO's face-only
IMDB-clean age/gender checkpoint.

Build the manual labeling workbook for rows where FastFace Large and public
FairFace-ONNX disagree:

```sh
cd <repo-root>
OUTPUT_DIR=outputs/analysis/manual-public-gender-review-current \
bash scripts/build-manual-gender-review.sh
```

## Artifact Policy

GitHub stores source, configs, scripts, and documentation. Do not commit raw
datasets, checkpoints, ONNX files, review workbooks, or cached downloads.

Release artifacts are uploaded to Hugging Face under the `fastface` model
repository with separate variant directories. See
[`docs/MODEL_RELEASE.md`](docs/MODEL_RELEASE.md).

## License

Repository code and FastFace model artifacts are released under Apache-2.0.
Dataset usage must also respect the upstream dataset licenses and access terms documented in
[`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md).

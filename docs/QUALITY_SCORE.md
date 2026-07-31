# Quality Score

Track quality by product area and architectural layer so agents can prioritize the weakest parts of the system.

## Suggested Scale

- `A`: strong coverage, stable behavior, clear docs, low operational risk.
- `B`: acceptable but still has known gaps.
- `C`: works but needs targeted hardening.
- `D`: fragile or underspecified.

## Current Snapshot

| Area | Score | Why | Next Step |
| --- | --- | --- | --- |
| Product surface | B | The first product surface is defined as CPU-efficient face image inference for gender and numeric age, with race explicitly out of scope. A full-image detector + FastFace CLI now makes no-face handling explicit while the owned detector remains planned. | Validate the pipeline on real face/no-face samples and train an owned `fastfacedetector`. |
| Architecture docs | B | FastFace model, dataset, GPU training, ONNX export, HF release, CPU benchmark docs, and the serving contract describe the phase-1 architecture and the two-stage raw-image path. | Add detector benchmark results and ONNX parity checks for the full-image pipeline. |
| Testing | C | Manifest builders compile and have been exercised against real FairFace, Lagenda, IMDB, and UTKFace annotations; completed runs have source-sliced evaluation, ONNX exports, model cards, INT8 variant sweeps, CPU benchmarks, and an IMDB manifest validation gate. Training now supports sample-limited source balancing for dominant datasets. The model factory now instantiates MobileNetV3, EfficientNet-B0, EfficientNetV2-S, ResNet18, ConvNeXt-Tiny, and Swin-T challengers, but full automated tests are still missing. | Add parser/unit tests for sample limits and an export parity check. |
| Observability | B | Training writes resolved config, per-epoch JSONL metrics, source-sliced evaluation JSON, best/last checkpoints, ONNX exports, benchmark JSON, model cards, thread-sweep summaries when practical, and long-running dataset preparation logs. The completed natural IMDB, source-balanced IMDB, IMDB-pretrained fine-tune, EfficientNetV2-S IMDB-inclusive teacher/challenger, Small112 IMDB teacher-distillation, and Large128 IMDB teacher-distillation runs exposed aggregate/source-slice divergence and documented the current strongest Small-family throughput-oriented and MobileNetV3 accuracy-oriented IMDB-inclusive gender candidates, with source caps and checkpoint initialization recorded in configs and final model cards. The gender comparison command now writes sample-level predictions, model disagreement CSVs, focused review slices, contact sheets, and an image-embedded manual-label workbook against public FairFace-ONNX and MiVOLO baselines. Dataset status now reports IMDB-clean actual bytes, expected bytes, total percentage, per-tar progress, compact tar summaries, transfer rate, and ETA; the IMDB pipeline monitor records repeated ensure/status snapshots. | Add a single comparison index across model cards and record ONNX parity metrics. |
| Artifact hygiene | B | `.gitignore` keeps raw data, local outputs, checkpoints, and ONNX files out of GitHub; `docs/MODEL_RELEASE.md` defines the HF release layout. | Add a release script that creates a SHA256 manifest and uploads a frozen HF revision. |
| Security | C | Repository scope and private disclosure policy are documented; dataset and model-card transparency are covered separately. | Add dependency scanning and model artifact checksum verification to CI once CI exists. |

# Face Attribute Training Bootstrap

## Goal

Build the first reproducible training and evaluation path for a CPU-efficient face age/gender model. The production model outputs gender and numeric age only, with no race prediction.

## Scope

- In scope:
  - Prepare GPU machine workspace and Python environment.
  - Download or stage approved public datasets.
  - Define dataset manifests and label normalization.
  - Train baseline and production-candidate student models.
  - Use teacher models such as MiVOLO for pseudo-labeling and distillation.
  - Export ONNX FP32 and INT8 candidates.
  - Benchmark CPU throughput and accuracy.
- Out of scope:
  - Race prediction.
  - Arbitrary web scraping of face images.
  - Production API service implementation.
  - Unreviewed commercial use of research-only datasets.

## Context

- Relevant docs:
  - `docs/design-docs/face-attribute-model.md`
  - `docs/datasets.md`
  - `docs/GPU_ENVIRONMENT.md`
  - `SECURITY.md`
- Relevant code paths:
  - `scripts/` for setup and automation.
  - `packages/fastface/data/` for manifest building and dataset loading.
- `packages/fastface/models/` for MobileNetV3 and torchvision age/gender models.
  - `packages/fastface/training/` for DDP training entry points.
- Constraints:
  - Production inference must optimize CPU throughput.
  - Gender accuracy is the primary model-selection target.
  - Age must be emitted as a single numeric value.
  - Teacher models can be heavy; student models cannot be.

## Risks

- Risk: public datasets have inconsistent licenses or gated access.
  - Mitigation: record access class per dataset and keep gated datasets out of production training until approved.
- Risk: celebrity-heavy datasets overfit to clean, frontal, high-quality images.
  - Mitigation: maintain a business validation set and source-sliced metrics.
- Risk: direct age regression produces unstable predictions.
  - Mitigation: train a `0..100` age distribution head and output the expectation.
- Risk: a teacher model transfers its biases into the student.
  - Mitigation: compare teacher labels against human-reviewed samples and fixed public validation sets.
- Risk: high GPU accuracy does not translate to CPU deployment.
  - Mitigation: benchmark ONNX FP32 and INT8 during every model round.

## Milestones

1. Repository docs and execution plan.
2. GPU workspace and clean conda environment.
3. Dataset download and manifest bootstrap.
4. Baseline training:
   - `ResNet18`, `128x128`.
   - `MobileNetV3-Large`, `128x128`.
5. Teacher-assisted training:
   - MiVOLO or ConvNeXt/EfficientNet teacher labels.
   - Distilled MobileNetV3 student.
6. Export and benchmark:
   - ONNX FP32.
   - ONNX INT8.
   - CPU throughput report.

## Validation

- Commands:
  - Environment import smoke test from `docs/GPU_ENVIRONMENT.md`.
  - Dataset manifest integrity checks.
  - Full manifest integrity checks.
  - Full training run on currently staged complete datasets.
  - Full validation run on held-out public and internal sets.
  - ONNX export parity check against PyTorch.
  - CPU throughput benchmark with batch sizes `1`, `8`, `32`, and `128`.
- Manual checks:
  - Inspect hard examples where teacher and student disagree.
  - Review dataset license/access status before production training.
- Observability checks:
  - Save training config, git commit, metrics, and artifact paths per run.
  - Keep benchmark JSON next to exported model artifacts.

## Progress Log

- [x] 2026-07-30: Confirmed product scope is age and gender only; race prediction removed.
- [x] 2026-07-30: Confirmed GPU host has multi-GPU accelerator capacity and sufficient storage.
- [x] 2026-07-30: Chose `MobileNetV3-Large 128x128` as first production student.
- [x] 2026-07-30: Chose age distribution head with numeric expectation output.
- [x] Prepare clean `faceattr` conda environment on `<remote-gpu-host>`.
- [x] Mirror this repository to `<repo-root>`.
- [x] Start approved dataset downloads under `data/raw`.
- [x] Stage FairFace `margin025` archive and train/val labels on GPU host.
- [x] Stage Lagenda and MiVOLO IMDB annotation archives on GPU host.
- [x] Stage IMDB-clean repository on GPU host.
- [x] Stage UTKFace cropped images from a public Hugging Face mirror on GPU host.
- [x] Create dataset manifest generation scripts.
- [x] Start 8-GPU MobileNetV3-Large training on FairFace + UTKFace.
- [x] Complete 40-epoch FairFace + UTKFace training run.
- [x] Export best checkpoint to ONNX FP32 and static INT8.
- [x] Run CPU throughput benchmark for batch sizes `1`, `8`, `32`, and `128`.
- [x] Run source-sliced validation for MobileNetV3-Large.
- [x] Complete 40-epoch MobileNetV3-Small 112x112 training run.
- [x] Export and benchmark MobileNetV3-Small 112x112.
- [x] Run ONNX Runtime CPU intra-op thread sweep for large/small FP32/static-INT8 candidates.
- [x] Complete 40-epoch MobileNetV3-Large 112x112 training run.
- [x] Export, evaluate, and thread-sweep MobileNetV3-Large 112x112.
- [x] Complete first 40-epoch MobileNetV3-Small 112x112 distillation run from the MobileNetV3-Large 112 teacher.
- [x] Export, evaluate, and thread-sweep the first MobileNetV3-Small distillation run.
- [x] Complete a lower-weight MobileNetV3-Small distillation run.
- [x] Export, evaluate, and thread-sweep the lower-weight MobileNetV3-Small distillation run.
- [x] Add a Lagenda-HF manifest path and training config for the next real data-expanded run.
- [x] Complete Lagenda-HF image mirror download and transfer.
- [x] Complete a no-crop MobileNetV3-Small 112x112 Lagenda-HF training run.
- [x] Add manifest bbox face-crop support for datasets that provide face boxes.
- [x] Complete a bbox-cropped MobileNetV3-Small 112x112 Lagenda-HF training run.
- [x] Export, evaluate, model-card, and thread-sweep both Lagenda-HF MobileNetV3-Small runs.
- [x] Add resumable original IMDB-WIKI download/extract/manifest automation for IMDB-clean.
- [x] Start real IMDB-clean original image acquisition on the GPU host.
- [x] Switch IMDB-clean image acquisition from single-connection `wget` to segmented `aria2c` resume mode.
- [x] Start a watcher that launches the 8-GPU IMDB-expanded training run once `imdb_clean.jsonl` exists.
- [x] Add post-training finalization automation for evaluation, ONNX export, CPU benchmark, thread sweep, and model card generation.
- [x] Add an idempotent IMDB pipeline ensure script for the long download/train handoff.
- [x] Run INT8 variant tuning on the MobileNetV3-Small 112 throughput candidate.
- [x] Run ONNX Runtime quant pre-processing plus INT8 variant tuning.
- [x] Complete INT8 variant tuning and keep FP32 as the current high-batch CPU throughput path.
- [x] Add source-sliced validation and model card generation.
- [x] Add compact IMDB-clean tar progress summaries for monitor snapshots.
- [x] Add a shared model factory with torchvision EfficientNet-B0/ResNet18 challengers.
- [x] Start a real 8-GPU EfficientNet-B0 128 challenger run on FairFace + UTKFace while IMDB-clean continues downloading.
- [x] Complete EfficientNet-B0 128 training, evaluation, ONNX export, CPU benchmark, thread sweep, and model card generation.
- [x] Complete ResNet18 128 training, evaluation, ONNX export, CPU benchmark, thread sweep, and model card generation.
- [x] Complete MobileNetV3-Small 112 distillation from the EfficientNet-B0 teacher.
- [x] Complete MobileNetV3-Small 128 training, evaluation, ONNX export, CPU benchmark, thread sweep, and model card generation.
- [x] Complete ConvNeXt-Tiny 128 training, evaluation, ONNX export, CPU benchmark, thread sweep, and model card generation.
- [x] Complete EfficientNetV2-S 128 training, evaluation, ONNX export, CPU benchmark, thread sweep, and model card generation.
- [x] Complete MobileNetV3-Small 112 distillation from the EfficientNetV2-S teacher.
- [x] Complete MobileNetV3-Large 128 distillation from the EfficientNetV2-S teacher.
- [x] Complete lower-weight MobileNetV3-Large 128 distillation from the EfficientNetV2-S teacher.
- [x] Complete gender-only MobileNetV3-Large 128 distillation from the EfficientNetV2-S teacher.
- [x] Complete gender-only MobileNetV3-Small 112 distillation from the EfficientNetV2-S teacher.
- [x] Fix config/checkpoint-derived input size for automatic finalization.
- [x] Complete gender-only MobileNetV3-Large 112 distillation from the EfficientNetV2-S teacher.
- [x] Complete standard MobileNetV3-Large 112 distillation from the EfficientNetV2-S teacher.
- [x] Complete gender-priority MobileNetV3-Large 128 distillation from the EfficientNetV2-S teacher.
- [x] Add Swin-T transformer challenger support and complete a real 8-GPU Swin-T 128 run.
- [x] Complete EfficientNet-B0 128 gender-priority training, evaluation, ONNX export, CPU benchmark, thread sweep, and model card generation.
- [x] Add an EfficientNetV2-S 128 gender-priority training config.
- [x] Start a real 8-GPU EfficientNetV2-S 128 gender-priority teacher challenger while IMDB-clean continues downloading.
- [x] Add an IMDB watcher idle-training gate so the IMDB 8-GPU run waits for any active full-machine challenger to finish.
- [x] Complete EfficientNetV2-S 128 gender-priority challenger training, evaluation, ONNX export, default CPU benchmark, and model-card generation.
- [x] Complete IMDB-clean image acquisition.
- [x] Validate IMDB-clean manifest with 285,946 rows.
- [x] Start IMDB-expanded MobileNetV3-Small 112 real 8-GPU training.
- [x] Complete natural-mix IMDB-expanded MobileNetV3-Small 112 training, evaluation, ONNX export, CPU benchmark, thread sweep, and model-card generation.
- [x] Add sample-limited train/validation loading for source-balanced follow-up runs.
- [x] Start source-balanced IMDB-expanded MobileNetV3-Small 112 real 8-GPU training.
- [x] Complete source-balanced IMDB-expanded MobileNetV3-Small 112 training, evaluation, ONNX export, CPU benchmark, thread sweep, and model-card generation.
- [x] Start IMDB-pretrained MobileNetV3-Small 112 fine-tuning on FairFace + UTKFace.
- [x] Add an EfficientNetV2-S 128 source-balanced gender-priority teacher/challenger config.
- [x] Queue EfficientNetV2-S 128 source-balanced gender-priority teacher/challenger after the current Small112 source-balanced job idles.
- [x] Complete IMDB-pretrained MobileNetV3-Small 112 fine-tuning and finalization.
- [x] Start EfficientNetV2-S 128 source-balanced gender-priority teacher/challenger training.
- [x] Complete EfficientNetV2-S 128 source-balanced gender-priority teacher/challenger training and finalization.
- [x] Add and start a MobileNetV3-Small 112 source-balanced IMDB gender-distillation run using the completed EfficientNetV2-S source-balanced gender-priority teacher.
- [x] Complete MobileNetV3-Small 112 source-balanced IMDB gender-distillation training and finalization.
- [x] Add a MobileNetV3-Large 128 source-balanced IMDB gender-distillation config using the completed EfficientNetV2-S source-balanced gender-priority teacher.
- [x] Complete MobileNetV3-Large 128 source-balanced IMDB gender-distillation training and finalization.
- [x] Run fixed public FairFace-ONNX vs FastFace gender comparison.
- [x] Build the public-vs-FastFace manual gender review workbook.
- [x] Freeze phase-1 variants: `fastface-large-128`, `fastface-small-112`, and `fastface-teacher-v2s-128`.
- [x] Move model binaries to the Hugging Face release path instead of GitHub.

## Decision Log

- 2026-07-30: Do not predict race. Rationale: race is hard to define consistently and adds product/compliance risk. Consequence: datasets with race labels are used only for validation slices when allowed.
- 2026-07-30: Use teacher/student training. Rationale: strong teachers can improve pseudo-labels while the student stays CPU efficient. Consequence: teacher models are training-only dependencies.
- 2026-07-30: Use MobileNetV3-Large as the first production student. Rationale: mature ONNX/INT8 CPU deployment path and strong speed/accuracy tradeoff. Consequence: transformer variants are challengers, not the default.
- 2026-07-30: Use a `0..100` age distribution head. Rationale: numeric output remains simple while training is more stable than scalar regression. Consequence: exact-age labels are preferred over age-range datasets for age supervision.
- 2026-07-30: Use `${FASTFACE_WORK_ROOT}` for large data and symlink it into `<repo-root>`. Rationale: external work storage avoids committing large artifacts. Consequence: scripts should not write large artifacts directly under the repository path.
- 2026-07-30: Prefer local download plus `rsync` when remote direct download fails. Rationale: `<remote-gpu-host>` currently fails TLS/SSL handshakes to multiple public sources. Consequence: dataset bootstrap must record whether assets were downloaded remotely or staged from the local workstation.
- 2026-07-30: Use FairFace + UTKFace as the first real training mix. Rationale: FairFace is complete and strong for gender/fairness, while UTKFace adds exact age labels immediately. Consequence: age metrics remain limited until Lagenda and IMDB-clean images are complete.
- 2026-07-30: Keep FP32 ONNX as the current throughput candidate. Rationale: static INT8 improved batch-1 latency but was slower at larger batches in the first CPU benchmark. Consequence: INT8 needs calibration/runtime tuning before being treated as production-best.
- 2026-07-30: Keep MobileNetV3-Large 128 as the current accuracy candidate and MobileNetV3-Small 112 as the current throughput candidate. Rationale: Large is about 2 points better in validation gender balanced accuracy, while Small FP32 is more than 2x faster at batch 128. Consequence: product selection should depend on the actual CPU batch pattern.
- 2026-07-30: Prefer ONNX Runtime explicit thread settings over defaults for CPU evaluation. Rationale: tuned intra-op threads substantially improved FP32 latency and throughput. Consequence: every deployment benchmark must report thread settings.
- 2026-07-30: Do not promote MobileNetV3-Large 112 as the default. Rationale: it improves batch-128 throughput over Large 128 but loses or ties at smaller batches while also losing a little accuracy. Consequence: keep Large 128 for accuracy and Small 112 for throughput until a better middle candidate is trained.
- 2026-07-30: Do not promote the first MobileNetV3-Small distillation run. Rationale: teacher loss weights `0.6` gender and `0.3` age improved age slightly but reduced gender balanced accuracy and FP32 throughput versus the non-distilled small model. Consequence: try lower teacher weights before spending effort on stronger teachers.
- 2026-07-30: Do not promote the lower-weight MobileNetV3-Small distillation run. Rationale: teacher loss weights `0.2` gender and `0.1` age still reduced gender balanced accuracy to `0.90622` and did not improve FP32 throughput. Consequence: stop tuning this teacher/student setup until either better teacher labels or more exact-age image data is available.
- 2026-07-30: Generate model cards into each run directory. Rationale: model selection needs config, validation, source slices, artifact sizes, and CPU benchmarks in one immutable artifact. Consequence: every completed run should include `model_card.md`.
- 2026-07-30: Prefer adding real exact-age data before more teacher tuning. Rationale: both tested distillation weights reduced the primary gender metric on the same data mix. Consequence: the next candidate is `MobileNetV3-Small 112` on FairFace + UTKFace + Lagenda-HF after the image mirror is complete.
- 2026-07-30: Do not train Lagenda-HF from full raw images without using `bbox_face`. Rationale: raw Lagenda images are full-scene Open Images assets; the no-crop run scored only `0.62406` gender balanced accuracy on the Lagenda validation slice and `0.87879` overall. Consequence: any Lagenda manifest consumer must crop face boxes before resizing.
- 2026-07-30: Do not promote the bbox-cropped Lagenda-HF MobileNetV3-Small run as the default. Rationale: face cropping lifted Lagenda validation gender balanced accuracy to `0.88850`, but mixed validation stayed at `0.90668`, below the FairFace + UTKFace small baseline `0.90979`. Consequence: keep Lagenda-HF for further data-quality work, not for the current production-candidate mix.
- 2026-07-30: Use original IMDB-WIKI images plus MiVOLO IMDB face boxes for IMDB-clean. Rationale: FastFace now has a shared bbox crop path, so generating a duplicate IMDB-clean-1024 crop tree is not required for the first real training pass. Consequence: `scripts/prepare-imdb-clean-images.sh` downloads the original 10 tar files, extracts them under raw data, and builds `imdb_clean.jsonl`.
- 2026-07-30: Use segmented `aria2c` for IMDB-clean downloads on the GPU host. Rationale: single-connection `wget` was progressing at only hundreds of KB/s per file, while the ETH source supports byte ranges. Consequence: the active task uses `DOWNLOAD_JOBS=10` and `ARIA2_CONNECTIONS=8` to resume existing partial tar files.
- 2026-07-30: Launch IMDB-expanded training through a manifest watcher instead of a manual follow-up. Rationale: IMDB acquisition takes hours, but training should start as soon as the real manifest is generated. Consequence: `scripts/wait-imdb-and-start-training.sh` polls `imdb_clean.jsonl`, runs the real 8-GPU config, and then runs final evaluation/export/benchmark automation.
- 2026-07-30: Add an idempotent IMDB pipeline ensure command. Rationale: the IMDB download is long-running and should be easy to recover without duplicating processes. Consequence: `scripts/ensure-imdb-pipeline-running.sh` checks the downloader and watcher and starts only missing pieces.
- 2026-07-30: Keep INT8 out of the default CPU deployment path after variant tuning. Rationale: QDQ, QOperator, per-channel, tensor-wise, S8/S8, U8/S8, and quant-preprocessed variants were benchmarked on the MobileNetV3-Small 112 candidate; the best preprocessed INT8 batch-128 result was `3,993.4 img/s`, while FP32 reaches `14,483.4 img/s`. Consequence: keep FP32 for throughput and revisit INT8 only with a different runtime/build or model architecture.
- 2026-07-30: Report IMDB-clean download progress against expected tar bytes. Rationale: segmented `aria2c` creates sparse files whose apparent sizes overstate progress. Consequence: `scripts/report-dataset-status.sh` now emits `raw_bytes_expected`, total `raw_progress`, and per-file `raw_progress_details` for the IMDB tar set; this was used while the tar download was still in progress.
- 2026-07-30: Require a minimum IMDB manifest size before launching training. Rationale: the automatic watcher should not spend 8 GPUs on a partial or bad manifest. Consequence: `scripts/wait-imdb-and-start-training.sh` and `scripts/ensure-imdb-pipeline-running.sh` default to `MIN_MANIFEST_ROWS=250000`.
- 2026-07-30: Validate IMDB tar completion before extraction. Rationale: interrupted segmented downloads can leave sparse files and `.aria2` control files. Consequence: `scripts/prepare-imdb-clean-images.sh` checks for missing control files and exact expected tar byte counts before extracting each tar.
- 2026-07-30: Add a single IMDB pipeline status entry point. Rationale: long jobs should be inspectable without ad hoc SSH command chains. Consequence: `scripts/imdb-pipeline-status.sh` reports manifest rows, process counts, disk space, dataset progress, and train log tail.
- 2026-07-30: Add transfer-rate and ETA fields to dataset status snapshots. Rationale: long downloads need evidence of current throughput, not only absolute progress. Consequence: `scripts/report-dataset-status.sh` compares against the previous `dataset-status.json` and emits `raw_rate_since_previous` plus per-tar `raw_rate_details_since_previous` when possible; the latest sample showed about 7.6 MB/s and about 7.7 hours remaining.
- 2026-07-30: Add a long-lived IMDB pipeline monitor. Rationale: the download-to-training handoff should keep self-healing and recording status without manual polling. Consequence: `scripts/monitor-imdb-pipeline.sh` periodically runs ensure plus status, writes latest and history logs, and exits after the IMDB run is finalized; `scripts/start-imdb-pipeline-monitor.sh` starts it idempotently.
- 2026-07-30: Pipeline IMDB extraction per completed tar. Rationale: waiting for all 10 large tar files before starting any extraction adds avoidable tail latency before manifest creation and training. Consequence: `scripts/prepare-imdb-clean-images.sh` defaults to `PIPELINE_EXTRACT=1`, downloading, validating, and extracting each tar in one per-part worker while retaining `PIPELINE_EXTRACT=0` for the old two-phase behavior; `scripts/ensure-imdb-pipeline-running.sh` passes the same default during self-healing restarts.
- 2026-07-30: Gate IMDB-expanded training on manifest validation. Rationale: row count alone does not prove a manifest is trainable. Consequence: `scripts/prepare-imdb-clean-images.sh` writes the IMDB manifest atomically via a temporary file, and `scripts/wait-imdb-and-start-training.sh` runs `scripts/validate-imdb-manifest.sh` before starting the 8-GPU run.
- 2026-07-30: Gate IMDB-expanded training on training/finalization idleness. Rationale: another full-machine challenger or its CPU benchmark finalizer may still be running when the IMDB manifest becomes ready. Consequence: `scripts/wait-imdb-and-start-training.sh` defaults to `WAIT_FOR_IDLE_TRAINING=1` and waits for existing training, export, benchmark, and model-card processes to exit before starting the IMDB 8-GPU run.
- 2026-07-30: Add compact IMDB tar progress summaries. Rationale: the monitor should show actionable timing without requiring humans to inspect all 10 tar entries. Consequence: `scripts/report-dataset-status.sh` now emits `raw_progress_summary.imdb_clean_tars` with total counts, closest/least-complete tar files, total rate and ETA, earliest per-tar ETA, and fastest/slowest active tar files.
- 2026-07-30: Add a torchvision model factory and complete EfficientNet-B0 128 as a real challenger. Rationale: MobileNetV3 remains the CPU default, but the user asked to evaluate whether alternatives can improve gender accuracy while staying CPU-deployable. Consequence: EfficientNet-B0 improves evaluation gender balanced accuracy to `0.93850`, but tuned FP32 batch-128 throughput is only `2,549.7 img/s`, far below MobileNetV3-Small 112 FP32 at `14,483.4 img/s`.
- 2026-07-30: Split IMDB-specific and global training process counts. Rationale: parallel challenger runs should not make the IMDB pipeline look as if IMDB training has started. Consequence: `scripts/imdb-pipeline-status.sh` now reports `imdb_training_processes` and `all_training_processes` separately.
- 2026-07-30: Complete ResNet18 128 as a sanity baseline. Rationale: ResNet18 gives a conventional CNN point in the architecture comparison. Consequence: it reaches evaluation gender balanced accuracy `0.93467`, above MobileNetV3-Large 128 and below EfficientNet-B0; tuned static INT8 batch-128 throughput is `4,163.2 img/s`, still far behind MobileNetV3-Small 112 FP32.
- 2026-07-30: Complete EfficientNet-B0 teacher distillation into MobileNetV3-Small 112. Rationale: a stronger teacher might improve the CPU student where MobileNetV3-Large112 teacher distillation did not. Consequence: evaluation gender balanced accuracy improves only slightly to `0.91051`, while tuned FP32 batch-128 throughput drops to `11,386.4 img/s`; keep the non-distilled small model as the throughput default.
- 2026-07-30: Do not promote MobileNetV3-Small 128 as the throughput default. Rationale: it reaches evaluation gender balanced accuracy `0.91028`, only about `0.00039` above MobileNetV3-Small 112, but tuned FP32 batch-128 throughput drops from `14,483.4` to `10,320.6 img/s`. Consequence: keep `MobileNetV3-Small 112 FP32` as the CPU throughput default while waiting for the IMDB-expanded run.
- 2026-07-30: Do not promote ConvNeXt-Tiny 128 as a deployment or teacher default. Rationale: it reaches evaluation gender balanced accuracy `0.93443`, below EfficientNet-B0 `0.93850` and ResNet18 `0.93467`; tuned FP32 batch-128 CPU throughput is only `298.4 img/s`. Consequence: keep ConvNeXt-Tiny as a completed negative challenger and continue using the strongest EfficientNet-family run as the teacher candidate.
- 2026-07-30: Promote EfficientNetV2-S 128 as the current public-data teacher candidate. Rationale: it reaches evaluation gender balanced accuracy `0.94594` and age MAE `4.87`, beating EfficientNet-B0, ResNet18, ConvNeXt-Tiny, and MobileNetV3-Large on the same FairFace + UTKFace validation setup. Consequence: use `runs/efficientnet_v2_s_128_real_fairface_utkface/best.pt` as the next teacher for MobileNetV3-Small distillation after IMDB or other data expansion is ready; do not use it as the CPU deployment default because tuned batch-128 throughput is only `1,051.8 img/s` on static INT8.
- 2026-07-30: Do not promote EfficientNetV2-S teacher distillation for MobileNetV3-Small 112. Rationale: the stronger teacher reduces the student to evaluation gender balanced accuracy `0.90711`, below the non-distilled small model `0.90989` and EfficientNet-B0 teacher distillation `0.91051`. Consequence: keep non-distilled MobileNetV3-Small 112 FP32 as the CPU throughput default and avoid stronger-teacher distillation until the data mix improves.
- 2026-07-30: Keep standard EfficientNetV2-S distillation as the best MobileNetV3-Large 128 variant so far. Rationale: it raises evaluation gender balanced accuracy from non-distilled Large 128 `0.92965` to `0.93092` while staying in the same CPU throughput class. Consequence: if the product chooses a higher-accuracy MobileNetV3 deployment over Small 112 throughput, prefer `runs/mobilenetv3_large128_distill_efficientnet_v2_s_fairface_utkface/best.pt` among completed MobileNetV3-Large 128 runs.
- 2026-07-30: Do not promote lower-weight EfficientNetV2-S distillation for MobileNetV3-Large 128. Rationale: reducing teacher weights to `0.05` gender and `0.025` age lands at evaluation gender balanced accuracy `0.92994`, below the standard V2-S distillation run. Consequence: do not spend more time lowering these weights on the current FairFace + UTKFace mix.
- 2026-07-30: Keep gender-only EfficientNetV2-S distillation as a completed MobileNetV3-Large gender variant, superseded later by the gender-priority run. Rationale: disabling age distillation and keeping gender teacher supervision reaches `0.93230` gender balanced accuracy, above standard V2-S distillation `0.93092` and non-distilled Large 128 `0.92965`. Consequence: prefer the later gender-priority Large128 run when gender accuracy is prioritized.
- 2026-07-30: Do not promote gender-only EfficientNetV2-S distillation for MobileNetV3-Small 112. Rationale: it reaches evaluation gender balanced accuracy `0.90677`, below regular V2-S distillation `0.90711` and the non-distilled Small112 baseline `0.90989`. Consequence: stop spending current FairFace + UTKFace runs on stronger-teacher Small112 distillation until the data mix changes.
- 2026-07-30: Derive automatic finalization input size from config/checkpoint. Rationale: `scripts/train-and-finalize.sh` previously defaulted `INPUT_SIZE=128`, which could export and benchmark 112-input runs incorrectly when not explicitly overridden. Consequence: `scripts/train-and-finalize.sh` reads `data.input_size` from the training config, and `scripts/finalize-model-run.sh` plus `scripts/export-and-benchmark.sh` read it from `best.pt` when unset.
- 2026-07-30: Do not promote gender-only EfficientNetV2-S distillation for MobileNetV3-Large 112 as a default. Rationale: it slightly improves Large112 gender balanced accuracy to `0.92895` from `0.92828`, with tuned FP32 batch-128 throughput `5,345.9` img/s, but still trails Large128 gender-priority accuracy `0.93376` and Small112 FP32 throughput `14,483.4` img/s. Consequence: keep it as a completed middle candidate only.
- 2026-07-30: Do not promote standard EfficientNetV2-S distillation for MobileNetV3-Large 112. Rationale: adding age distillation improves age MAE to `5.46`, but drops gender balanced accuracy to `0.92687`, below both non-distilled Large112 `0.92828` and gender-only Large112 `0.92895`. Consequence: keep age-distilled Large112 as an analysis artifact only; for gender-first MobileNetV3-Large candidates prefer the Large128 gender-priority run.
- 2026-07-30: Keep gender-priority EfficientNetV2-S distillation as the best completed MobileNetV3-Large gender variant. Rationale: increasing supervised gender loss weight to `4.0`, lowering age weight to `0.5`, and keeping gender-only teacher distillation reaches evaluation gender balanced accuracy `0.93376`, above the previous Large128 gender-only run `0.93230`. Consequence: use `runs/mobilenetv3_large128_distill_gender_priority_efficientnet_v2_s_fairface_utkface/best.pt` when MobileNetV3-Large gender accuracy is prioritized; keep MobileNetV3-Small 112 FP32 as the CPU throughput default.
- 2026-07-30: Do not promote Swin-T 128 as a teacher or deployment candidate on the current data mix. Rationale: it reaches only `0.92206` evaluation gender balanced accuracy, far below EfficientNetV2-S `0.94594`, EfficientNet-B0 `0.93850`, ResNet18 `0.93467`, ConvNeXt-Tiny `0.93443`, and the best MobileNetV3-Large run `0.93376`; default FP32 batch-128 CPU throughput is only `105.9` img/s. Consequence: keep Swin-T as a completed transformer negative challenger, and use `FINALIZE_THREAD_SWEEP=0` for similar slow negative runs after default CPU benchmarks are complete.
- 2026-07-30: Do not promote EfficientNet-B0 128 gender-priority as the teacher or CPU default. Rationale: raising supervised gender weight to `4.0` and lowering age weight to `0.5` improves B0 gender balanced accuracy from `0.93850` to `0.93910`, but still trails EfficientNetV2-S `0.94594`, worsens age MAE to `5.42`, and tuned FP32 batch-128 throughput is only `2,513.0` img/s. Consequence: keep it as the best completed B0 gender variant only.
- 2026-07-30: Promote EfficientNetV2-S 128 gender-priority as the current public-data teacher candidate. Rationale: raising supervised `gender_weight` to `4.0` and lowering `age_weight` to `0.5` improves evaluation gender balanced accuracy from regular V2-S `0.94594` to `0.94703`. Consequence: use `runs/efficientnet_v2_s_128_gender_priority_real_fairface_utkface/best.pt` as the strongest completed public-data teacher for future distillation, but do not use it as the CPU deployment default because default FP32 batch-128 throughput is only `885.1` img/s and default static INT8 batch-128 throughput is `841.0` img/s.
- 2026-07-30: Skip the full thread sweep for EfficientNetV2-S 128 gender-priority after default CPU benchmarks. Rationale: the regular V2-S run already has a same-architecture full sweep and the IMDB-expanded run needed the machine. Consequence: keep default CPU benchmark JSONs and `model_card.md`, and remove this run from `scripts/benchmark-thread-sweep.sh` defaults.
- 2026-07-30: Complete IMDB-clean acquisition and validate the manifest before training. Rationale: the first IMDB-expanded run should use complete original images with real face boxes, not a partial manifest. Consequence: `data/manifests/imdb_clean.jsonl` has 285,946 rows, 183,887 train rows, 102,059 validation rows, no missing images, no invalid bboxes, and no duplicate sample ids.
- 2026-07-30: Complete the natural-mix IMDB-expanded MobileNetV3-Small 112 run, but do not promote it as the default. Rationale: validated IMDB-clean adds much more exact-age/gender data while preserving the CPU-efficient deployment backbone, but the natural manifest mix lets IMDB-clean dominate validation and checkpoint selection. Consequence: `runs/mobilenetv3_small112_imdb_facecrop_real_fairface_utkface` reached mixed evaluation gender balanced accuracy `0.96994`, age MAE `5.95`, and tuned FP32 batch-128 throughput `12,748.3` img/s, but source-sliced FairFace gender balanced accuracy was only `0.90004`.
- 2026-07-30: Complete the source-balanced IMDB-expanded MobileNetV3-Small 112 rerun, but do not promote it as the default. Rationale: the natural IMDB run proved the data is useful but exposed celebrity-domain dominance, so the rerun capped IMDB-clean train rows at `90,000` and validation rows at `10,954` for checkpoint selection. Consequence: `runs/mobilenetv3_small112_imdb_source_balanced_facecrop_real_fairface_utkface` reached full-manifest mixed evaluation gender balanced accuracy `0.96553`, age MAE `6.28`, and tuned FP32 batch-128 throughput `12,579.9` img/s, but source-sliced FairFace gender balanced accuracy was still only `0.89950`.
- 2026-07-30: Complete the IMDB-pretrained MobileNetV3-Small 112 fine-tune on FairFace + UTKFace, but do not promote it as the default. Rationale: source caps did not restore FairFace performance, so this cheap student experiment kept the IMDB-pretrained representation but updated the checkpoint using only the public FairFace + UTKFace train/val distribution. Consequence: `runs/mobilenetv3_small112_imdb_pretrain_finetune_fairface_utkface` reached full-manifest mixed gender balanced accuracy `0.96784`, age MAE `6.21`, and tuned FP32 batch-128 throughput `13,050.6` img/s, but source-sliced FairFace gender balanced accuracy was still only `0.90024`.
- 2026-07-30: Start EfficientNetV2-S 128 on the source-balanced IMDB mix as the active teacher/challenger. Rationale: MobileNetV3-Small answers the CPU-throughput path, but the stronger V2-S model should measure whether the corrected IMDB mix raises the gender-first teacher ceiling. Consequence: `configs/train/efficientnet_v2_s_128_imdb_source_balanced_gender_priority_real.yaml` uses the same source caps, `gender_weight=4.0`, `age_weight=0.5`, and the watcher started the run at 2026-07-30 19:42:48 UTC after prior training/finalization idled.
- 2026-07-30: Complete EfficientNetV2-S 128 on the source-balanced IMDB mix, but do not make it the CPU default or the FairFace-robustness teacher. Rationale: it proves the mixed IMDB-inclusive teacher ceiling is much higher, but the full-manifest aggregate is still dominated by IMDB-clean and the FairFace slice does not beat the earlier public-data V2-S gender-priority teacher. Consequence: `runs/efficientnet_v2_s_128_imdb_source_balanced_gender_priority_real_fairface_utkface` reached mixed gender balanced accuracy `0.98605`, age MAE `4.84`, FairFace gender balanced accuracy `0.94386`, IMDB-clean `0.99138`, and UTKFace `0.95424`; keep it as an IMDB-inclusive teacher/challenger and continue using MobileNetV3-Small 112 FP32 as the CPU throughput default.
- 2026-07-30: Complete MobileNetV3-Small 112 source-balanced IMDB gender distillation from the completed IMDB V2-S teacher. Rationale: prior Small IMDB runs kept CPU throughput but did not recover the FairFace slice, so the next real run tested whether low-weight gender-only teacher logits plus supervised gender-priority loss could lift the student while retaining the 112-input Small deployment class. Consequence: `runs/mobilenetv3_small112_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface` reached mixed gender balanced accuracy `0.96800`, FairFace `0.90562`, IMDB-clean `0.97542`, UTKFace `0.94101`, age MAE `6.25`, and tuned FP32 batch-128 throughput `10,837.8` img/s; it is the strongest completed Small-family IMDB-inclusive gender candidate so far but not the pure throughput winner.
- 2026-07-30: Complete MobileNetV3-Large 128 source-balanced IMDB gender distillation from the completed IMDB V2-S teacher. Rationale: the Small112 distillation run proved teacher-guided IMDB source balancing can recover source robustness, so Large128 measures the next accuracy/CPU tradeoff tier. Consequence: `runs/mobilenetv3_large128_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface` reached mixed gender balanced accuracy `0.97929`, FairFace `0.92877`, IMDB-clean `0.98548`, UTKFace `0.95017`, age MAE `5.71`, and tuned FP32 batch-128 throughput `4,477.7` img/s; it is the strongest completed MobileNetV3 accuracy candidate but not the throughput winner.

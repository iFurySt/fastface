# GPU Environment

This document records the initial training machine setup for FastFace.

## Host

SSH target:

```sh
ssh <remote-gpu-host>
```

Observed on 2026-07-30:

- Hostname: `<gpu-hostname>`.
- GPUs: 8 x NVIDIA GPU accelerator.
- Driver: `535.129.03`.
- Existing conda root: `<conda-root>`.
- Project workspace: `<repo-root>`.

Storage:

Use an external workspace with enough space for raw archives, extracted images,
checkpoints, ONNX exports, and analysis outputs. Set `FASTFACE_WORK_ROOT` to that
workspace when it should live outside the repository checkout.

## Recommended Layout

Use `<repo-root>` for source and project metadata. Put large data under
`FASTFACE_WORK_ROOT` and symlink it into the repo if needed.

```text
<repo-root>
  data -> data
  models -> ${FASTFACE_WORK_ROOT}/models
  runs -> runs
  third_party -> third_party
```

## Conda Environment

Do not use the base Python environment for training. The observed base environment has missing packages and NumPy ABI conflicts.

Create a clean environment manually or by running `scripts/bootstrap-gpu-env.sh` on the GPU host. On 2026-07-30, direct conda package resolution failed with SSL errors, so the scripted setup cloned the existing `vlm` environment and then installed the missing packages with pip.

```sh
ssh <remote-gpu-host>

conda create -n faceattr python=3.11 -y
conda run -n faceattr python -m pip install --upgrade pip
conda run -n faceattr python -m pip install \
  torch torchvision --index-url https://download.pytorch.org/whl/cu128
conda run -n faceattr python -m pip install \
  timm==0.8.13.dev0 opencv-python pandas numpy scikit-learn tqdm albumentations \
  datasets huggingface_hub gdown kaggle onnx onnxruntime pillow pyyaml
```

Scripted setup:

```sh
cd <repo-root>
bash scripts/bootstrap-gpu-env.sh
```

Observed `faceattr` versions after bootstrap:

| Package | Version |
| --- | --- |
| `torch` | `2.9.0+cu128` |
| `torchvision` | `0.24.0+cu128` |
| `opencv-python` | `4.13.0` |
| `timm` | `0.8.13dev0` |
| `pandas` | `2.3.3` |
| `numpy` | `2.2.6` |
| `onnx` | `1.21.0` |
| `onnxruntime` | `1.26.0` |

Verify:

```sh
conda run -n faceattr python - <<'PY'
import torch, torchvision, cv2, timm, pandas, numpy, onnx, onnxruntime
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())
print("torchvision", torchvision.__version__)
print("cv2", cv2.__version__)
print("timm", timm.__version__)
print("numpy", numpy.__version__)
print("onnx", onnx.__version__)
print("onnxruntime", onnxruntime.__version__)
PY
```

## Download Workflow

Run the dataset bootstrap script from the GPU host:

```sh
cd <repo-root>
bash scripts/download-public-datasets.sh
```

Use remote logs:

```text
<repo-root>/logs/downloads/
```

Rules:

- Prefer resumable download commands.
- Keep raw archives and extracted data separate.
- Record every URL or dataset identifier in `data/manifests/downloads.jsonl`.
- If a source requires credentials or manual license acceptance, mark it as gated instead of bypassing the gate.
- If remote network is blocked, download to the local workstation and `scp` or `rsync` to `data/raw/`.

Observed on 2026-07-30:

- Remote direct downloads failed with TLS/SSL errors against Hugging Face, Google Drive, and GitHub.
- Local download plus `rsync` succeeded for FairFace `margin025`, FairFace labels, UTKFace cropped images, Lagenda-HF mirror images and annotations, MiVOLO IMDB annotations, and the IMDB-clean repository.
- MiVOLO's original Lagenda Google Drive image archive is about 50.6 GB and still has only a partial local download. The usable staged Lagenda source is the smaller Hugging Face mirror at `data/raw/lagenda_hf_uaebn`.
- The ETH IMDB-WIKI source is reachable from `<remote-gpu-host>`, but slow with single-connection `wget`. The completed IMDB-clean task used segmented `aria2c`; it wrote original tar files to `data/raw/imdb_clean/tars`, extracted to `data/raw/imdb_clean/images`, and built `data/manifests/imdb_clean.jsonl`. Observed completed state on 2026-07-30: 460,723 raw images, 285,946 valid manifest rows, `183,887` train rows, `102,059` validation rows, no missing images, no invalid bboxes, and no duplicate sample ids.

Rerun IMDB-clean preparation only if the manifest needs to be rebuilt:

```sh
cd <repo-root>
setsid env PROJECT_DIR=<repo-root> DATA_ROOT=data \
  DOWNLOAD_TOOL=aria2c DOWNLOAD_JOBS=10 ARIA2_CONNECTIONS=8 ARIA2_SPLIT=8 EXTRACT_JOBS=4 \
  bash scripts/prepare-imdb-clean-images.sh \
  > runs/imdb_clean_prepare/prepare.log \
  2>&1 < /dev/null &
```

The IMDB-clean preparation process writes its PID to:

```text
runs/imdb_clean_prepare/prepare.pid
```

Dataset status reports include actual allocated bytes, expected total bytes, total percent, per-tar progress, and a compact progress summary for IMDB-clean:

```sh
cd <repo-root>
bash scripts/report-dataset-status.sh
```

Before extraction, `scripts/prepare-imdb-clean-images.sh` verifies that the matching `.aria2` control file is gone and the tar apparent size matches the expected byte count. By default, `PIPELINE_EXTRACT=1` makes each tar extract as soon as that tar finishes downloading and passes validation, instead of waiting for all 10 tar files to finish first. Set `PIPELINE_EXTRACT=0` to use the older two-phase download-then-extract path. The IMDB manifest is written to a temporary path and moved into place atomically.

The IMDB-expanded training watcher uses `MIN_MANIFEST_ROWS=250000` by default before it launches the 8-GPU run. After the row threshold is met, it runs `scripts/validate-imdb-manifest.sh`, which checks required fields, train/val splits, `imdb-clean` dataset membership, age/gender ranges, bbox format, duplicate sample ids, and image path existence. The first validated IMDB-expanded MobileNetV3-Small 112 run started on 2026-07-30 17:55:48 UTC, completed 40 epochs, and finalized with mixed evaluation gender balanced accuracy `0.96994` and age MAE `5.95`.

The watcher also defaults `WAIT_FOR_IDLE_TRAINING=1`. After IMDB manifest validation passes, it waits until no other training or finalization process is active before starting the 8-GPU IMDB run. This prevents a late manifest handoff from colliding with another full-machine challenger run or its CPU benchmark finalization.

Use the single pipeline status entry point when checking a long-running IMDB job:

```sh
cd <repo-root>
bash scripts/imdb-pipeline-status.sh
```

It reports manifest row count, downloader/watcher PIDs, active `aria2c`, extraction, IMDB-specific training processes, all age/gender training processes, finalizer process counts, disk space, dataset progress, compact tar summary, transfer rate and ETA when a previous status snapshot exists, and the IMDB training log tail.

Idempotently ensure the downloader and training watcher are running:

```sh
cd <repo-root>
bash scripts/ensure-imdb-pipeline-running.sh
```

## Gender Disagreement Review

Run the converged candidate comparison on the GPU host with:

```sh
cd <repo-root>
OUTPUT_DIR=outputs/analysis/gender-comparison-current \
  bash scripts/compare-gender-models.sh
```

The command compares the current Large128 IMDB-distilled candidate, Small112
IMDB-distilled candidate, EfficientNetV2-S IMDB teacher, public FairFace-ONNX
baseline, and MiVOLO on FairFace validation, UTKFace validation, and a
seed-stable IMDB-clean validation sample. It writes `summary.json`,
`predictions.jsonl`, `gender_disagreements.csv`, focused disagreement CSVs, and
`gender_disagreements_top.jpg` under the output directory.

MiVOLO staging on the GPU host:

```text
third_party/MiVOLO
third_party/mivolo/weights/model_imdb_face_4.22_99.38.pth.tar
```

The checkpoint is the official MiVOLO face-only IMDB-clean age+gender model from
Google Drive file id `1NlsNEVijX2tjMe8LBb1rI56WB_ADVHeP`. Because
`<remote-gpu-host>` cannot reach Google Drive directly, it was downloaded locally and
copied to the GPU host. SHA256:

```text
3711b3530e94fa904fe3c9043dd9c5b73a13c8c347b435eb5eed89d97a8aaa4a
```

MiVOLO code requires `timm==0.8.13.dev0`; newer `timm 1.x` builds raise VOLO
constructor/import compatibility errors. The comparison adapter imports only the
MiVOLO model and preprocessing path, not the YOLO detector wrapper.

Build the manual review workbook from the public-vs-our focused disagreement CSV with:

```sh
cd <repo-root>
OUTPUT_DIR=outputs/analysis/manual-public-gender-review-current \
  bash scripts/build-manual-gender-review.sh
```

The workbook defaults to `focused/public_vs_our_large.csv` only. It embeds face-crop thumbnails, renames `image_path` to `image`, adds explicit `our_large_gender` and `public_fairface_gender` columns, and leaves `manual_gender` blank for human labels. Internal FastFace model disagreements are intentionally excluded from this default review.

Run the long-lived monitor when the IMDB download is expected to continue unattended:

```sh
cd <repo-root>
bash scripts/start-imdb-pipeline-monitor.sh
```

The monitor acquires a lock, calls `scripts/ensure-imdb-pipeline-running.sh`, then writes full status snapshots every `INTERVAL_SECONDS` seconds, default `600`. Logs and PID files are under `runs/imdb_pipeline_monitor/`.

## Training Workflow

Install the repository package after syncing code:

```sh
cd <repo-root>
conda run -n faceattr python -m pip install -e .
```

Build manifests for currently staged complete image datasets:

```sh
cd <repo-root>
bash scripts/build-training-manifests.sh
```

Start a real 8-GPU training run:

```sh
cd <repo-root>
setsid bash scripts/start-real-training.sh \
  > runs/mobilenetv3_real_fairface_utkface/train.log \
  2>&1 < /dev/null &
```

Start a real config-driven training run with automatic evaluation, ONNX export, CPU benchmark, thread sweep, and model card finalization:

```sh
cd <repo-root>
RUN_DIR=runs/efficientnet_b0_128_real_fairface_utkface
mkdir -p "${RUN_DIR}"
setsid env PROJECT_DIR=<repo-root> \
  CONFIG=<repo-root>/configs/train/efficientnet_b0_128_real.yaml \
  RUN_DIR="${RUN_DIR}" NPROC_PER_NODE=8 INPUT_SIZE=128 FINALIZE_AFTER_TRAIN=1 \
  bash scripts/train-and-finalize.sh \
  > "${RUN_DIR}/train.log" 2>&1 < /dev/null &
echo $! > "${RUN_DIR}/launcher.pid"
```

The `EfficientNet-B0 128` challenger was started this way and finalized on 2026-07-30 while IMDB-clean was still downloading. It reached evaluation gender balanced accuracy `0.93850`, with tuned FP32 CPU throughput `515.7`, `1,433.8`, `1,973.2`, and `2,549.7` img/s for batches `1`, `8`, `32`, and `128`.

The same entry point was also used for an `EfficientNet-B0 128` gender-priority challenger with supervised `gender_weight=4.0` and `age_weight=0.5`. It reached evaluation gender balanced accuracy `0.93910` and age MAE `5.42`; tuned FP32 CPU throughput was `500.8`, `1,374.3`, `2,013.7`, and `2,513.0` img/s for batches `1`, `8`, `32`, and `128`. This is the best completed B0 gender run, but it does not replace EfficientNetV2-S gender-priority as the teacher or MobileNetV3-Small 112 FP32 as the CPU throughput default.

The same entry point was also used for an `EfficientNetV2-S 128` gender-priority challenger with supervised `gender_weight=4.0` and `age_weight=0.5`. It reached evaluation gender balanced accuracy `0.94703` and age MAE `5.28`, making it the best completed public-data teacher candidate for the gender-first target. Default CPU throughput was `45.4`, `240.3`, `536.1`, and `885.1` img/s for FP32 batches `1`, `8`, `32`, and `128`; static INT8 reached `101.2`, `381.7`, `637.9`, and `841.0` img/s. The full thread sweep was stopped after default CPU benchmarks and model-card generation because the regular V2-S run already has a same-architecture full sweep and the IMDB-expanded run needed the machine.

The same entry point was used for the `ResNet18 128` challenger on 2026-07-30. It reached evaluation gender balanced accuracy `0.93467`; its best tuned CPU path was static INT8 with `1,156.1`, `3,323.3`, `4,084.8`, and `4,163.2` img/s for batches `1`, `8`, `32`, and `128`.

The same entry point was also used for a `MobileNetV3-Small 112` distillation run with the completed `EfficientNet-B0 128` checkpoint as teacher. It reached evaluation gender balanced accuracy `0.91051`, a very small improvement over the non-distilled small model, but tuned FP32 batch-128 throughput dropped to `11,386.4` img/s.

The same entry point was also used for a `MobileNetV3-Small 128` run on FairFace + UTKFace. It reached evaluation gender balanced accuracy `0.91028`, only a tiny improvement over `MobileNetV3-Small 112`, while tuned FP32 batch-128 throughput dropped to `10,320.6` img/s. Keep `MobileNetV3-Small 112 FP32` as the throughput default.

The same entry point was also used for a `ConvNeXt-Tiny 128` teacher/accuracy challenger. The first launch caused all DDP ranks to request the same torchvision weight file at once; the fix was to stop the launcher, remove partial cache files, verify `<torch-cache>/convnext_tiny-983f1562.pth` in a single process, then restart. The completed run reached evaluation gender balanced accuracy `0.93443` and age MAE `5.13`; tuned FP32 batch-128 CPU throughput was only `298.4` img/s, so this is not a deployment candidate and does not displace EfficientNet-B0 as the current accuracy candidate.

The same entry point was also used for a `Swin-T 128` transformer challenger after pre-warming `<torch-cache>/swin_t-704ceda3.pth`. Remote pre-warming was too slow and left a partial file, so the checkpoint was downloaded locally and copied into the remote torch cache. The completed run reached evaluation gender balanced accuracy `0.92206` and age MAE `5.53`; default FP32 batch-128 CPU throughput was only `105.9` img/s, and default static INT8 batch-128 throughput was `29.6` img/s. The full thread sweep was stopped because `model_fp32_threads1` had run for more than 8 minutes without completing, and the default benchmark already ruled the model out for CPU deployment. Use `FINALIZE_THREAD_SWEEP=0` for similar slow negative challengers after default CPU benchmark is complete.

The same entry point was also used for an `EfficientNetV2-S 128` teacher/accuracy challenger after pre-warming `<torch-cache>/efficientnet_v2_s-dd5fe13b.pth` in a single process. The completed run reached evaluation gender balanced accuracy `0.94594` and age MAE `4.87`, making it the current public-validation accuracy and teacher candidate. Tuned CPU throughput is still far below MobileNetV3-Small 112 FP32: static INT8 reaches `1,051.8` img/s at batch 128, while the throughput candidate reaches `14,483.4` img/s.

The completed `EfficientNetV2-S 128` gender-priority teacher challenger is stored at `runs/efficientnet_v2_s_128_gender_priority_real_fairface_utkface`. It keeps the proven V2-S backbone and training mix, raises supervised `gender_weight` to `4.0`, lowers `age_weight` to `0.5`, and used 8 GPUs via `scripts/train-and-finalize.sh`. It is now the current public-data teacher candidate for future distillation.

The same entry point was also used for a `MobileNetV3-Small 112` distillation run with the completed `EfficientNetV2-S 128` checkpoint as teacher. It reached evaluation gender balanced accuracy `0.90711`, below both the non-distilled small model and the EfficientNet-B0 teacher distillation run, so do not promote it.

The same entry point was also used for a `MobileNetV3-Small 112` gender-only distillation run with the completed `EfficientNetV2-S 128` checkpoint as teacher. It reached evaluation gender balanced accuracy `0.90677`, below the regular V2-S distillation run and the non-distilled small model, so do not promote it. This run also exposed a finalization bug: `scripts/train-and-finalize.sh` used to default `INPUT_SIZE=128` and override the finalizer default, which could export and benchmark Small112 runs with the wrong input size when `INPUT_SIZE` was not explicitly set. The script now derives `INPUT_SIZE` from the training config, while `scripts/finalize-model-run.sh` and `scripts/export-and-benchmark.sh` derive it from the checkpoint when unset.

The same entry point was also used for four `MobileNetV3-Large 128` distillation runs with the completed `EfficientNetV2-S 128` checkpoint as teacher. The standard low-weight run reached evaluation gender balanced accuracy `0.93092`, a small improvement over non-distilled Large 128 `0.92965`, with tuned FP32 batch-128 throughput `4,555.4` img/s. The lighter `0.05`/`0.025` teacher-weight run reached only `0.92994`, so do not promote the lighter run. A gender-only distillation run reached `0.93230`, with tuned FP32 batch-128 throughput `4,365.8` img/s, but age MAE regressed to `5.72`. A gender-priority loss variant reached the best completed MobileNetV3-Large gender balanced accuracy so far, `0.93376`, with age MAE `5.61` and tuned FP32 batch-128 throughput `4,376.9` img/s; use it only if gender accuracy is prioritized over MobileNetV3-Small throughput.

The same entry point was also used for a `MobileNetV3-Large 112` gender-only distillation run with the completed `EfficientNetV2-S 128` checkpoint as teacher. It reached evaluation gender balanced accuracy `0.92895`, slightly above non-distilled Large112 `0.92828`, with tuned FP32 batch-128 throughput `5,345.9` img/s. It is a completed middle candidate, but does not displace Large128 gender-priority for accuracy or Small112 FP32 for throughput.

The same entry point was also used for a `MobileNetV3-Large 112` standard distillation run with the completed `EfficientNetV2-S 128` checkpoint as teacher. It reached evaluation gender balanced accuracy `0.92687` and age MAE `5.46`; the age result improved, but gender fell below non-distilled Large112 and gender-only Large112, so do not promote it for the gender-first target.

Start or recover the real 8-GPU IMDB-expanded run automatically:

```sh
cd <repo-root>
RUN_DIR=runs/mobilenetv3_small112_imdb_facecrop_real_fairface_utkface
mkdir -p "${RUN_DIR}"
setsid env PROJECT_DIR=<repo-root> DATA_ROOT=data \
  CONFIG=<repo-root>/configs/train/mobilenetv3_small_112_imdb_real.yaml \
  RUN_DIR="${RUN_DIR}" NPROC_PER_NODE=8 POLL_SECONDS=300 \
  bash scripts/wait-imdb-and-start-training.sh \
  > "${RUN_DIR}/train.log" 2>&1 < /dev/null &
```

The active IMDB training watcher or launcher is recorded at:

```text
runs/mobilenetv3_small112_imdb_facecrop_real_fairface_utkface/launcher.pid
```

The watcher runs `scripts/finalize-model-run.sh` after training exits. The finalizer evaluates the best checkpoint, exports FP32 and static INT8 ONNX, runs default CPU benchmarks, runs the full thread sweep for both ONNX models, summarizes the sweep, and writes `model_card.md`.

The completed natural-mix IMDB-expanded run is stored at `runs/mobilenetv3_small112_imdb_facecrop_real_fairface_utkface`. It reached mixed evaluation gender balanced accuracy `0.96994`, but source-sliced FairFace gender balanced accuracy was only `0.90004`, so it is not promoted as the default. The completed source-balanced run is stored at `runs/mobilenetv3_small112_imdb_source_balanced_facecrop_real_fairface_utkface`; it used `configs/train/mobilenetv3_small_112_imdb_source_balanced_real.yaml` to cap IMDB-clean training rows at `90,000` and IMDB-clean validation rows at `10,954` for checkpoint selection. It reached full-manifest mixed evaluation gender balanced accuracy `0.96553`, but source-sliced FairFace was still only `0.89950`, so source caps alone do not fix the public FairFace regression. The completed IMDB-pretrained fine-tune is stored at `runs/mobilenetv3_small112_imdb_pretrain_finetune_fairface_utkface`; it initialized from the natural IMDB-expanded Small112 checkpoint, fine-tuned for 20 epochs on FairFace + UTKFace, and reached FairFace gender balanced accuracy `0.90024`, still below the original non-IMDB Small112 FairFace slice. The completed EfficientNetV2-S 128 source-balanced gender-priority teacher/challenger is stored at `runs/efficientnet_v2_s_128_imdb_source_balanced_gender_priority_real_fairface_utkface`; it reached full-manifest mixed gender balanced accuracy `0.98605`, age MAE `4.84`, and FairFace source-sliced gender balanced accuracy `0.94386`. It is useful as an IMDB-inclusive teacher/challenger, but it does not replace the earlier public-data V2-S gender-priority teacher for FairFace robustness and is not a CPU deployment default. The completed MobileNetV3-Small 112 IMDB source-balanced gender-distillation run is stored at `runs/mobilenetv3_small112_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface`; it reached mixed gender balanced accuracy `0.96800`, FairFace `0.90562`, IMDB-clean `0.97542`, UTKFace `0.94101`, and tuned FP32 batch-128 throughput `10,837.8` img/s, making it the strongest completed Small-family IMDB-inclusive gender candidate so far. The completed MobileNetV3-Large 128 IMDB source-balanced gender-distillation run is stored at `runs/mobilenetv3_large128_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface`; it reached mixed gender balanced accuracy `0.97929`, FairFace `0.92877`, IMDB-clean `0.98548`, UTKFace `0.95017`, and tuned FP32 batch-128 throughput `4,477.7` img/s, making it the strongest completed MobileNetV3 accuracy/CPU tradeoff candidate so far.

Candidate run artifacts:

```text
runs/mobilenetv3_real_fairface_utkface/
  config.resolved.yaml
  metrics.jsonl
  best.pt
  last.pt
  model_fp32.onnx
  model_int8_static.onnx
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
```

Observed first real run results on 2026-07-30:

| Artifact | Value |
| --- | --- |
| Training mix | FairFace + UTKFace |
| Epochs | 40 |
| Best checkpoint | `best.pt`, selected at epoch 34 |
| Validation gender balanced accuracy | `0.9296485426995538` |
| Validation gender accuracy | `0.9298416019127316` |
| Validation age MAE | `5.534534005155409` |
| FP32 ONNX size | 14 MB |
| Static INT8 ONNX size | 3.9 MB |

CPU benchmark on the GPU host with ONNX Runtime `CPUExecutionProvider`:

| Model | Batch 1 | Batch 8 | Batch 32 | Batch 128 |
| --- | ---: | ---: | ---: | ---: |
| FP32 ONNX | 196.4 img/s | 1,131.2 img/s | 2,986.3 img/s | 4,633.1 img/s |
| Static INT8 ONNX | 438.5 img/s | 952.5 img/s | 1,315.3 img/s | 1,350.2 img/s |

Static INT8 currently improves some single-image latency cases but hurts high-batch throughput. Treat the FP32 model as the current throughput winner.

After ONNX Runtime thread tuning, `MobileNetV3-Small 112 FP32` reaches 14,483 img/s at batch 128 and 1,468 img/s at batch 1 on the GPU host CPU. The tested distillation runs did not beat the non-distilled small model on the primary gender metric.

Lagenda-HF experiments showed that datasets with `bbox_face` metadata must crop the face box before resizing. The un-cropped Lagenda-HF run scored only `0.62406` gender balanced accuracy on the Lagenda validation slice; using `face_crop_margin: 0.2` raised that slice to `0.88850`, but the mixed metric still stayed below the FairFace + UTKFace small baseline.

INT8 tuning on `MobileNetV3-Small 112` tested QDQ, QOperator, per-channel, tensor-wise, S8/S8, U8/S8, and ONNX Runtime quant pre-processing. The best preprocessed INT8 batch-128 result was `3,993 img/s`, still far below the FP32 tuned result `14,483 img/s`.

`scripts/evaluate-checkpoint.sh` evaluates the manifests stored in each checkpoint config by default, so data-expanded runs keep their own source slices. `scripts/export-and-benchmark.sh` also reads checkpoint manifests and `face_crop_margin` for INT8 calibration unless overridden. See `docs/model-runs.md` for the complete large-vs-small, distillation, Lagenda-HF, INT8 tuning, and thread-sweep comparison.

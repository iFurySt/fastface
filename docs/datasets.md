# Datasets

This document records candidate datasets for the FastFace age and gender model.

## Policy

Use datasets only when the license and access terms allow the intended research, evaluation, or deployment use. Do not scrape arbitrary face images until legal, privacy, retention, and consent requirements are explicit.

Race labels are not part of the product output. If a dataset includes race labels, they may be used only for bias analysis or validation when permitted.

## Target Dataset Mix

The first training pass should use public or explicitly authorized datasets:

| Dataset | Primary Use | Notes |
| --- | --- | --- |
| FairFace | Gender training, age-range auxiliary signal, fairness validation | Balanced demographics; age labels are ranges, so it is not the main source for numeric age regression. |
| Lagenda | Age and gender training | Strong fit for the MiVOLO-style age/gender target; download links are public Google Drive links from the MiVOLO project. |
| IMDB-clean | Age and gender training | Large celebrity-oriented dataset; useful for age coverage but has celebrity/domain bias. |
| UTKFace | Small age/gender benchmark | Convenient labels; treat as research/evaluation unless license review approves broader use. |
| CelebA | Gender auxiliary training | Large celebrity face dataset; age labels are not direct numeric ages. |
| Adience | Benchmark and robustness evaluation | Age groups and gender; better as an evaluation set than numeric-age training. |

Datasets that usually require extra approval or payment:

| Dataset | Reason |
| --- | --- |
| MORPH-II | Useful for adult age estimation but commonly requires license approval. |
| AFAD | Useful for age/gender, especially Asian faces, but license/access terms need confirmation before inclusion. |

## Download Layout

Remote GPU workspace:

```text
<repo-root>
  data/
    raw/
    interim/
    processed/
    manifests/
  models/
  runs/
  third_party/
```

Raw downloads must stay immutable. Processing scripts should write derived manifests and aligned crops under `data/interim/` or `data/processed/`.

## Dataset Status

| Dataset | Current Status | Download Route |
| --- | --- | --- |
| FairFace | Staged on GPU host | Official Google Drive `margin025` archive and train/val labels were downloaded locally, copied to `data/raw/fairface`, and extracted on `<remote-gpu-host>`. Counts observed on 2026-07-30: 86,744 train images and 10,954 val images. |
| Lagenda | Hugging Face mirror staged on GPU host | `uaebn/lagenda` was downloaded locally through `hf download` with the Hugging Face mirror endpoint, copied to `data/raw/lagenda_hf_uaebn`, and verified complete for the mirror. Counts observed on 2026-07-30: 9,999 JPEG images and 12,409 manifest rows. MiVOLO's original Google Drive image archive remains incomplete locally. |
| IMDB-clean | Staged and manifest validated on GPU host | The IMDB-clean repository is staged under `third_party/imdb-clean`; MiVOLO IMDB annotations are staged at `data/raw/imdb_clean/mivolo_imdb_annotations.zip`. The 10 original IMDB-WIKI tar files were downloaded with segmented `aria2c`, extracted to `data/raw/imdb_clean/images`, and converted into `data/manifests/imdb_clean.jsonl`. Counts observed on 2026-07-30: 460,723 raw images and 285,946 valid age/gender rows after manifest validation. Kaggle mirrors may be faster for future rebuilds, but `KAGGLE_USERNAME` and `KAGGLE_KEY` are not configured on the GPU host. |
| UTKFace | Staged on GPU host from a public Hugging Face mirror | `py97/UTKFace-Cropped` was downloaded locally, copied to `data/raw/utkface`, and extracted. Counts observed on 2026-07-30: 23,708 images, 23,705 manifest rows after filename parsing. |
| CelebA | Gated or mirror-dependent | Source selection and license/access confirmation are still required. |
| Adience | Pending | Official route still needs scripted confirmation and license review before production use. |

Remote direct downloads failed on 2026-07-30 because the GPU host had TLS/SSL failures against Hugging Face, Google Drive, and GitHub. The working route was local download followed by `rsync` to `${FASTFACE_WORK_ROOT}`.

## Current Training Manifests

Available training manifests on the GPU host:

| Manifest | Rows | Label Type |
| --- | ---: | --- |
| `data/manifests/fairface.jsonl` | 97,698 | Gender exact, age range weak supervision. |
| `data/manifests/utkface.jsonl` | 23,705 | Gender exact, age exact from filename. |
| `data/manifests/lagenda_hf_uaebn.jsonl` | 12,409 | Gender exact, age exact from Lagenda annotation CSV, with `bbox_face` for required face cropping. |
| `data/manifests/imdb_clean.jsonl` | 285,946 | Gender exact, age exact, bbox_face for face cropping. |

The current default throughput family remains `MobileNetV3-Small 112 FP32`. The completed natural-mix data-expanded run is `runs/mobilenetv3_small112_imdb_facecrop_real_fairface_utkface`; it trains on FairFace + UTKFace + validated IMDB-clean with manifest `bbox_face` cropping. It is not promoted as the default because the mixed aggregate is dominated by IMDB-clean and FairFace validation regresses. The completed source-balanced follow-up is `runs/mobilenetv3_small112_imdb_source_balanced_facecrop_real_fairface_utkface`; it caps IMDB-clean rows during training and checkpoint-selection validation, but final full-manifest evaluation still leaves FairFace gender balanced accuracy at only `0.89950`. The completed IMDB-pretrained fine-tune is `runs/mobilenetv3_small112_imdb_pretrain_finetune_fairface_utkface`; it initializes from the natural IMDB-expanded checkpoint and fine-tunes on FairFace + UTKFace only, but still lands at only `0.90024` FairFace gender balanced accuracy. The completed EfficientNetV2-S IMDB source-balanced gender-priority run is `runs/efficientnet_v2_s_128_imdb_source_balanced_gender_priority_real_fairface_utkface`; it is a strong IMDB-inclusive teacher/challenger with mixed gender balanced accuracy `0.98605`, but its FairFace slice `0.94386` is slightly below the earlier public-data V2-S gender-priority FairFace slice `0.94586`. The completed Small112 IMDB source-balanced gender-distillation run is `runs/mobilenetv3_small112_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface`; it uses that V2-S checkpoint as a low-weight gender teacher and improves Small-family FairFace to `0.90562`, IMDB-clean to `0.97542`, and UTKFace to `0.94101`, while keeping tuned FP32 batch-128 throughput at `10,837.8` img/s. The completed Large128 IMDB source-balanced gender-distillation run is `runs/mobilenetv3_large128_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface`; it is the strongest completed MobileNetV3 accuracy/CPU tradeoff candidate with mixed gender balanced accuracy `0.97929`, FairFace `0.92877`, IMDB-clean `0.98548`, UTKFace `0.95017`, and tuned FP32 batch-128 throughput `4,477.7` img/s. Lagenda-HF is staged and usable for experiments, but runs that include it must use manifest `bbox_face` cropping before resize. The un-cropped Lagenda run failed badly on Lagenda validation because the raw images are Open Images-style full scenes, not aligned face crops.

Generate the current raw/manifest summary on the GPU host with:

```sh
cd <repo-root>
bash scripts/report-dataset-status.sh
```

For IMDB-clean, `raw_bytes.imdb_clean_tars` reports actual allocated disk blocks. `raw_bytes_expected.imdb_clean_tars` and `raw_progress.imdb_clean_tars.percent` report progress against the expected 10-tar total. `raw_progress_details.imdb_clean_tars` breaks progress down by individual tar file. `raw_progress_summary.imdb_clean_tars` condenses that detail into completed/active/pending counts, closest and least-complete tar files, total transfer rate, total ETA, earliest per-tar ETA, and fastest/slowest active tar files when rate data exists. When `--output` points at an existing prior status JSON, `raw_rate_since_previous.imdb_clean_tars` and `raw_rate_details_since_previous.imdb_clean_tars` report byte deltas, transfer rates, remaining bytes, and ETA. This avoids over-counting sparse files created by segmented `aria2c` downloads.

The IMDB-expanded training watcher requires at least `250000` manifest rows before launching the 8-GPU run. This keeps a partial or bad IMDB manifest from accidentally starting a full training run.

After the row threshold is met, `scripts/validate-imdb-manifest.sh` must pass before training starts. It validates required JSONL fields, train/val splits, the `imdb-clean` dataset label, age/gender ranges, bbox format, duplicate sample ids, and image path existence. The current IMDB-clean manifest passed this validation before the completed natural-mix IMDB-expanded run and the completed source-balanced rerun started.

Continue or restart the resumable IMDB-clean image acquisition on the GPU host with:

```sh
cd <repo-root>
setsid bash scripts/prepare-imdb-clean-images.sh \
  > runs/imdb_clean_prepare/prepare.log \
  2>&1 < /dev/null &
```

The current GPU-host run uses:

```sh
DOWNLOAD_TOOL=aria2c DOWNLOAD_JOBS=10 ARIA2_CONNECTIONS=8 ARIA2_SPLIT=8
```

## Manifest Requirements

Every processed example should be represented in a manifest with:

- Stable sample id.
- Source dataset.
- Raw image path.
- Processed crop path.
- Gender label and label source.
- Numeric age label when available.
- Age range label when only a range is available.
- Split assignment.
- License/access class.
- Optional teacher outputs.
- Optional quality metadata such as face detector score, pose, blur, crop size, and occlusion flags.

## Label Rules

Gender:

- Normalize labels into `female=0`, `male=1`.
- Preserve original labels in the manifest for auditability.
- Drop ambiguous or missing gender labels from supervised gender training unless explicitly reviewed.

Age:

- Prefer exact numeric age labels for primary age training.
- Convert exact ages into soft `0..100` distributions during training.
- Do not convert broad public age ranges into fake exact ages for primary supervision.
- Age ranges may be used for auxiliary consistency checks or weak supervision with lower loss weight.

## References

- FairFace paper: https://arxiv.org/abs/1908.04913
- FairFace repository: https://github.com/dchen236/FairFace
- MiVOLO repository and Lagenda links: https://github.com/WildChlamydia/MiVOLO
- Lagenda page: https://wildchlamydia.github.io/lagenda/
- Lagenda Hugging Face mirror: https://huggingface.co/datasets/uaebn/lagenda
- IMDB-clean: https://github.com/yiminglin-ai/imdb-clean
- UTKFace: https://susanqq.github.io/UTKFace/
- CelebA: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- Adience: https://talhassner.github.io/home/projects/Adience/Adience-data.html

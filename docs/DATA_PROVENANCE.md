# Data Provenance

This document records the datasets used or staged during FastFace phase 1. It is
intended to make the training data transparent without redistributing raw
third-party images from this repository.

## Policy

- Do not commit raw face images to GitHub.
- Do not upload third-party raw datasets unless the upstream license explicitly
  allows redistribution.
- Keep dataset acquisition scripts, manifest builders, row counts, and source
  notes in this repository.
- Keep large raw data under `data` on the GPU host.
- Use public datasets only for the allowed research/evaluation/deployment scope
  after license review.

## Staged Sources

| Dataset | Status | Rows | Raw/Staging Path | Notes |
| --- | --- | ---: | --- | --- |
| FairFace | complete | 97,698 | `data/raw/fairface` | Official `margin025` archive and train/val labels. |
| UTKFace | complete | 23,705 | `data/raw/utkface` | Public Hugging Face mirror of cropped images. |
| IMDB-clean | complete | 285,946 | `data/raw/imdb_clean` and `third_party/imdb-clean` | Original IMDB-WIKI images plus MiVOLO IMDB annotations and face boxes. |
| Lagenda-HF | staged | 12,409 | `data/raw/lagenda_hf_uaebn` | Hugging Face mirror; requires `bbox_face` crop. Not promoted in phase 1. |

## Manifest Paths

```text
data/manifests/fairface.jsonl
data/manifests/utkface.jsonl
data/manifests/imdb_clean.jsonl
data/manifests/lagenda_hf_uaebn.jsonl
```

## Source Notes

### FairFace

- Source: FairFace repository and paper.
- Purpose: gender training, public validation, fairness-oriented source slice.
- Labels: exact gender; age range.
- Phase-1 issue: FairFace age labels are ranges, so age MAE is only directional.

### UTKFace

- Source: public UTKFace cropped image mirror.
- Purpose: exact-age and gender training/validation.
- Labels: gender and exact age parsed from filename.
- Phase-1 issue: license/redistribution must be reviewed before any broader
  product use.

### IMDB-clean

- Sources:
  - IMDB-clean repository staged under `third_party/imdb-clean`.
  - MiVOLO IMDB annotations staged as
    `data/raw/imdb_clean/mivolo_imdb_annotations.zip`.
  - Original IMDB-WIKI tar files downloaded from ETH with segmented `aria2c`.
- Purpose: exact-age/gender data expansion and teacher/student training.
- Labels: exact age, exact gender, face boxes.
- Completed state:
  - 460,723 raw images.
  - 285,946 valid manifest rows.
  - 183,887 train rows.
  - 102,059 validation rows.
  - No missing images, invalid bboxes, or duplicate sample ids after validation.
- Phase-1 issue: celebrity-heavy distribution can dominate mixed metrics.

### Lagenda-HF

- Source: `uaebn/lagenda` Hugging Face mirror.
- Purpose: staged exact-age/gender data; not promoted.
- Labels: exact age/gender and `bbox_face`.
- Phase-1 issue: raw images are full scenes; training without face cropping
  failed badly on Lagenda validation.

## Rebuild Commands

Download/stage public datasets:

```sh
cd <repo-root>
bash scripts/download-public-datasets.sh
```

Build manifests:

```sh
cd <repo-root>
bash scripts/build-training-manifests.sh
```

Validate IMDB-clean:

```sh
cd <repo-root>
bash scripts/validate-imdb-manifest.sh
```

Report dataset state:

```sh
cd <repo-root>
bash scripts/report-dataset-status.sh
```

## References

- FairFace paper: https://arxiv.org/abs/1908.04913
- FairFace repository: https://github.com/dchen236/FairFace
- MiVOLO repository: https://github.com/WildChlamydia/MiVOLO
- IMDB-clean repository: https://github.com/yiminglin-ai/imdb-clean

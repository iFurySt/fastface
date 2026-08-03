# Avatar Labeling Workflow

This workflow builds pseudo labels for the local avatar image corpus by model
agreement.

Source dataset:

```text
~/datasets/avatar_merged_20260730
```

The dataset contains raw avatar images plus `metadata.csv` with blank `gender`
and `age` columns. Some images are faces and some are not.

## Labeling Rule

The workflow uses one shared face/alignment step before comparing attribute
models:

1. Run the owned `fastfacedetector` on the raw image.
2. Accept only single-face images for automatic pseudo labeling.
3. Align the face with 5-point landmarks.
4. Run FastFace Large and public FairFace ONNX on the same aligned crop.
5. If both models predict the same gender and both confidence scores are at or
   above the threshold, write the row to `accepted`.
6. Write disagreements, low-confidence rows, invalid images, and multi-face rows
   to `review`.
7. Write detector no-face rows to `no_face`.

The default confidence threshold is `0.80`.

## Setup

Use a Python environment with:

- `onnxruntime`
- `opencv-python`
- `pillow`
- `openpyxl` when `--review-xlsx` is used
- `uniface[cpu]` for the shared landmark alignment helper

The public FairFace ONNX baseline can be downloaded from:

```text
https://github.com/yakhyo/fairface-onnx/releases/download/weights/fairface.onnx
```

The default wrapper expects it at:

```text
outputs/models/public/fairface.onnx
```

## Smoke Test

```sh
PYTHON_BIN=.venv-avatar-label/bin/python \
bash scripts/label-avatar-dataset.sh \
  --split validation \
  --max-images 50 \
  --batch-size 16 \
  --review-xlsx \
  --output-dir outputs/avatar-labeling-smoke
```

Observed smoke-test output:

| Bucket | Rows |
| --- | ---: |
| accepted | 13 |
| review | 16 |
| no_face | 21 |

The generated review workbook had 16 review rows plus the header row.

## Full Run

```sh
PYTHON_BIN=.venv-avatar-label/bin/python \
bash scripts/label-avatar-dataset.sh \
  --batch-size 64 \
  --review-xlsx
```

Optional split-only runs:

```sh
PYTHON_BIN=.venv-avatar-label/bin/python \
bash scripts/label-avatar-dataset.sh \
  --split validation \
  --batch-size 64 \
  --review-xlsx
```

## Outputs

Default output directory:

```text
outputs/avatar-labeling
```

Files:

- `accepted.jsonl` / `accepted.csv`: single-face model-agreement pseudo labels.
- `review.jsonl` / `review.csv`: rows requiring manual review.
- `no_face.jsonl` / `no_face.csv`: detector no-face rows.
- `review.xlsx`: optional image-embedded workbook for manual labeling.
- `review_images/`: thumbnails used by `review.xlsx`.
- `summary.json`: counts, rates, model paths, and output paths.

The `review.xlsx` workbook has `manual_gender` values:

- `female`
- `male`
- `unclear`
- `not_face`

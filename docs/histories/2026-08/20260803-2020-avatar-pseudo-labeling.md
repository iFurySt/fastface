## [2026-08-03 20:20] | Task: Avatar Pseudo-Labeling Workflow

### Execution Context

- Agent ID: `traecode`
- Base Model: `GPT-5`
- Runtime: `local macOS shell`

### User Query

> Build a local workflow for `~/datasets` avatar images: use
> public and FastFace models to auto-accept matching results and output a file
> for human review when results differ.

### Changes Overview

- Area: evaluation and dataset labeling.
- Key actions:
  - Added a batch avatar pseudo-labeling command.
  - Added a wrapper script with default local dataset/model paths.
  - Documented the agreement/disagreement workflow and output files.
  - Verified a 50-image validation smoke test.

### Design Intent

The workflow uses the owned FastFace detector as the shared face/alignment step
so FastFace Large and public FairFace ONNX are compared on the same aligned
crop. Rows are auto-accepted only when the image has exactly one detected face,
both models predict the same gender, and both confidence scores meet the
threshold. No-face, multi-face, low-confidence, and disagreement rows are routed
to review so manual labeling can build a cleaner future benchmark dataset.

The smoke test on 50 validation images produced 13 accepted rows, 16 review
rows, and 21 no-face rows, and generated an image-embedded review workbook.

### Files Modified

- `packages/fastface/evaluation/label_avatar_dataset.py`
- `scripts/label-avatar-dataset.sh`
- `docs/AVATAR_LABELING.md`
- `README.md`

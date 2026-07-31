## [2026-07-31 10:54] | Task: Phase-1 Repo And Release Cleanup

### Execution Context

- Agent ID: `codex`
- Base Model: `GPT-5`
- Runtime: `Codex CLI`

### User Query

> Treat phase 1 as complete, clean useless GitHub/docs template files, update AGENTS/README/docs, freeze model versions, keep large model weights outside GitHub, publish to Hugging Face, create cards, and document training/data provenance in a transparent technical-report style.

### Changes Overview

- Area: repository cleanup, documentation, model release policy, artifact hygiene.
- Key actions:
  - Removed stale GitHub issue/PR templates and the template project-initialization script.
  - Replaced template docs with FastFace-specific technical report, data provenance, model release, and model-card source docs.
  - Updated `AGENTS.md`, `README.md`, `CONTRIBUTING.md`, `SECURITY.md`, `Makefile`, CI/CD posture, quality scoring, and design-doc index for the phase-1 FastFace project.
  - Tightened `.gitignore` so large root-level artifacts are ignored without hiding package source directories.
  - Archived the training bootstrap execution plan as completed.
  - Staged and uploaded the frozen `fastface-v0.1.0` model release to Hugging Face.
  - Hardened GPU source sync so `rsync --delete` does not remove remote data, run, model, third-party, or log directories.

### Design Intent

GitHub is kept as a source, config, and documentation repository. Frozen checkpoints, ONNX exports, release manifests, and benchmark artifacts belong in the Hugging Face model repository so the GitHub repo stays lightweight and reproducible.

### Files Modified

- `.github/`
- `.gitignore`
- `AGENTS.md`
- `README.md`
- `CODEOWNERS`
- `CONTRIBUTING.md`
- `Makefile`
- `SECURITY.md`
- `docs/CICD.md`
- `docs/QUALITY_SCORE.md`
- `docs/REPO_COLLAB_GUIDE.md`
- `docs/TECHNICAL_REPORT.md`
- `docs/DATA_PROVENANCE.md`
- `docs/MODEL_RELEASE.md`
- `docs/cards/fastface-hf-model-card.md`
- `docs/exec-plans/completed/faceattr-training-bootstrap.md`
- `scripts/stage-hf-release.sh`

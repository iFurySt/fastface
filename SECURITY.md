# Security Policy

## Reporting

Do not open a public issue for a suspected vulnerability.

Instead, report it through the private security channel your organization uses and include:

- Affected area and impact.
- Reproduction steps or proof of concept.
- Known mitigations or workarounds.

## Scope

This repository contains FastFace training, evaluation, export, and release automation. The GitHub repository should contain source, configs, documentation, and lightweight metadata only.

Out of scope for GitHub:

- Raw datasets and processed face crops.
- Checkpoints, ONNX exports, and calibration bundles.
- Manual review workbooks that embed source images.
- Secrets, local credentials, or host-specific tokens.

Frozen model artifacts belong in the Hugging Face model repository described in `docs/MODEL_RELEASE.md`, with checksums and transparent data provenance.

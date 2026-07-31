## [2026-07-31 11:04] | Task: Apache License And Repository Topics

### Execution Context

- Agent ID: `codex`
- Base Model: `GPT-5`
- Runtime: `Codex CLI`

### User Query

> If the license is Apache II, unify both sides and set GitHub About topics.

### Changes Overview

- Area: license metadata and repository presentation.
- Key actions:
  - Added a standard Apache-2.0 `LICENSE` file.
  - Updated README, package metadata, and model-release docs to state Apache-2.0.
  - Kept the Hugging Face model card license aligned with `license: apache-2.0`.
  - Uploaded `LICENSE` to the Hugging Face model repository.
  - Set GitHub repository topics and homepage for FastFace discoverability.

### Design Intent

Use one explicit license across source code and frozen model artifacts while keeping upstream dataset license obligations separate.

### Files Modified

- `LICENSE`
- `README.md`
- `pyproject.toml`
- `docs/MODEL_RELEASE.md`
- `docs/histories/2026-07/20260731-1104-apache-license-and-topics.md`

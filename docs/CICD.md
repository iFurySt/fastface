# CI/CD Guide

FastFace intentionally does not ship GitHub Actions yet.

## Current State

- There are no default workflows under `.github/workflows/`.
- The repository has no workflows under `.github/workflows/`.
- Local validation is `make check`.
- Model artifacts are released through Hugging Face, not GitHub Actions.

## Design Principle

CI/CD should protect the real training and release workflow without pulling large artifacts into GitHub.

The first useful workflow should run repository validation only: Python compile checks, shell syntax checks, and YAML config parsing. Later workflows can add unit tests, ONNX export parity, and benchmark smoke tests using small fixtures. Pin new GitHub Actions to commit SHAs instead of floating tags.

## Recommended Customization Sequence

1. Add a minimal pull-request gate for `make check`.
2. Add parser/unit tests for manifests, labels, and model factory construction.
3. Add ONNX export and PyTorch-vs-ONNX parity checks on a tiny fixture.
4. Add release checks that verify HF staging manifests and SHA256 files.
5. Keep heavyweight benchmarks and full training on the GPU host.

## When Adding CI/CD Back

- Do not commit checkpoints, ONNX files, datasets, or review workbooks.
- Do not expose stale or unmaintained commands in `Makefile`.
- If release automation is added, update `docs/MODEL_RELEASE.md` in the same change.

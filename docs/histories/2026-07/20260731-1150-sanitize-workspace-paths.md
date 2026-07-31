## [2026-07-31 11:50] | Task: Sanitize workspace paths before publish

### Execution Context

- Agent ID: `TRAE CLI`
- Base Model: `GPT-5`
- Runtime: `TraeCode`

### User Query

> Remove repository references to the concrete GPU SSH target and workspace
> path, rebuild commits, and force-push so those details are not leaked.

### Changes Overview

- Area: repository hygiene, scripts, configs, and documentation.
- Key actions:
  - Replaced concrete machine and workspace paths with repository-relative
    paths, environment variables, or placeholders.
  - Made training/evaluation/export paths resolve environment variables and
    repository-relative paths.
  - Removed remote host defaults from release staging and sync commands so users
    must pass deployment-specific locations explicitly.
  - Kept large artifacts outside Git while documenting `FASTFACE_WORK_ROOT` as
    the configurable external workspace.

### Design Intent

Repository files should remain reproducible without embedding private machine
names, user directories, or storage mount paths. Scripts now default to paths
relative to the checkout, while operators can set environment variables for a
real training workspace.

### Files Modified

- `configs/train/*.yaml`
- `packages/fastface/paths.py`
- `packages/fastface/training/train_age_gender.py`
- `packages/fastface/evaluation/*.py`
- `packages/fastface/export/*.py`
- `scripts/*.sh`
- `docs/*.md`
- `docs/exec-plans/completed/faceattr-training-bootstrap.md`
- `Makefile`

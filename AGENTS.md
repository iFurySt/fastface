# fastface

FastFace is a CPU-oriented face age/gender model project. The product model
outputs gender and numeric age only; race prediction is out of scope.

`AGENTS.md` stays short on purpose. Treat it as a map, not the encyclopedia.
Repository-local markdown under `docs/` is the system of record.

If a code or workflow change makes a doc stale, update the doc in the same task.

## Read At The Start Of Each Task

- `docs/REPO_COLLAB_GUIDE.md`: repository-wide collaboration, commit, documentation, and testing expectations.
- `docs/ARCHITECTURE.md`: top-level architecture map and intended package boundaries.
- `docs/TECHNICAL_REPORT.md`: frozen phase-1 training, data, evaluation, and release report.
- `docs/design-docs/core-beliefs.md`: agent-first operating principles and repository design intent.

## Read Before Finishing A Code Change

- `docs/HISTORY_GUIDE.md`: when to record code changes, naming rules, and redaction rules.
- `docs/QUALITY_SCORE.md`: current quality targets and gaps by area.

## Read When The Task Needs It

- `docs/PLANS_GUIDE.md`: when to create an execution plan and how to maintain it.
- `docs/DATA_PROVENANCE.md`: dataset sources, access class, staging paths, and license cautions.
- `docs/MODEL_RELEASE.md`: Hugging Face model hosting, frozen artifact layout, and release checklist.
- `docs/GPU_ENVIRONMENT.md`: GPU host setup, data layout, and long-running job commands.
- `docs/model-runs.md`: completed training runs, metrics, CPU benchmarks, and disagreement review outputs.
- `docs/CICD.md`: current no-CI stance and future real checks.
- `CONTRIBUTING.md`: pull request expectations and default review checklist.

## Working Rules

- Prefer small, explicit, repository-legible abstractions.
- Keep prompts, policies, and architectural rules versioned in-repo.
- For complex work, create an execution plan instead of relying on long chat context.
- Record finished code changes in `docs/histories/`.
- Do not commit raw datasets, checkpoints, ONNX files, or review workbooks. Large artifacts live under `${FASTFACE_WORK_ROOT}` during training and Hugging Face for releases.

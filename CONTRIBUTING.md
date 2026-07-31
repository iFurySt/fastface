# Contributing

This repository is designed for agent-first FastFace development, but the same rules apply to humans and bots.

## Working Agreement

- Start from `AGENTS.md`, then read the linked docs that match the task.
- Keep repository knowledge in versioned files, not only in chat or ticket comments.
- If behavior changes, update code, docs, tests, and release/history records together.
- For large or risky work, create an execution plan under `docs/exec-plans/active/`.

## Before Opening A Pull Request

- Run `make check` for repository changes, plus the narrow training/evaluation command that matches the change.
- Add or update a history entry if the task changed repository code or workflow.
- Update `docs/MODEL_RELEASE.md`, model cards, or technical-report sections when the change affects released weights.
- Verify examples and scripts still match the current behavior.
- Do not commit raw datasets, generated crops, checkpoints, ONNX files, zipped review workbooks, or local `outputs/`.

## Review Expectations

- Prefer small, scoped pull requests.
- Call out risks, migrations, and deferred follow-ups explicitly.
- Link to the relevant plan, spec, or history file when context is important.

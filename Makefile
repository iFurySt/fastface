SLUG ?=
REMOTE ?=
REMOTE_DIR ?=

.PHONY: check sync-gpu stage-hf-release compare-gender manual-review new-history new-plan

check:
	PYTHONPATH=packages python -m compileall -q packages
	bash -n scripts/*.sh
	PYTHONPATH=packages python -c 'import pathlib, yaml; [yaml.safe_load(path.open("r", encoding="utf-8")) for path in pathlib.Path("configs").rglob("*.yaml")]; print("validated yaml configs")'

sync-gpu:
	@if [ -z "$(REMOTE)" ] || [ -z "$(REMOTE_DIR)" ]; then echo "usage: make sync-gpu REMOTE=<host> REMOTE_DIR=<path>"; exit 1; fi
	rsync -az --delete \
		--exclude '.git/' \
		--exclude '.cache/' \
		--exclude 'outputs/' \
		--exclude 'data/' \
		--exclude 'runs/' \
		--exclude 'models/' \
		--exclude 'third_party/' \
		--exclude 'logs/' \
		--exclude '__pycache__/' \
		./ "$(REMOTE):$(REMOTE_DIR)/"

stage-hf-release:
	./scripts/stage-hf-release.sh

compare-gender:
	./scripts/compare-gender-models.sh

manual-review:
	./scripts/build-manual-gender-review.sh

new-history:
	@if [ -z "$(SLUG)" ]; then echo "usage: make new-history SLUG=my-change"; exit 1; fi
	./scripts/new-history.sh "$(SLUG)"

new-plan:
	@if [ -z "$(SLUG)" ]; then echo "usage: make new-plan SLUG=my-plan"; exit 1; fi
	./scripts/new-exec-plan.sh "$(SLUG)"

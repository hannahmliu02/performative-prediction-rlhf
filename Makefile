UV := $(HOME)/.local/bin/uv
export VIRTUAL_ENV=

.PHONY: fmt lint test smoke

fmt:
	$(UV) run ruff format llm

lint:
	$(UV) run ruff check llm

test:
	$(UV) run pytest llm/tests -v; s=$$?; [ $$s -eq 5 ] && exit 0 || exit $$s

smoke:
	$(UV) run python -m llm.scripts.run_semi_synthetic --config llm/configs/semi_synthetic_smoke.yaml

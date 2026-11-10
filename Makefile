.PHONY: chase lead autoresearch

OUTPUT ?= runs/$@-dev-$(shell date -u +%Y%m%dT%H%M%SZ)

chase lead:
	uv run --locked python -m worldsmith evaluate --controller $@ --output "$(OUTPUT)"

autoresearch:
	@./scripts/run-autoresearch.sh

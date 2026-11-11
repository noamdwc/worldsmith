.PHONY: chase lead autoresearch clear-autoresearch

OUTPUT ?= runs/$@-dev-$(shell date -u +%Y%m%dT%H%M%SZ)

chase lead:
	uv run --locked python -m worldsmith evaluate --controller $@ --output "$(OUTPUT)"

autoresearch:
	@./scripts/run-autoresearch.sh

clear-autoresearch:
	@archive_id=$$(uv run --locked python -m worldsmith.research_lifecycle archive) && \
		uv run --locked python -m worldsmith.research_lifecycle reset "$$archive_id" && \
		printf 'Workspace reset. Archived research work in research-history/%s\n' "$$archive_id"

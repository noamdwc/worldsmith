.PHONY: chase lead autoresearch autoresearch-maniskill autoresearch-libero autoresearch-robosuite clear-autoresearch

OUTPUT ?= runs/$@-dev-$(shell date -u +%Y%m%dT%H%M%SZ)

chase lead:
	uv run --locked python -m worldsmith evaluate --controller $@ --output "$(OUTPUT)"

autoresearch:
	@./scripts/run-autoresearch.sh

autoresearch-maniskill:
	@./scripts/run-autoresearch.sh maniskill

autoresearch-libero:
	@./scripts/run-autoresearch.sh libero

autoresearch-robosuite:
	@./scripts/run-autoresearch.sh robosuite

clear-autoresearch:
	@archive_id=$$(uv run --locked python -m worldsmith.research_lifecycle archive) && \
		uv run --locked python -m worldsmith.research_lifecycle reset "$$archive_id" && \
		printf 'Workspace reset. Archived research work in research-history/%s\n' "$$archive_id"

.PHONY: chase lead autoresearch

OUTPUT ?= runs/$@-dev-$(shell date -u +%Y%m%dT%H%M%SZ)
LIFECYCLE := uv run --locked python -m worldsmith.research_lifecycle

chase lead:
	uv run --locked python -m worldsmith evaluate --controller $@ --output "$(OUTPUT)"

autoresearch:
	@set -e; \
	$(LIFECYCLE) start; \
	status=0; \
	(cd autoresearch && codex exec --sandbox workspace-write \
		"Run the three-iteration AutoResearch session in RESEARCH.md and summarize your findings.") \
		|| status=$$?; \
	archive_id=$$($(LIFECYCLE) archive); \
	printf 'Archived session in research-history/%s\n' "$$archive_id"; \
	if [ "$$status" -ne 0 ]; then \
		printf 'Codex exited with status %s. Workspace kept for inspection.\n' "$$status" >&2; \
		exit "$$status"; \
	fi; \
	$(LIFECYCLE) reset "$$archive_id"

.PHONY: autoresearch autoresearch-maniskill autoresearch-libero autoresearch-robosuite clear-autoresearch colab-build-envs colab-status colab-stop

SIMULATOR ?= all

autoresearch:
	@uv run --locked python -m worldsmith.colab run

autoresearch-maniskill:
	@uv run --locked python -m worldsmith.colab run maniskill

autoresearch-libero:
	@uv run --locked python -m worldsmith.colab run libero

autoresearch-robosuite:
	@uv run --locked python -m worldsmith.colab run robosuite

colab-status:
	@uv run --locked python -m worldsmith.colab status

colab-stop:
	@uv run --locked python -m worldsmith.colab stop

colab-build-envs:
	@uv run --locked python -m worldsmith.colab build-envs "$(SIMULATOR)"

clear-autoresearch:
	@archive_id=$$(uv run --locked python -m worldsmith.research_lifecycle archive) && \
		uv run --locked python -m worldsmith.research_lifecycle reset "$$archive_id" && \
		printf 'Workspace reset. Archived research work in research-history/%s\n' "$$archive_id"

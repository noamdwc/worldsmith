.PHONY: autoresearch autoresearch-maniskill autoresearch-libero autoresearch-robosuite clear-autoresearch colab-build-envs colab-status colab-stop colab-slm-smoke

SIMULATOR ?= all

# Load local configuration when present; allow callers to choose another env file.
export UV_ENV_FILE ?= $(wildcard $(CURDIR)/.env)

autoresearch:
	@uv run --locked python -m worldsmith.research_launcher

autoresearch-maniskill:
	@uv run --locked python -m worldsmith.research_launcher maniskill

autoresearch-libero:
	@uv run --locked python -m worldsmith.research_launcher libero

autoresearch-robosuite:
	@uv run --locked python -m worldsmith.research_launcher robosuite

colab-status:
	@uv run --locked python -m worldsmith.colab status

colab-stop:
	@uv run --locked python -m worldsmith.colab stop

colab-build-envs:
	@uv run --locked python -m worldsmith.colab build-envs "$(SIMULATOR)"

colab-slm-smoke:
	@uv run --locked python -m worldsmith.colab slm-smoke --new-session

clear-autoresearch:
	@archive_id=$$(uv run --locked python -m worldsmith.research_lifecycle archive) && \
		uv run --locked python -m worldsmith.research_lifecycle reset "$$archive_id" && \
		printf 'Workspace reset. Archived research work in research-history/%s\n' "$$archive_id"

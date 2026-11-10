#!/bin/sh
set -e

CDPATH= cd "$(dirname "$0")/.."

lifecycle() {
    uv run --locked python -m worldsmith.research_lifecycle "$@"
}

lifecycle start
status=0
(
    cd autoresearch && exec codex exec --sandbox workspace-write \
        "Run the three-iteration AutoResearch session in RESEARCH.md and summarize your findings."
) || status=$?

archive_id=$(lifecycle archive)
printf 'Archived session in research-history/%s\n' "$archive_id"

if [ "$status" -ne 0 ]; then
    printf 'Codex exited with status %s. Workspace kept for inspection.\n' "$status" >&2
    exit "$status"
fi

lifecycle reset "$archive_id"

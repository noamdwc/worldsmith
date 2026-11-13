#!/bin/sh
set -e

CDPATH= cd "$(dirname "$0")/.."

case "${1:-all}" in
    all) suite=all; tasks="the full development suite" ;;
    maniskill) suite=maniskill; tasks="the six tabletop manipulation tasks" ;;
    libero) suite=libero; tasks="the three arm manipulation tasks with varied starting layouts" ;;
    *) printf 'Usage: %s [all|maniskill|libero]\n' "$0" >&2; exit 2 ;;
esac
export WORLDSMITH_RESEARCH_SUITE="$suite"

lifecycle() {
    uv run --locked python -m worldsmith.research_lifecycle "$@"
}

lifecycle start
status=0
(
    cd autoresearch && exec codex exec --sandbox workspace-write \
        "Run the three-iteration AutoResearch session in RESEARCH.md on $tasks and summarize your findings."
) || status=$?

archive_id=$(lifecycle archive)
printf 'Archived session in research-history/%s\n' "$archive_id"

if [ "$status" -ne 0 ]; then
    printf 'Codex exited with status %s. Workspace kept for inspection.\n' "$status" >&2
    exit "$status"
fi

lifecycle reset "$archive_id"

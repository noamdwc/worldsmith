# AutoResearch on Google Colab

Install and authenticate the [Google Colab CLI](https://github.com/googlecolab/google-colab-cli) in your local shell. Your account needs access to an L4 runtime and sufficient Colab compute units.

```sh
uv tool install google-colab-cli
colab sessions
uv sync --locked --inexact
```

Create a private [Cloudflare R2 bucket and S3 API token](https://developers.cloudflare.com/r2/api/tokens/) with object read/write access to that bucket. Use the same external credential-file format as `colab-utils`. Store the file outside the Worldsmith checkout with permissions `600`.

```dotenv
R2_ACCOUNT_ID=your-account-id
R2_ACCESS_KEY_ID=your-access-key-id
R2_SECRET_ACCESS_KEY=your-secret-access-key
```

```sh
export R2_CREDENTIALS_FILE="$HOME/.config/colab-utils/r2.env"
export R2_BUCKET=your-bucket-name
make autoresearch
```

## Package preparation and startup

The launcher checks R2 before allocating an L4. On a package cache miss, it builds a pure-Python Worldsmith wheel locally with `uv build` and uploads it to R2. The package key hashes the Python source, package data, and build configuration. An existing key skips the local build and upload. The small bootstrap bundle carries setup scripts, pinned dependency lists, and short-lived R2 URLs.

Colab verifies Linux x86-64 and uses `nvidia-smi` to check the assigned GPU. It creates fresh Python 3.11 or 3.12 environments and installs the same Worldsmith wheel into each. A cold simulator cache builds or downloads Linux-compatible wheels from the exact pins in [simulation_environments/colab](simulation_environments/colab). A warm cache installs those wheels offline. The wheelhouse keys hash the dependency pins and include the Python version and platform. The Linux pins preserve the direct simulator versions, with CPU builds of the pinned PyTorch releases for the existing CPU simulation settings.

LIBERO-PRO source and assets use a separate cache keyed by their pinned revisions and checksum list. Colab verifies the pinned data before registering it in the LIBERO environment. The ManiSkill Go2 cache is keyed by the pinned ManiSkill release and its asset ID. R2 caches contain wheel archives and simulator data. Python environments are recreated per session, and candidate snapshots are staged separately for each job. Colab installs the base system prerequisites and runs one smoke episode per simulator before starting the local researcher.

`make colab-status` reports the tracked runtime and its availability. `make colab-stop` releases it and can be repeated. If automatic cleanup fails, the tracked name is retained for manual cleanup. After a force kill during allocation, also inspect `colab sessions` for assignments the CLI may have created.

## Evaluation artifacts

Each tool call freezes the candidate locally and uses its run ID as the R2 job ID under `worldsmith/evaluations/<job-id>/`. AutoResearch evaluates one job at a time. Each simulator gets a fresh Python process for the candidate snapshot and retains its existing episode worker count.

| Object | Contents |
| --- | --- |
| `input/request.json` | Job ID, snapshot hash, tasks, seeds and worker counts |
| `input/candidate.tar.gz` | Frozen controller and its workspace helpers |
| `output/artifacts.tar.gz` | Native summaries and traces, process logs and an execution manifest |
| `output/completed.json` | Confirmation that the successful job's archive was retrieved and checked locally |

The long-lived R2 credentials stay local. The CLI sends the snapshot and a short-lived, write-only [presigned URL](https://developers.cloudflare.com/r2/api/s3/presigned-urls/) for that job's artifact bundle. The local server retrieves the bundle from R2, keeps the existing owner results under `runs/owner/<job-id>/`, and publishes the same filtered development feedback through MCP.

## Recovering a job after a CLI failure

If execution fails, times out, or loses its reply, the local server checks the original job's R2 output key. For an ambiguous reply it waits up to 15 seconds for an in-flight upload. It checks the archive against the job ID, controller digest, requested groups and tasks, seeds, and expected output files. A valid bundle recovers either the scored result or the reported controller or simulator failure with its available diagnostics. A missing or invalid bundle produces an infrastructure error. The original job runs at most once.

`artifacts.tar.gz` records the remote outcome, including partial diagnostics on some failures. The local backend writes `completed.json` only after accepting a successful result and validating its development feedback. Use that marker when consuming completed jobs from R2.

## Manual verification

With the environment variables above set, run the following from the repository root. This exercises the existing MCP evaluation handler twice on one L4 while keeping the editable research workspace untouched. Select `all` in place of `navigation` to run the full development benchmark twice.

```sh
(
    set -e
    export WORLDSMITH_RESEARCH_WORKSPACE="$(mktemp -d)"
    cp worldsmith/research_baseline/controller.py "$WORLDSMITH_RESEARCH_WORKSPACE/controller.py"
    uv run --locked python -m worldsmith.colab start
    trap 'make colab-stop' EXIT
    make colab-status
    export WORLDSMITH_EVALUATION_BACKEND=colab
    export WORLDSMITH_RESEARCH_SUITE=navigation
    uv run --locked python - <<'PY'
from worldsmith.research_mcp.research_mcp import evaluate_controller, get_episode_trace

for _ in range(2):
    result = evaluate_controller()
    trace = get_episode_trace(0)
    print(result["run_id"], result["run_dir"], result["aggregate_score"], trace["seed"])
PY
    make colab-status
)
make colab-status
```

Check that the status calls inside the subshell show the same session and that the final call reports no tracked runtime. The two calls should produce different job IDs with the same seed list and score for the same candidate. After stopping Colab, use the R2 dashboard to download `output/artifacts.tar.gz` for both jobs and check for each `output/completed.json`. Run `make autoresearch` to verify startup and shutdown with the actual Codex MCP connection.

Repeat the start and stop with the same checkout and R2 bucket to verify a warm start. The existing package, wheelhouse, and asset keys under `worldsmith/cache/` should be reused. The startup history should show offline simulator installs and smoke episodes in the new runtime.

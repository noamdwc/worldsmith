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
make colab-build-envs
make autoresearch
```

## Prepare simulator environments

Run `make colab-build-envs` to prepare all four simulators. Set `SIMULATOR` to `pymunk`, `maniskill`, `libero`, or `robosuite` to prepare one.

```sh
make colab-build-envs SIMULATOR=maniskill
make autoresearch-maniskill
```

Preparation starts a temporary Linux x86-64 Colab L4 session only when a required cache is missing. It verifies the NVIDIA L4 before building or validating environments. It builds complete wheelhouses from the exact pins in [simulation_environments/colab](simulation_environments/colab), installs each new wheelhouse from its archive with `--no-index`, and checks the installed dependency closure. The Linux pins use CPU builds of the pinned PyTorch releases for the existing simulation settings.

LIBERO-PRO source and assets and the ManiSkill Go2 assets have separate cache keys and verification. Preparation checks the pinned LIBERO-PRO data and the referenced Go2 meshes. It also caches the pinned standalone CPython and uv tools so a fresh research runtime can install offline.

Each verified archive is published with a conditional, checksum-checked R2 upload under `worldsmith/cache/`. Prepared cache keys include the Colab Linux x86-64 L4/CUDA target. Wheelhouse keys also include the dependency pins and Python version, and asset keys include the relevant pinned simulator or asset revisions. The command checks the committed objects and attempts to release the temporary session on success, failure, or interruption. A later invocation skips completed caches. If an object at a prepared key is invalid, remove the named object before rerunning preparation. Run preparation once to populate the L4-specific keys.

## Start AutoResearch

Research startup checks for the required prepared caches before allocating an L4. A missing cache reports the affected artifacts and directs the owner to `make colab-build-envs`. The launcher separately builds the pure-Python Worldsmith wheel locally with `uv build` when its source and build configuration are absent from the package cache.

The L4 runtime uses the base system libraries from the Colab image and downloads the prepared tooling, wheelhouses, and assets from R2. It checks the downloaded archives, creates fresh Python environments, installs the pinned wheels offline, and runs one smoke episode per selected simulator. The bootstrap carries read-only cache URLs. Candidate snapshots are staged separately for each job.

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

With the environment variables above set, run the following from the repository root. This exercises the existing MCP evaluation handler twice on one L4 while keeping the editable research workspace untouched. To run the full development benchmark twice, prepare all environments, start the session with `start all`, and set `WORLDSMITH_RESEARCH_SUITE=all`.

```sh
make colab-build-envs SIMULATOR=pymunk
(
    set -e
    export WORLDSMITH_RESEARCH_WORKSPACE="$(mktemp -d)"
    cp worldsmith/research_baseline/controller.py "$WORLDSMITH_RESEARCH_WORKSPACE/controller.py"
    uv run --locked python -m worldsmith.colab start pymunk
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

Repeat `make colab-build-envs SIMULATOR=pymunk` to verify that the prepared caches are reused without allocating another preparation session. Repeat the research start and stop to verify that the new L4 session uses the same cache keys and installs offline.

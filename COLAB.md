# AutoResearch on Google Colab

## Setup

Install the Google Colab CLI and AWS CLI, authenticate with Colab, and sync the
checkout. Your Colab account needs access to an L4 runtime and enough compute
units. [Google Colab CLI](https://github.com/googlecolab/google-colab-cli)

```sh
uv tool install google-colab-cli
uv tool install awscli
colab sessions
uv sync --locked --inexact
```

Create a private R2 bucket and an S3 API token with object read/write access to
that bucket. Save the credentials in a file outside the checkout with
permissions `600`. [Cloudflare R2 API tokens](https://developers.cloudflare.com/r2/api/tokens/)

```dotenv
R2_ACCOUNT_ID=your-account-id
R2_ACCESS_KEY_ID=your-access-key-id
R2_SECRET_ACCESS_KEY=your-secret-access-key
```

```sh
export R2_CREDENTIALS_FILE="$HOME/.config/colab-utils/r2.env"
export R2_BUCKET=your-bucket-name
```

## Prepare simulator environments

Run `make colab-build-envs` before your first Colab run. Use
`make colab-build-envs SIMULATOR=maniskill` to prepare just one environment.
The supported selectors are `pymunk`, `maniskill`, `libero`, and `robosuite`.
Rerun preparation when startup reports a missing cache. It reuses prepared
caches and starts a temporary L4 only when needed.

## Start and stop

`make autoresearch` starts Codex with one L4 session for all evaluations in the
run and releases it on exit. Use `make autoresearch-maniskill`,
`make autoresearch-libero`, or `make autoresearch-robosuite` for a single
simulator. To run locally with installed simulators, set
`WORLDSMITH_EVALUATION_BACKEND=local` when invoking the same target.

For manual testing, start and inspect a persistent Colab session from the
repository root. Pass a simulator name to initialize just that environment.

```sh
uv run --locked python -m worldsmith.colab start all
make colab-status
make colab-stop
```

For manual evaluations against that runtime, set
`WORLDSMITH_EVALUATION_BACKEND=colab` and `WORLDSMITH_COLAB_SESSION` to the
session name printed at startup. Release the manual session before starting
`make autoresearch`, which creates its own runtime.

## Troubleshooting

- **Allocation failed.** Check CLI authentication with `colab sessions` and
  verify that your account can allocate an L4.
- **A prepared cache is missing or invalid.** Rerun `make colab-build-envs` for
  the affected simulator. If an existing R2 object is reported as invalid,
  remove the named object before retrying.
- **Initialization failed.** Check the reported setup stage and the local
  Colab CLI history.
- **A session is still running.** Run `make colab-status` and retry
  `make colab-stop`. After a force kill during allocation, also check
  `colab sessions` for an assignment and release it with
  `colab stop -s SESSION_NAME` if needed.

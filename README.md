# Worldsmith

Worldsmith is an ongoing research project asking whether an autonomous AI researcher can improve general-purpose robot control through iterative simulation experiments.
The goal is to develop reusable skills and representations that transfer across tasks and embodiments, reducing the need to engineer a separate solution for each problem.
Generalization remains an open research question.

## Research approach

An autonomous AI researcher develops and improves a shared robot controller, tests it across simulation tasks, and uses performance feedback to guide the next experiment.
The researcher works between experiments; the runtime controller observes and acts within each episode.
Development feedback is kept separate from owner validation to help assess transfer beyond the tasks and initial conditions used during experimentation.

## Research loop

```mermaid
flowchart LR
    R[AI researcher] -->|improves| C[Shared robot controller]
    C -->|acts in| S[Simulation tasks]
    S -->|performance feedback| R
```

## Benchmarks

Worldsmith spans **20 tasks across four simulation environments**, covering planar movement, manipulation, locomotion, and coordinated control.

| Environment | Tasks | Control problems |
| --- | ---: | --- |
| Pymunk | 5 | Interception, navigation, tracking, and pushing |
| ManiSkill | 8 | Manipulation, precision insertion, and quadruped locomotion |
| LIBERO-PRO | 3 | Drawer opening and object placement |
| RoboSuite | 4 | Doors, assembly, two-arm handover, and wiping |

Current tasks use structured state observations rather than camera images.
See the [benchmark inventory](BENCHMARKS.md) for task details and development/validation partitions.

## Getting started

From the repository root, install the pinned Pymunk environment with [uv](https://docs.astral.sh/uv/) and run the fast tests:

```sh
uv sync --locked
uv run --locked python -m pytest
```

The other simulators use separate environments with their own setup instructions and opt-in tests:

- [ManiSkill](simulation_environments/maniskill/README.md)
- [LIBERO-PRO](simulation_environments/libero/README.md)
- [RoboSuite](simulation_environments/robosuite/README.md)

The locked local environment uses Python 3.12+ and includes the Google Colab and
AWS CLIs. For Colab and AutoResearch, copy `.env.example` to `.env` and set your R2
bucket and external credentials-file path. Make loads this git-ignored file automatically through uv;
existing terminal settings take precedence. Keep the credential file outside the
repository, with plain `R2_ACCOUNT_ID=...`, `R2_ACCESS_KEY_ID=...`, and
`R2_SECRET_ACCESS_KEY=...` lines. The keys stay in that external file.

```sh
cp .env.example .env  # edit the bucket and credentials-file path
uv run --locked --env-file .env colab sessions  # authenticate when prompted, if needed
make colab-build-envs
make autoresearch
```

If the CLI reports a certificate error, uncomment the CA-bundle settings in `.env`
and point them to a trusted bundle (`/etc/ssl/cert.pem` on macOS).

### SLM inference

[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) is served
by vLLM with Hermes tool calling through the OpenAI-compatible API and Python client.

```sh
make colab-slm-smoke
```

Diagnostic logs: `runs/colab/<session-name>/{slm-install.log,slm-smoke.log,vllm.log}`.

## Documentation

- [Research workflow](autoresearch/RESEARCH.md): experiment process and comparison rules.
- [Controller API](autoresearch/API.md): observations, actions, and feedback tools.
- [Development task guide](autoresearch/TASKS.md): task fields, limits, and horizons.
- [Benchmark inventory](BENCHMARKS.md): the full task catalog.

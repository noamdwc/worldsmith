"""The wheel script uses content-based caching and immutable R2 uploads."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def wheel_script(tmp_path):
    project = tmp_path / "checkout"
    for path, text in (
        ("worldsmith/__init__.py", ""),
        ("worldsmith/api.py", "value = 1\n"),
        ("worldsmith/research_baseline/JOURNAL.md", "# Research\n"),
        ("pyproject.toml", '[project]\nname = "worldsmith"\nversion = "0.1.0"\n'),
        ("autoresearch/controller.py", "# candidate\n"),
    ):
        destination = project / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
    script = project / "scripts/prepare-colab-wheel.sh"
    script.parent.mkdir()
    shutil.copyfile(ROOT / "scripts/prepare-colab-wheel.sh", script)
    credentials = tmp_path / "r2.env"
    credentials.write_text(
        f"R2_ACCOUNT_ID={'a' * 32}\nR2_ACCESS_KEY_ID=test-key\nR2_SECRET_ACCESS_KEY=test-secret\n",
    )
    tools = tmp_path / "bin"
    tools.mkdir()
    cli = tools / "aws"
    cli.write_text(f"#!{sys.executable}\n" + r'''
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
directory = Path(os.environ["WHEEL_TEST_STATE"])
with (directory / "calls.jsonl").open("a") as file:
    file.write(json.dumps([Path(sys.argv[0]).name, args]) + "\n")
def argument(flag):
    return args[args.index(flag) + 1]
if Path(sys.argv[0]).name == "uv":
    source = Path(args[-1])
    assert not (source / "autoresearch").exists()
    assert not (source / "simulation_environments").exists()
    output = Path(argument("--out-dir"))
    (output / "worldsmith-0.1.0-py3-none-any.whl").write_bytes(b"wheel")
elif "list-objects-v2" in args:
    if os.environ.get("WHEEL_TEST_FAIL") == "list":
        print("AccessDenied test-secret", file=sys.stderr)
        sys.exit(1)
    matches = [p.name for p in directory.glob("*.key") if p.read_text().startswith(argument("--prefix"))]
    print("\t".join((directory / name).read_text() for name in matches) or "None")
elif "put-object" in args:
    assert argument("--if-none-match") == "*"
    if os.environ.get("WHEEL_TEST_FAIL") == "put":
        print("network error test-secret", file=sys.stderr)
        sys.exit(1)
    if os.environ.get("WHEEL_TEST_FAIL") == "concurrent":
        print("PreconditionFailed", file=sys.stderr)
        sys.exit(1)
    (directory / f"{len(list(directory.glob('*.key')))}.key").write_text(argument("--key"))
elif "presign" in args:
    print("https://example.test/wheel?signed")
else:
    raise AssertionError(args)
''')
    cli.chmod(0o755)
    (tools / "uv").symlink_to(cli)
    state = tmp_path / "state"
    state.mkdir()
    env = {
        **os.environ, "PATH": f"{tools}:{os.environ['PATH']}", "R2_BUCKET": "test-bucket",
        "R2_CREDENTIALS_FILE": str(credentials), "WHEEL_TEST_STATE": str(state),
    }
    return script, project, state, env


def run_wheel(fixture, **overrides):
    script, _, _, env = fixture
    return subprocess.run(
        ["bash", str(script), "600"], env={**env, **overrides}, capture_output=True, text=True,
    )


def test_script_reuses_unchanged_source_and_excludes_candidates(wheel_script):
    _, project, state, _ = wheel_script
    first = run_wheel(wheel_script)
    assert first.returncode == 0, first.stderr
    assert first.stdout.splitlines() == ["worldsmith-0.1.0-py3-none-any.whl", "https://example.test/wheel?signed"]
    (project / "autoresearch/controller.py").write_text("# changed candidate\n")
    assert run_wheel(wheel_script).returncode == 0
    calls = [json.loads(line) for line in (state / "calls.jsonl").read_text().splitlines()]
    assert sum(name == "uv" for name, _ in calls) == 1
    assert sum("put-object" in args for _, args in calls) == 1
    (project / "worldsmith/api.py").write_text("value = 2\n")
    assert run_wheel(wheel_script).returncode == 0
    assert len(list(state.glob("*.key"))) == 2


@pytest.mark.parametrize("operation", ["list", "put"])
def test_script_reports_r2_failure_without_signing_or_leaking_secrets(wheel_script, operation):
    _, _, state, _ = wheel_script
    result = run_wheel(wheel_script, WHEEL_TEST_FAIL=operation)
    assert result.returncode != 0
    assert not result.stdout
    assert "R2 wheel" in result.stderr and "test-secret" not in result.stderr
    calls = [json.loads(line) for line in (state / "calls.jsonl").read_text().splitlines()]
    assert all("presign" not in args for _, args in calls)
    if operation == "list":
        assert all(name != "uv" for name, _ in calls)


def test_script_accepts_an_immutable_upload_won_by_another_startup(wheel_script):
    result = run_wheel(wheel_script, WHEEL_TEST_FAIL="concurrent")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == "worldsmith-0.1.0-py3-none-any.whl"


def test_script_rejects_credentials_inside_the_checkout(wheel_script):
    _, project, _, _ = wheel_script
    credentials = project / "r2.env"
    credentials.touch()
    result = run_wheel(wheel_script, R2_CREDENTIALS_FILE=str(credentials))
    assert result.returncode != 0
    assert "must be outside" in result.stderr

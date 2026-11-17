"""Exercise the Colab lifecycle and artifact protocol without cloud services."""

import ast
import base64
import json
import runpy
import shutil
import signal
import subprocess
import tarfile
from contextlib import nullcontext
from hashlib import md5
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse
from zipfile import ZipFile

import pytest
from botocore.response import StreamingBody

from worldsmith import research_lifecycle
from worldsmith.colab import __main__ as owner
from worldsmith.colab import backend, packaging, remote_job, session, storage
from worldsmith.research_mcp import research_mcp

PREPARED_OBJECT = {
    "Metadata": packaging.VERIFIED_METADATA,
    "ContentLength": 42,
    "ETag": f'"{"a" * 32}"',
}


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    """Fake the CLI transport and R2, with real job unpacking and evaluation."""
    staging = tmp_path / "vm"
    staging.mkdir()
    calls = []
    objects = {}
    requests = []

    class Storage:
        def check(self):
            pass

        def exists(self, key):
            return key in objects

        def head(self, key):
            return PREPARED_OBJECT

        def put(self, key, source, *, cache=False):
            assert key not in objects
            objects[key] = source.read_bytes()

        def signed_url(self, key, *, expires, upload=False):
            assert 0 < expires <= 3600
            return f"https://example.test/{key}?private-signature"

        def download(self, key, destination):
            if key not in objects:
                raise storage.ObjectMissing("R2 has no artifact bundle for this job")
            destination.write_bytes(objects[key])

    def upload(request, *, timeout):
        key = request.full_url.split("example.test/")[1].split("?")[0]
        assert request.get_method() == "PUT"
        assert request.get_header("If-none-match") == "*"
        assert key not in objects
        objects[key] = request.data.read()
        return nullcontext()

    def command(self, action, *args, **kwargs):
        calls.append((self.name, action, args))
        if action == "new":
            assert args == ("--gpu", "L4")
        elif action == "upload":
            local, remote = args
            (staging / Path(remote).name).write_bytes(Path(local).read_bytes())
        elif action == "exec":
            code = kwargs["code"]
            if "setup-colab.py" in code:
                with tarfile.open(staging / "bootstrap.tar.gz") as archive:
                    assert "scripts/setup-colab.py" in archive.getnames()
                    assert "simulation_environments/colab/pymunk.txt" in archive.getnames()
                    assert "worldsmith/colab/remote_job.py" not in archive.getnames()
                    assert "autoresearch/controller.py" not in archive.getnames()
                return "WORLDSMITH_COLAB_READY\n"
            assert "private-signature" not in code
            call = next(node for node in ast.walk(ast.parse(code)) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute) and node.func.attr == "Popen")
            job_id, budget = ast.literal_eval(call.args[0])[-2:]
            dispatch = json.loads((staging / f"{job_id}.json").read_text())
            requests.append(dispatch["request"])
            status = remote_job.run_job(job_id, float(budget), staging)
            return f"WORLDSMITH_JOB {job_id} {status}\n"
        return ""

    monkeypatch.setattr(session, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(session.ColabSession, "command", command)
    monkeypatch.setattr(session, "R2", Storage)
    monkeypatch.setattr(session, "prepare_wheel", lambda *a, **kw: {"filename": "test.whl", "url": "test"})
    monkeypatch.setattr(backend, "R2", Storage)
    monkeypatch.setattr(backend, "RECOVERY_WINDOW_SECONDS", 0)
    monkeypatch.setenv("PYTHONPATH", str(research_lifecycle.ROOT))
    monkeypatch.setattr(remote_job, "urlopen", upload)
    return calls, objects, requests


@pytest.fixture
def researcher(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    candidate = workspace / "controller.py"
    candidate.write_bytes((research_lifecycle.BASELINE / "controller.py").read_bytes())
    monkeypatch.setattr(research_mcp, "WORKSPACE", workspace)
    monkeypatch.setattr(research_mcp, "OWNER_RUNS", tmp_path / "owner")
    monkeypatch.setattr(research_mcp, "SUITE", "navigation")
    monkeypatch.setattr(research_mcp, "latest_feedback", None)
    monkeypatch.setattr(research_mcp, "retained_summary", None)
    monkeypatch.setenv("WORLDSMITH_EVALUATION_BACKEND", "colab")
    monkeypatch.delenv("WORLDSMITH_COLAB_SESSION", raising=False)
    return candidate


def test_mcp_reuses_session_and_isolates_candidate_processes(fake_cli, researcher, monkeypatch):
    calls, objects, requests = fake_cli
    runtime = session.start_session()
    monkeypatch.setenv("WORLDSMITH_COLAB_SESSION", runtime.name)
    monkeypatch.setattr(backend, "WORKER_OVERRIDE", 2)
    # The two evaluations import the same module name. An interpreter reused
    # across jobs would either cache the old source or retain this global.
    prefix = (
        "import builtins\n"
        "assert not hasattr(builtins, 'worldsmith_colab_test')\n"
        "builtins.worldsmith_colab_test = True\n"
    )
    source = researcher.read_text()
    researcher.write_text(prefix + source)
    first = research_mcp.evaluate_controller()
    first_trace = research_mcp.get_episode_trace(0)
    researcher.write_text(prefix + source + "\n# second candidate\n")
    second = research_mcp.evaluate_controller()
    runtime.stop()

    assert first["controller_snapshot_sha256"] != second["controller_snapshot_sha256"]
    assert second["retained_controller_snapshot_sha256"] == first["controller_snapshot_sha256"]
    assert set(first["tasks"]) == {"navigation"}
    assert first["aggregate_score"] == second["aggregate_score"]
    assert research_mcp.get_episode_trace(0) == first_trace
    assert first["seeds"] == second["seeds"] == list(range(8))
    assert first["run_dir"] == f"runs/{first['run_id']}/results"
    for result, request in zip((first, second), requests, strict=True):
        run_id = result["run_id"]
        job = f"{backend.PREFIX}/{run_id}"
        assert request["groups"] == {"pymunk": {"tasks": ["navigation"], "workers": 2}}
        assert request == json.loads(objects[f"{job}/input/request.json"])
        assert request["job_id"] == run_id
        with tarfile.open(fileobj=BytesIO(objects[f"{job}/input/candidate.tar.gz"])) as archive:
            stored_source = archive.extractfile("autoresearch/controller.py").read().decode()
        assert stored_source == (prefix + source if result is first else researcher.read_text())
        assert json.loads(objects[f"{job}/output/completed.json"])["job_id"] == run_id
        assert (research_mcp.OWNER_RUNS / run_id / "pymunk-stderr.txt").is_file()

    assert [action for _, action, _ in calls if action in {"new", "exec", "stop"}] == [
        "new", "exec", "exec", "exec", "stop",
    ]
    assert {name for name, _, _ in calls} == {runtime.name}
    assert session.tracked_session() is None
    assert not any(b"private-signature" in content for content in objects.values())


def test_mcp_requires_existing_session(fake_cli, researcher):
    with pytest.raises(RuntimeError, match="Colab session is unavailable"):
        research_mcp.evaluate_controller()
    assert fake_cli == ([], {}, [])


def test_candidate_and_simulator_pins_do_not_change_the_worldsmith_wheel(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    sources = (
        Path("pyproject.toml"), Path("worldsmith/__init__.py"),
        Path("worldsmith/research_baseline/JOURNAL.md"), Path("autoresearch/controller.py"),
        packaging.REQUIREMENTS / "pymunk.txt",
    )
    for path in sources:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(packaging.ROOT / path, root / path)
    monkeypatch.setattr(packaging, "ROOT", root)

    class Cache:
        key = None

        def exists(self, key):
            return key == self.key

        def signed_url(self, key, *, expires):
            return key

        def put(self, key, source, *, cache):
            assert cache and source.read_bytes() == b"wheel"
            self.key = key

    builds = []

    def build(command, **kwargs):
        source = Path(command[-1])
        assert not (source / "autoresearch").exists()
        assert not (source / "simulation_environments").exists()
        builds.append(source)
        (source.parent / "worldsmith-0.1.0-py3-none-any.whl").write_bytes(b"wheel")

    monkeypatch.setattr(packaging.subprocess, "run", build)
    cache = Cache()
    package = packaging.prepare_wheel(cache, expires=600)
    keys = packaging.runtime_cache_keys("pymunk")
    (root / "autoresearch/controller.py").write_text("# edited candidate\n")
    (root / packaging.REQUIREMENTS / "pymunk.txt").write_text("pymunk==7.3.2\n")

    updated = packaging.runtime_cache_keys("pymunk")
    assert keys["tools"] == updated["tools"]
    assert keys["wheelhouses"] != updated["wheelhouses"]
    assert packaging.prepare_wheel(cache, expires=600) == package
    assert len(builds) == 1


def test_research_requires_prepared_caches_before_allocating(fake_cli, monkeypatch):
    calls, _, _ = fake_cli
    monkeypatch.setattr(session.R2, "head", lambda self, key: None)
    monkeypatch.setattr(session, "prepare_wheel", lambda *a, **kw: pytest.fail("built Worldsmith wheel"))
    with pytest.raises(storage.StorageError, match="Run make colab-build-envs"):
        session.start_session("pymunk")
    assert calls == []
    assert session.tracked_session() is None


@pytest.mark.parametrize("selector", ["all", "maniskill"])
def test_research_bootstrap_has_only_downloads_for_selected_simulators(fake_cli, monkeypatch, selector):
    initialize = session.initialize

    def check_bootstrap(runtime, archive, *, marker):
        with tarfile.open(archive) as bundle:
            config = json.load(bundle.extractfile("setup.json"))
        assert config["mode"] == "research"
        assert config["wheel"]
        expected_wheels = set(packaging.ENVIRONMENT_PYTHONS) if selector == "all" else {"maniskill"}
        assert set(config["wheelhouses"]) == expected_wheels
        assert set(config["assets"]) == expected_wheels & {"libero", "maniskill"}
        for family in ("tools", "wheelhouses", "assets"):
            assert all("get" in entry and "put" not in entry for entry in config[family].values())
        initialize(runtime, archive, marker=marker)

    monkeypatch.setattr(session, "initialize", check_bootstrap)
    session.start_session(selector).stop()


@pytest.mark.parametrize("failure", [None, "setup", "interrupt", "missing after success"])
def test_owner_builds_selected_caches_and_releases_temporary_session(tmp_path, monkeypatch, failure):
    keys = packaging.runtime_cache_keys("maniskill")
    tool_key = next(iter(keys["tools"].values()))
    prepared = {tool_key}
    actions = []

    class Storage:
        def check(self):
            pass

        def head(self, key):
            return PREPARED_OBJECT if key in prepared else None

        def signed_url(self, key, *, expires, upload=False, metadata=None):
            assert metadata == (packaging.VERIFIED_METADATA if upload else None)
            return key

    def command(self, action, *args, **kwargs):
        assert self.name.startswith("worldsmith-env-")
        actions.append((action, args))

    def initialize(runtime, archive, *, marker):
        assert marker == "WORLDSMITH_COLAB_PREPARED"
        with tarfile.open(archive) as bundle:
            config = json.load(bundle.extractfile("setup.json"))
        assert config["mode"] == "prepare"
        assert "wheel" not in config
        assert set(config["wheelhouses"]) == {"maniskill"}
        assert set(config["assets"]) == {"maniskill"}
        assert next(iter(config["tools"].values()))["get"] == tool_key
        if failure == "setup":
            raise session.ColabError("verification failed")
        if failure == "interrupt":
            raise SystemExit(130)
        if failure is None:
            prepared.update([config["wheelhouses"]["maniskill"]["put"], config["assets"]["maniskill"]["put"]])

    monkeypatch.setattr(session, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(session, "R2", Storage)
    monkeypatch.setattr(session, "initialize", initialize)
    monkeypatch.setattr(session.ColabSession, "command", command)
    monkeypatch.setattr(session, "prepare_wheel", lambda *a, **kw: pytest.fail("built Worldsmith wheel"))

    session.STATE_DIR.mkdir()
    session.save_session("active-research", ready=True)
    if failure == "interrupt":
        with pytest.raises(SystemExit, match="130"):
            session.build_environments("maniskill")
    elif failure == "setup":
        with pytest.raises(session.ColabError, match="verification failed"):
            session.build_environments("maniskill")
    elif failure == "missing after success":
        with pytest.raises(session.ColabError, match="Prepared cache was not found"):
            session.build_environments("maniskill")
    else:
        assert session.build_environments("maniskill") == 0
        assert session.build_environments("maniskill") == 0
    assert actions == [("new", ("--gpu", "L4")), ("stop", ())]
    assert session.tracked_session()["name"] == "active-research"


@pytest.mark.parametrize("mode", ["prepare", "research"])
@pytest.mark.parametrize("gpu", ["", "NVIDIA T4\n"])
def test_colab_rejects_unsupported_gpu_before_environment_work(tmp_path, monkeypatch, mode, gpu):
    main = runpy.run_path(str(packaging.ROOT / "scripts/setup-colab.py"))["main"]
    (tmp_path / "setup.json").write_text(json.dumps({
        "mode": mode, "wheelhouses": {"pymunk": {"put": "cache"}},
    }))
    globals_ = main.__globals__
    monkeypatch.setitem(globals_, "ROOT", tmp_path)
    monkeypatch.setitem(globals_, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setitem(globals_, "platform", SimpleNamespace(machine=lambda: "x86_64"))
    monkeypatch.setitem(globals_, "subprocess", SimpleNamespace(check_output=lambda *a, **kw: gpu))
    monkeypatch.setitem(globals_, "stage", lambda *a: pytest.fail("started build commands before GPU check"))
    monkeypatch.setitem(globals_, "install_tools", lambda *a, **kw: pytest.fail("started installing tools"))

    with pytest.raises(RuntimeError, match="Expected NVIDIA L4"):
        main()


def test_runtime_cache_families_depend_on_the_l4_cuda_target(monkeypatch):
    assert packaging.RUNTIME_CACHE_TARGET == "colab-linux-x86_64-l4-cuda"
    l4_keys = packaging.runtime_cache_keys("all")
    monkeypatch.setattr(packaging, "RUNTIME_CACHE_TARGET", "colab-linux-x86_64-cpu")
    cpu_keys = packaging.runtime_cache_keys("all")
    for family, entries in l4_keys.items():
        assert set(entries.values()).isdisjoint(cpu_keys[family].values())


def test_preparation_plan_skips_other_simulators_with_complete_caches():
    keys = packaging.runtime_cache_keys("all")
    missing_asset = keys["assets"]["libero"]

    class Storage:
        def head(self, key):
            return None if key == missing_asset else PREPARED_OBJECT

        def signed_url(self, key, **kwargs):
            return key

    config, missing = packaging.plan_runtime(Storage(), "all", prepare=True, expires=600)
    assert missing == [missing_asset]
    assert set(config["assets"]) == {"libero"}
    assert set(config["wheelhouses"]) == {"libero"}
    assert config["wheelhouses"]["libero"]["get"] == keys["wheelhouses"]["libero"]
    assert set(config["tools"]) == {packaging.ENVIRONMENT_PYTHONS["libero"]}


def test_tool_preparation_uses_uv_directly_from_the_wheel(tmp_path, monkeypatch):
    script = runpy.run_path(str(packaging.ROOT / "scripts/setup-colab.py"))
    install = script["install_tools"]
    version = packaging.ENVIRONMENT_PYTHONS["pymunk"]
    staging = tmp_path / "staging"
    staging.mkdir()

    def stage(label, *command):
        if label == "uv wheel download":
            directory = Path(command[command.index("--dest") + 1])
            directory.mkdir()
            with ZipFile(directory / "uv-linux.whl", "w") as wheel:
                wheel.writestr("uv-0.11.15.data/scripts/uv", b"uv executable")
        elif label == f"Python {version} download":
            binary = Path(command[0])
            assert binary.read_bytes() == b"uv executable"
            assert binary.stat().st_mode & 0o100
            destination = Path(command[command.index("--install-dir") + 1])
            (destination / f"cpython-{version}-linux-x86_64-gnu").mkdir(parents=True)

    monkeypatch.setitem(install.__globals__, "ROOT", tmp_path)
    monkeypatch.setitem(install.__globals__, "stage", stage)
    monkeypatch.setitem(install.__globals__, "publish", lambda *args: None)
    tools = install(
        {"uv_version": packaging.UV_VERSION, "tools": {version: {"put": "cache"}}},
        staging, prepare=True,
    )
    assert Path(tools[version][0]).read_bytes() == b"uv executable"


def test_maniskill_preparation_publishes_assets_under_the_data_directory(tmp_path, monkeypatch):
    install = runpy.run_path(str(packaging.ROOT / "scripts/setup-colab.py"))["install_assets"]
    monkeypatch.setenv("MS_ASSET_DIR", str(tmp_path))
    staging = tmp_path / "staging"
    staging.mkdir()

    def download(label, *command):
        assert label == "ManiSkill Go2 asset download"
        urdf = tmp_path / "maniskill-assets/data/robots/unitree_go2/urdf"
        urdf.mkdir(parents=True)
        for name in ("go2_description.urdf", "go2_description_simplified_locomotion.urdf"):
            (urdf / name).write_text('<robot><mesh filename="foot.stl"/></robot>')
        (urdf / "foot.stl").touch()

    def publish(cache, bundle):
        with tarfile.open(bundle) as archive:
            assert "data/robots/unitree_go2/urdf/foot.stl" in archive.getnames()

    monkeypatch.setitem(install.__globals__, "ROOT", tmp_path)
    monkeypatch.setitem(install.__globals__, "stage", download)
    monkeypatch.setitem(install.__globals__, "publish", publish)
    install({"maniskill": {"put": "cache"}}, {"maniskill": "python"}, staging, prepare=True)


@pytest.mark.parametrize("prepare", [True, False])
def test_wheelhouse_installs_offline_before_publication(tmp_path, monkeypatch, prepare):
    script = runpy.run_path(str(packaging.ROOT / "scripts/setup-colab.py"))
    install = script["install_environment"]
    environment = install.__globals__
    commands = []
    wheel = "pymunk-example-linux_x86_64.whl"

    def stage(name, *args):
        commands.append(args)
        if "wheel" in args:
            assert "--no-deps" not in args
            wheelhouse = Path(args[args.index("--wheel-dir") + 1])
            (wheelhouse / wheel).write_bytes(b"new wheel")
        if "install" in args:
            assert "--offline" in args and "--no-index" in args and "--no-deps" in args
            wheels = Path(args[args.index("--find-links") + 1])
            assert (wheels / wheel).is_file()
        if "check" in args:
            raise RuntimeError("dependency closure is incomplete")

    def restore(cache, bundle, destination):
        bundle.touch()
        (destination / wheel).write_bytes(b"cached wheel")

    monkeypatch.setitem(environment, "stage", stage)
    monkeypatch.setitem(environment, "restore", restore)
    monkeypatch.setitem(environment, "publish", lambda *a: pytest.fail("published an unverified cache"))
    argument = {"python": "3.12.13", **({"put": "cache"} if prepare else {"get": "cache"})}
    tools = {"3.12.13": ("uv", "python")}
    if prepare:
        with pytest.raises(RuntimeError, match="closure is incomplete"):
            install("pymunk", argument, tools, tmp_path, prepare=True)
        assert any("wheel" in command for command in commands)
        assert any("check" in command for command in commands)
    else:
        install("pymunk", argument, tools, tmp_path, prepare=False)
        assert all("--offline" in command for command in commands)


def test_cache_publication_attaches_the_checksum_and_rejects_a_bad_reply(tmp_path, monkeypatch):
    script = runpy.run_path(str(packaging.ROOT / "scripts/setup-colab.py"))
    bundle = tmp_path / "pymunk.tar"
    bundle.write_bytes(b"verified wheelhouse")
    digest = md5(bundle.read_bytes(), usedforsecurity=False)

    def upload(request, *, timeout):
        assert request.get_header("If-none-match") == "*"
        assert request.get_header("Content-md5") == base64.b64encode(digest.digest()).decode()
        assert request.get_header("X-amz-meta-worldsmith-verified") == "1"
        return nullcontext(SimpleNamespace(headers={"ETag": '"different"'}))

    publish = script["publish"]
    monkeypatch.setitem(publish.__globals__, "urlopen", upload)
    cache = {"put": "https://example.test/cache", "headers": {"x-amz-meta-worldsmith-verified": "1"}}
    with pytest.raises(RuntimeError, match="checksum confirmation failed"):
        publish(cache, bundle)


def test_r2_download_streams_the_sdk_body(tmp_path):
    contents = b"evaluation artifacts"
    raw = BytesIO(contents)
    body = StreamingBody(raw, len(contents))
    bucket = storage.R2.__new__(storage.R2)
    bucket.bucket = "test-bucket"
    bucket.client = SimpleNamespace(get_object=lambda **kwargs: {"Body": body})
    destination = tmp_path / "artifact.tar.gz"

    bucket.download("output/artifacts.tar.gz", destination)

    assert destination.read_bytes() == contents
    assert raw.closed


def test_storage_signs_a_single_conditional_put_with_external_credentials(tmp_path, monkeypatch):
    credentials = tmp_path / "r2.env"
    credentials.write_text(
        f"R2_ACCOUNT_ID={'a' * 32}\nR2_ACCESS_KEY_ID=example-key\nR2_SECRET_ACCESS_KEY=example-secret\n",
    )
    monkeypatch.setenv("R2_BUCKET", "test-bucket")
    monkeypatch.setenv("R2_CREDENTIALS_FILE", str(credentials))
    url = urlparse(storage.R2().signed_url(
        "worldsmith/evaluations/job/output/artifacts.tar.gz", expires=500, upload=True,
    ))
    query = parse_qs(url.query)
    assert url.hostname == f"{'a' * 32}.r2.cloudflarestorage.com"
    assert url.path == "/test-bucket/worldsmith/evaluations/job/output/artifacts.tar.gz"
    assert query["X-Amz-SignedHeaders"] == ["host;if-none-match"]
    assert query["X-Amz-Expires"] == ["500"]
    assert "example-secret" not in url.geturl()

    verified = urlparse(storage.R2().signed_url(
        "worldsmith/cache/wheels/pymunk/wheelhouse.tar",
        expires=500, upload=True, metadata=packaging.VERIFIED_METADATA,
    ))
    assert parse_qs(verified.query)["X-Amz-SignedHeaders"] == ["host;if-none-match;x-amz-meta-worldsmith-verified"]

    monkeypatch.setenv("R2_CREDENTIALS_FILE", str(storage.ROOT / "pyproject.toml"))
    with pytest.raises(storage.StorageError, match="must be outside"):
        storage.R2()


@pytest.mark.parametrize("fault, message", [
    ("controller", "Could not load controller"),
    ("simulation", "Simulation execution failed"),
    ("runtime", "Colab session disappeared"),
    ("command", "Colab remote evaluation command did not complete"),
    ("transfer", "R2 remote artifact upload failed"),
    ("download", "R2 download failed"),
    ("corrupt", "Colab returned invalid evaluation artifacts"),
    ("overlap", "Another Colab evaluation is in progress"),
    ("timeout", "Evaluation timed out"),
])
def test_failures_are_distinct_and_cannot_complete(fake_cli, researcher, monkeypatch, fault, message):
    _, objects, _ = fake_cli
    session.start_session()
    if fault == "controller":
        researcher.write_text("raise ValueError('private controller diagnostic')\n")
    elif fault == "simulation":
        monkeypatch.setattr(remote_job, "run_group_process", lambda *a, **kw: subprocess.CompletedProcess([], 3))
    elif fault == "timeout":
        def expire(command, *, timeout, **kwargs):
            raise subprocess.TimeoutExpired(command, timeout)
        monkeypatch.setattr(remote_job, "run_group_process", expire)
    elif fault in {"runtime", "command"}:
        monkeypatch.setattr(session.ColabSession, "execute", lambda *a, **kw: "No completion marker")
        if fault == "runtime":
            monkeypatch.setattr(session.ColabSession, "is_assigned", lambda self: False)
    elif fault == "transfer":
        def abort(*a, **kw):
            raise OSError("private-signature")
        monkeypatch.setattr(remote_job, "urlopen", abort)
    elif fault == "download":
        storage = backend.R2
        def abort(self, key, destination):
            raise backend.StorageError("R2 download failed")
        monkeypatch.setattr(storage, "download", abort)
    elif fault == "corrupt":
        monkeypatch.setattr(backend.R2, "download", lambda self, key, destination: destination.write_bytes(b"partial"))
    lock = session.session_lock("evaluation") if fault == "overlap" else nullcontext()
    with lock, pytest.raises(RuntimeError, match=message) as error:
        research_mcp.evaluate_controller()
    assert "private" not in str(error.value)
    assert research_mcp.latest_feedback is None
    assert all(not key.endswith("completed.json") for key in objects)
    [run] = (researcher.parent / "runs").iterdir()
    status = json.loads((run / "evaluation.json").read_text())
    assert status["status"] == ("timed_out" if fault == "timeout" else "failed")
    assert not (run / "results").exists()


def test_interrupted_evaluation_keeps_r2_inputs_without_a_completion_marker(fake_cli, researcher, monkeypatch):
    _, objects, _ = fake_cli
    session.start_session()

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(session.ColabSession, "execute", interrupt)
    with pytest.raises(KeyboardInterrupt):
        research_mcp.evaluate_controller()
    assert any(key.endswith("input/request.json") for key in objects)
    assert all("/output/" not in key for key in objects)
    assert research_mcp.latest_feedback is None


@pytest.mark.parametrize("reply", ["failed", "timed out", "lost", "controller error"])
def test_recovers_result_or_controller_diagnostics_after_lost_reply(fake_cli, researcher, monkeypatch, reply):
    calls, objects, requests = fake_cli
    session.start_session()
    execute = session.ColabSession.execute
    if reply == "controller error":
        researcher.write_text("raise RuntimeError('private candidate import failure')\n")

    def lose_reply(self, code, *, timeout):
        execute(self, code, timeout=timeout)
        if reply == "lost":
            return ""
        raise session.ColabError(f"Colab exec {reply}")

    monkeypatch.setattr(session.ColabSession, "execute", lose_reply)
    if reply == "controller error":
        with pytest.raises(RuntimeError, match="Could not load controller"):
            research_mcp.evaluate_controller()
        assert all(not key.endswith("completed.json") for key in objects)
    else:
        result = research_mcp.evaluate_controller()
        completed = f"{backend.PREFIX}/{result['run_id']}/output/completed.json"
        assert json.loads(objects[completed])["seeds"] == list(range(8))
    assert len(requests) == 1
    assert [action for _, action, _ in calls].count("exec") == 2  # Setup and one job.


@pytest.mark.parametrize("fault", [
    "job_id", "seeds", "groups", "missing trace", "trace seeds", "trace body", "truncated gzip",
])
def test_recovery_rejects_mismatched_or_incomplete_artifacts(fake_cli, researcher, tmp_path, monkeypatch, fault):
    _, objects, requests = fake_cli
    session.start_session()
    execute = session.ColabSession.execute

    def damage_output(self, code, *, timeout):
        execute(self, code, timeout=timeout)
        key = next(key for key in objects if key.endswith("artifacts.tar.gz"))
        if fault == "truncated gzip":
            objects[key] = objects[key][:-4]
            return ""
        artifacts = tmp_path / "damaged"
        with tarfile.open(fileobj=BytesIO(objects[key])) as archive:
            archive.extractall(artifacts, filter="data")
        manifest_file = artifacts / "manifest.json"
        trace = artifacts / "results/pymunk/navigation/episodes.jsonl"
        if fault == "missing trace":
            trace.unlink()
        elif fault == "trace seeds":
            first, *rest = trace.read_text().splitlines()
            trace.write_text("\n".join([json.dumps({**json.loads(first), "seed": 8}), *rest]) + "\n")
        elif fault == "trace body":
            first, *rest = trace.read_text().splitlines()
            trace.write_text("\n".join([json.dumps({**json.loads(first), "steps": []}), *rest]) + "\n")
        else:
            manifest = json.loads(manifest_file.read_text())
            del manifest[fault]
            if fault == "job_id":
                manifest["job_id"] = "different-job"
            manifest_file.write_text(json.dumps(manifest))
        modified = BytesIO()
        with tarfile.open(fileobj=modified, mode="w:gz") as archive:
            archive.add(artifacts, arcname=".")
        objects[key] = modified.getvalue()
        raise session.ColabError("Colab lost its response")

    monkeypatch.setattr(session.ColabSession, "execute", damage_output)
    with pytest.raises(RuntimeError, match="invalid evaluation artifacts"):
        research_mcp.evaluate_controller()
    assert len(requests) == 1
    assert all(not key.endswith("completed.json") for key in objects)


def test_missing_artifacts_after_exec_failure_are_reported_without_resubmission(fake_cli, researcher, monkeypatch):
    _, objects, requests = fake_cli
    session.start_session()

    def fail(*a, **kw):
        requests.append("submitted")
        raise session.ColabError("Colab exec timed out")

    monkeypatch.setattr(session.ColabSession, "execute", fail)
    with pytest.raises(RuntimeError, match="Colab exec timed out.*R2 has no artifact bundle"):
        research_mcp.evaluate_controller()
    assert requests == ["submitted"]
    assert all("/output/" not in key for key in objects)


def test_remote_job_id_cannot_be_started_twice(fake_cli, researcher, tmp_path, monkeypatch):
    _, objects, _ = fake_cli
    session.start_session()
    result = research_mcp.evaluate_controller()
    job_id = result["run_id"]
    staging = tmp_path / "vm"
    (staging / f"{job_id}.json").write_text("{}")
    (staging / f"{job_id}.tar.gz").write_bytes(objects[f"{backend.PREFIX}/{job_id}/input/candidate.tar.gz"])

    monkeypatch.setattr(remote_job, "evaluate", lambda *a: pytest.fail("duplicate evaluation executed"))
    assert remote_job.run_job(job_id, 10, staging) == remote_job.JOB_ALREADY_STARTED


def test_failed_initialization_stops_session_without_launching_research(fake_cli, monkeypatch):
    calls, _, _ = fake_cli
    monkeypatch.setattr(session.ColabSession, "execute", lambda *a, **kw: "WORLDSMITH_SETUP GPU mismatch")
    monkeypatch.setattr(owner.subprocess, "Popen", lambda *a, **kw: pytest.fail("research started"))
    with pytest.raises(session.ColabError, match="GPU mismatch"):
        owner.run_research("all")
    assert [action for _, action, _ in calls] == ["new", "upload", "stop"]
    assert session.tracked_session() is None


@pytest.mark.parametrize("status, interrupt, cleanup_fails", [
    (0, False, False), (7, False, False), (7, False, True), (0, False, True), (0, True, False),
])
def test_owner_shutdown_preserves_exit_status(fake_cli, tmp_path, monkeypatch, capsys, status, interrupt, cleanup_fails):
    calls, _, _ = fake_cli
    real_start = owner.start_session
    def start(selector):
        runtime = real_start(selector)
        if cleanup_fails:
            def fail_stop(self):
                raise session.ColabError("Could not stop Colab")
            monkeypatch.setattr(session.ColabSession, "stop", fail_stop)
        return runtime
    monkeypatch.setattr(owner, "start_session", start)
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    script = root / "scripts/run-autoresearch.sh"
    if interrupt:
        script.write_text('#!/bin/sh\nkill -TERM "$PPID"\nsleep 30\n')
    else:
        script.write_text(f'#!/bin/sh\ntest -n "$WORLDSMITH_COLAB_SESSION" || exit 99\nexit {status}\n')
    script.chmod(0o700)
    monkeypatch.setattr(owner, "ROOT", root)
    if interrupt:
        previous = signal.signal(signal.SIGTERM, owner.handle_interrupt)
        try:
            with pytest.raises(SystemExit) as exit_info:
                owner.run_research("all")
            assert exit_info.value.code == 143
        finally:
            signal.signal(signal.SIGTERM, previous)
    else:
        assert owner.run_research("all") == (status or int(cleanup_fails))
    if cleanup_fails:
        assert "Retry cleanup with make colab-stop" in capsys.readouterr().err
        assert session.tracked_session()["ready"]
    else:
        assert calls[-1][1] == "stop"
        assert session.tracked_session() is None

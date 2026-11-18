"""Linux simulator preparation, cache identity, and offline installs."""

import base64
import json
import runpy
import tarfile
from contextlib import nullcontext
from hashlib import md5
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from worldsmith.colab import environment_cache, session

PREPARED_OBJECT = {
    "Metadata": environment_cache.VERIFIED_METADATA,
    "ContentLength": 42,
    "ETag": f'"{"a" * 32}"',
}


def test_project_and_lock_changes_invalidate_the_wheelhouse(tmp_path, monkeypatch):
    project = tmp_path / "pyproject.toml"
    lock = tmp_path / "uv.lock"
    project.write_text("[project]\n")
    lock.write_text("# first resolution\n")
    monkeypatch.setattr(environment_cache, "ROOT", tmp_path)
    first = environment_cache.runtime_cache_keys("pymunk")
    lock.write_text("# second resolution\n")
    second = environment_cache.runtime_cache_keys("pymunk")
    assert first["tools"] == second["tools"]
    assert first["wheelhouses"] != second["wheelhouses"]
    project.write_text("[project]\nname = 'changed'\n")
    assert environment_cache.runtime_cache_keys("pymunk")["wheelhouses"] != second["wheelhouses"]


@pytest.mark.parametrize("failure", [None, "setup", "interrupt", "missing after success"])
def test_owner_builds_selected_caches_and_releases_temporary_session(tmp_path, monkeypatch, failure):
    keys = environment_cache.runtime_cache_keys("maniskill")
    tool_key = next(iter(keys["tools"].values()))
    prepared = {tool_key}
    actions = []

    class Storage:
        def check(self):
            pass

        def head(self, key):
            return PREPARED_OBJECT if key in prepared else None

        def signed_url(self, key, *, expires, upload=False, metadata=None):
            assert metadata == (environment_cache.VERIFIED_METADATA if upload else None)
            return key

    def _command(self, action, *args, **kwargs):
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
    monkeypatch.setattr(session.ColabSession, "_command", _command)
    monkeypatch.setattr(session.subprocess, "run", lambda *a, **kw: pytest.fail("built Worldsmith wheel"))

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
    main = runpy.run_path(str(environment_cache.ROOT / "scripts/setup-colab.py"))["main"]
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
    assert environment_cache.RUNTIME_CACHE_TARGET == "colab-linux-x86_64-l4-cuda"
    l4_keys = environment_cache.runtime_cache_keys("all")
    monkeypatch.setattr(environment_cache, "RUNTIME_CACHE_TARGET", "colab-linux-x86_64-cpu")
    cpu_keys = environment_cache.runtime_cache_keys("all")
    for family, entries in l4_keys.items():
        assert set(entries.values()).isdisjoint(cpu_keys[family].values())


def test_preparation_plan_skips_other_simulators_with_complete_caches():
    keys = environment_cache.runtime_cache_keys("all")
    missing_asset = keys["assets"]["libero"]

    class Storage:
        def head(self, key):
            return None if key == missing_asset else PREPARED_OBJECT

        def signed_url(self, key, **kwargs):
            return key

    config, missing = environment_cache.plan_runtime(Storage(), "all", prepare=True, expires=600)
    assert missing == [missing_asset]
    assert set(config["assets"]) == {"libero"}
    assert set(config["wheelhouses"]) == {"libero"}
    assert config["wheelhouses"]["libero"]["get"] == keys["wheelhouses"]["libero"]
    assert set(config["tools"]) == {environment_cache.ENVIRONMENT_PYTHONS["libero"]}


def test_tool_preparation_uses_uv_directly_from_the_wheel(tmp_path, monkeypatch):
    script = runpy.run_path(str(environment_cache.ROOT / "scripts/setup-colab.py"))
    install = script["install_tools"]
    version = environment_cache.ENVIRONMENT_PYTHONS["pymunk"]
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
        {"uv_version": environment_cache.UV_VERSION, "tools": {version: {"put": "cache"}}},
        staging, prepare=True,
    )
    assert Path(tools[version][0]).read_bytes() == b"uv executable"


def test_maniskill_preparation_publishes_assets_under_the_data_directory(tmp_path, monkeypatch):
    install = runpy.run_path(str(environment_cache.ROOT / "scripts/setup-colab.py"))["install_assets"]
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
@pytest.mark.parametrize("group", ["pymunk", "maniskill", "libero", "robosuite"])
def test_wheelhouse_installs_offline_before_publication(tmp_path, monkeypatch, prepare, group):
    script = runpy.run_path(str(environment_cache.ROOT / "scripts/setup-colab.py"))
    install = script["install_environment"]
    environment = install.__globals__
    commands = []
    wheel = "pymunk-example-linux_x86_64.whl"

    def stage(name, *args):
        commands.append(args)
        if "export" in args:
            assert "--offline" in args and "--frozen" in args
            if group == "pymunk":
                assert args[-2:] == ("--only-group", "simulation")
            else:
                assert "--only-group" not in args
            Path(args[args.index("--output-file") + 1]).write_text("pymunk==7.3.1\n")
        if "wheel" in args:
            assert "--no-deps" in args
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
            install(group, argument, tools, tmp_path, prepare=True)
        assert any("wheel" in command for command in commands)
        assert any("check" in command for command in commands)
    else:
        install(group, argument, tools, tmp_path, prepare=False)
        assert all("--offline" in command for command in commands)


def test_cache_publication_attaches_the_checksum_and_rejects_a_bad_reply(tmp_path, monkeypatch):
    script = runpy.run_path(str(environment_cache.ROOT / "scripts/setup-colab.py"))
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

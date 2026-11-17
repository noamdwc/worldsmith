"""Prepare the Worldsmith wheel and the small launch bundle for Colab."""

import json
import os
import re
import shutil
import subprocess
import tarfile
import tomllib
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from worldsmith.research_lifecycle import ROOT

from .storage import StorageError

CACHE_PREFIX = "worldsmith/cache"
RUNTIME_CACHE_TARGET = "colab-linux-x86_64-l4-cuda"
REQUIREMENTS = Path("simulation_environments/colab")
LIBERO = Path("simulation_environments/libero")
ENVIRONMENT_PYTHONS = {
    "pymunk": "3.12.13", "maniskill": "3.11.15", "libero": "3.11.15", "robosuite": "3.12.13",
}
UV_VERSION = "0.11.15"
VERIFIED_METADATA = {"worldsmith-verified": "1"}
SETUP_FILES = (
    Path("scripts/setup-colab.py"), LIBERO / "setup.sh", LIBERO / "assets.sha256",
    *(REQUIREMENTS / f"{group}.txt" for group in ENVIRONMENT_PYTHONS),
)


def content_key(paths: list[Path], *, format_id: str) -> str:
    digest = sha256(format_id.encode())
    for path in sorted(paths):
        digest.update(path.as_posix().encode() + b"\0")
        digest.update(sha256((ROOT / path).read_bytes()).digest())
    return digest.hexdigest()


def prepare_wheel(storage, *, expires: int) -> dict:
    paths = [path.relative_to(ROOT) for path in (ROOT / "worldsmith").rglob("*.py")]
    paths += [Path("pyproject.toml"), Path("worldsmith/research_baseline/JOURNAL.md")]
    digest = content_key(paths, format_id="python-wheel-v1")
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    filename = f"worldsmith-{version}-py3-none-any.whl"
    key = f"{CACHE_PREFIX}/package/{digest}/{filename}"
    if not storage.exists(key):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "source"
            for path in paths:
                (source / path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / path, source / path)
            subprocess.run(
                ["uv", "build", "--wheel", "--out-dir", str(directory), str(source)],
                check=True, env={**os.environ, "SOURCE_DATE_EPOCH": "315532800"},
            )
            storage.put(key, directory / filename, cache=True)
    return {"filename": filename, "url": storage.signed_url(key, expires=expires)}


def runtime_cache_keys(selector: str) -> dict:
    groups = tuple(ENVIRONMENT_PYTHONS) if selector == "all" else (selector,)
    keys = {"tools": {}, "wheelhouses": {}, "assets": {}}
    for group in groups:
        python_version = ENVIRONMENT_PYTHONS[group]
        tool_digest = sha256(f"toolchain-v1-{RUNTIME_CACHE_TARGET}\0{UV_VERSION}\0{python_version}".encode()).hexdigest()
        keys["tools"][python_version] = f"{CACHE_PREFIX}/tools/{tool_digest}.tar.gz"
        digest = content_key(
            [REQUIREMENTS / f"{group}.txt"],
            format_id=f"wheelhouse-v2-{RUNTIME_CACHE_TARGET}-cp{python_version}",
        )
        keys["wheelhouses"][group] = f"{CACHE_PREFIX}/wheels/{group}/{digest}.tar"

    if "maniskill" in groups:
        maniskill_pin = next(
            line for line in (ROOT / REQUIREMENTS / "maniskill.txt").read_text().splitlines()
            if line.startswith("mani-skill-nightly==")
        )
        digest = sha256(f"go2-assets-v2-{RUNTIME_CACHE_TARGET}\0{maniskill_pin}\0unitree_go2".encode()).hexdigest()
        keys["assets"]["maniskill"] = f"{CACHE_PREFIX}/assets/maniskill/{digest}.tar.gz"
    if "libero" in groups:
        digest = content_key(
            [LIBERO / "setup.sh", LIBERO / "assets.sha256"], format_id=f"libero-assets-v2-{RUNTIME_CACHE_TARGET}",
        )
        keys["assets"]["libero"] = f"{CACHE_PREFIX}/assets/libero/{digest}.tar.gz"
    return keys


def prepared_etag(storage, key: str) -> str | None:
    metadata = storage.head(key)
    if metadata is None:
        return None
    etag = metadata.get("ETag", "").strip('"')
    if (
        metadata.get("Metadata") != VERIFIED_METADATA
        or metadata.get("ContentLength", 0) <= 0
        or not re.fullmatch(r"[0-9a-f]{32}", etag)
    ):
        raise StorageError(f"Invalid prepared cache at {key}. Remove that object and run make colab-build-envs")
    return etag


def plan_runtime(storage, selector: str, *, prepare: bool, expires: int):
    keys = runtime_cache_keys(selector)
    etags = {
        key: prepared_etag(storage, key)
        for family in keys.values() for key in family.values()
    }
    missing = [key for key, etag in etags.items() if etag is None]
    if not prepare and missing:
        names = [
            f"{name} {family}" for family, entries in keys.items() for name, key in entries.items()
            if key in missing
        ]
        raise StorageError(f"Missing prepared Colab caches ({', '.join(names)}). Run make colab-build-envs")
    if prepare and not missing:
        return None, missing

    if prepare:
        keys["assets"] = {group: key for group, key in keys["assets"].items() if key in missing}
        keys["wheelhouses"] = {
            group: key for group, key in keys["wheelhouses"].items()
            if key in missing or group in keys["assets"]
        }
        versions = {ENVIRONMENT_PYTHONS[group] for group in keys["wheelhouses"]}
        keys["tools"] = {
            version: key for version, key in keys["tools"].items()
            if key in missing or version in versions
        }

    config = {"mode": "prepare" if prepare else "research", "uv_version": UV_VERSION}
    for family, entries in keys.items():
        config[family] = {}
        for name, key in entries.items():
            if etags[key] is None:
                url = storage.signed_url(key, expires=expires, upload=True, metadata=VERIFIED_METADATA)
                entry = {"put": url, "headers": {f"x-amz-meta-{k}": v for k, v in VERIFIED_METADATA.items()}}
            else:
                entry = {"get": storage.signed_url(key, expires=expires), "etag": etags[key]}
            if family == "wheelhouses":
                entry["python"] = ENVIRONMENT_PYTHONS[name]
            config[family][name] = entry
    return config, missing


def package_bootstrap(config: dict, archive_path: Path):
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in SETUP_FILES:
            archive.add(ROOT / path, arcname=path.as_posix())
        config_path = archive_path.parent / "setup.json"
        config_path.write_text(json.dumps(config))
        archive.add(config_path, arcname="setup.json")
        config_path.unlink()

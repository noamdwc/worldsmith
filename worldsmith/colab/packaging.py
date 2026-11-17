"""Prepare the Worldsmith wheel and the small launch bundle for Colab."""

import json
import os
import shutil
import subprocess
import tarfile
import tomllib
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from worldsmith.research_lifecycle import ROOT

CACHE_PREFIX = "worldsmith/cache"
REQUIREMENTS = Path("simulation_environments/colab")
LIBERO = Path("simulation_environments/libero")
SETUP_FILES = (
    Path("scripts/setup-colab.py"), LIBERO / "setup.sh", LIBERO / "assets.sha256",
    *(REQUIREMENTS / f"{group}.txt" for group in ("pymunk", "maniskill", "libero", "robosuite")),
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


def cache_access(storage, family: str, digest: str, *, expires: int, extension: str) -> dict:
    key = f"{CACHE_PREFIX}/{family}/{digest}{extension}"
    hit = storage.exists(key)
    url = storage.signed_url(key, expires=expires, upload=not hit)
    return {"get" if hit else "put": url}


def package_bootstrap(storage, archive_path: Path, *, expires: int):
    config = {"wheel": prepare_wheel(storage, expires=expires), "wheelhouses": {}, "assets": {}}
    for group, python_version in (("pymunk", "3.12"), ("maniskill", "3.11"), ("libero", "3.11"), ("robosuite", "3.12")):
        pins = REQUIREMENTS / f"{group}.txt"
        digest = content_key([pins], format_id=f"wheelhouse-v1-colab-linux-x86_64-cp{python_version}")
        config["wheelhouses"][group] = {
            "python": python_version,
            **cache_access(storage, f"wheels/{group}", digest, expires=expires, extension=".tar"),
        }
    # Only the asset downloader's pinned release and asset ID select the Go2 bundle.
    maniskill_pin = next(
        line for line in (ROOT / REQUIREMENTS / "maniskill.txt").read_text().splitlines()
        if line.startswith("mani-skill-nightly==")
    )
    maniskill_digest = sha256(f"go2-assets-v1\0{maniskill_pin}\0unitree_go2".encode()).hexdigest()
    libero_digest = content_key([LIBERO / "setup.sh", LIBERO / "assets.sha256"], format_id="libero-assets-v1")
    for group, digest in (("maniskill", maniskill_digest), ("libero", libero_digest)):
        config["assets"][group] = cache_access(
            storage, f"assets/{group}", digest, expires=expires, extension=".tar.gz",
        )

    with tarfile.open(archive_path, "w:gz") as archive:
        for path in SETUP_FILES:
            archive.add(ROOT / path, arcname=path.as_posix())
        config_path = archive_path.parent / "setup.json"
        config_path.write_text(json.dumps(config))
        archive.add(config_path, arcname="setup.json")
        config_path.unlink()
